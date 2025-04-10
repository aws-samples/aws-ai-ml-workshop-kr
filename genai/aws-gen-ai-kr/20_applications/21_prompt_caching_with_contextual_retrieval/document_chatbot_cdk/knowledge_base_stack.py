from aws_cdk import (
    Stack,
    aws_opensearchserverless as opensearchserverless,
    aws_s3 as s3,
    aws_lambda as lambda_,
    aws_bedrock as bedrock,
    RemovalPolicy,
    CustomResource,
    CfnOutput,
    Duration,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_iam as iam,
    CfnDeletionPolicy
)
from aws_cdk.custom_resources import Provider
from constructs import Construct
import json
from datetime import datetime
from typing import TypedDict


class KnowledgebaseStackOutputs(TypedDict):
    opensearch_endpoint: str
    index_name: str
    knowledgebase_id: str
    document_cloudfront_url: str
    data_source_id: str
    document_bucket_name: str


class KnowledgebaseStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create S3 bucket for document uploads (no auto-deployment from pdf_docs)
        document_bucket = s3.Bucket(
            self, "DocumentBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True
        )

        supplemental_data_bucket = s3.Bucket(
            self, "SupplementalDataBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True
        )

        # Create Origin Access Identity for CloudFront
        origin_access_identity = cloudfront.OriginAccessIdentity(
            self, 'DocumentBucketOAI',
            comment=f'OAI for accessing document bucket {document_bucket.bucket_name}'
        )

        # Grant read permissions to CloudFront
        document_bucket.grant_read(origin_access_identity)

        # Create CloudFront distribution for document access
        document_distribution = cloudfront.Distribution(
            self, 'DocumentDistribution',
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3Origin(
                    document_bucket,
                    origin_access_identity=origin_access_identity
                ),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                allowed_methods=cloudfront.AllowedMethods.ALLOW_GET_HEAD,
                cached_methods=cloudfront.CachedMethods.CACHE_GET_HEAD,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
                origin_request_policy=cloudfront.OriginRequestPolicy.CORS_S3_ORIGIN,
                response_headers_policy=cloudfront.ResponseHeadersPolicy.CORS_ALLOW_ALL_ORIGINS,
            ),
            enable_logging=True,
            price_class=cloudfront.PriceClass.PRICE_CLASS_100
        )

        # Define a valid collection name
        collection_name = f"collection-{self.account}"
        contextual_retrieval_index_name = "contextual_retrieval_text"
        knowledgebase_index_name = "knowledgebase"

        # Lambda role with necessary permissions for index initializer
        lambda_role = iam.Role(
            self, "IndexInitializerRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com")
        )

        lambda_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
        )

        lambda_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "aoss:APIAccessAll",
                    "aoss:List*",
                    "aoss:Create*",
                    "aoss:Update*",
                    "aoss:Delete*",
                ],
                resources=["*"]
            )
        )

        # 1. Create Encryption Policy (required before collection)
        encryption_policy = opensearchserverless.CfnSecurityPolicy(
            self, "CollectionEncryptionPolicy",
            name=f"encryption-policy",
            type="encryption",
            policy=json.dumps({
                "Rules": [{
                    "ResourceType": "collection",
                    "Resource": [f"collection/{collection_name}"]
                }],
                "AWSOwnedKey": True
            })
        )

        encryption_policy.cfn_options.deletion_policy = CfnDeletionPolicy.DELETE

        # 2. Create Network Policy
        network_policy = opensearchserverless.CfnSecurityPolicy(
            self, "CollectionNetworkPolicy",
            name=f"network-policy",
            type="network",
            policy=json.dumps([{
                "Rules": [{
                    "ResourceType": "collection",
                    "Resource": [f"collection/{collection_name}"]
                }, {
                    "ResourceType": "dashboard",
                    "Resource": [f"collection/{collection_name}"]
                }],
                "AllowFromPublic": True  # For demo purposes only
            }])
        )

        network_policy.cfn_options.deletion_policy = CfnDeletionPolicy.DELETE

        # 3. Create OpenSearch Serverless Collection
        collection = opensearchserverless.CfnCollection(
            self, "DocumentsSearchCollection",
            name=collection_name,
            type="VECTORSEARCH",
            description="Collection for contextual retrieval"
        )

        collection.cfn_options.deletion_policy = CfnDeletionPolicy.DELETE

        # Add dependencies to ensure proper creation order
        collection.add_dependency(encryption_policy)
        collection.add_dependency(network_policy)

        # 4. Create Data Access Policy with proper format
        data_access_policy = opensearchserverless.CfnAccessPolicy(
            self, "CollectionAccessPolicy",
            name=f"data-access-policy",
            type="data",
            policy=json.dumps([{
                "Rules": [
                    {
                        "ResourceType": "index",
                        "Resource": [f"index/{collection_name}/*"],
                        "Permission": ["aoss:*"]
                    },
                    {
                        "ResourceType": "collection",
                        "Resource": [f"collection/{collection_name}"],
                        "Permission": ["aoss:*"]
                    }
                ],
                "Principal": [
                    lambda_role.role_arn,
                    f"arn:aws:iam::{self.account}:root"
                ]
            }])
        )

        # Add dependency to ensure collection exists before access policy
        data_access_policy.add_dependency(collection)
        data_access_policy.add_dependency(encryption_policy)
        data_access_policy.add_dependency(network_policy)
        data_access_policy.cfn_options.deletion_policy = CfnDeletionPolicy.DELETE

        # Create Lambda layer for requests
        requests_layer = lambda_.LayerVersion(
            self, "RequestsLayer",
            code=lambda_.Code.from_asset("./layers/requests.zip"),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_9],
            description="Layer containing the requests module"
        )

        # Create index initializer Lambda
        index_initializer = lambda_.Function(
            self, "IndexInitializerFunction",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="index_initializer.handler",
            code=lambda_.Code.from_asset("./lambda/knowledge_base"),
            environment={
                "COLLECTION_ENDPOINT": f"{collection.attr_collection_endpoint}",
                "CR_INDEX_NAME": contextual_retrieval_index_name,
                "KB_INDEX_NAME": knowledgebase_index_name,
                "REGION": self.region
            },
            timeout=Duration.minutes(5),
            layers=[requests_layer],
            role=lambda_role
        )

        # Ensure Lambda waits for access policy to be created
        index_initializer.node.add_dependency(data_access_policy)
        index_initializer.node.add_dependency(collection)

        # Create a custom resource that triggers the index initializer
        index_init_trigger = CustomResource(
            self, "IndexInitTrigger",
            service_token=Provider(
                self, "IndexInitProvider",
                on_event_handler=index_initializer
            ).service_token
        )

        # Add dependencies
        index_init_trigger.node.add_dependency(collection)
        index_init_trigger.node.add_dependency(data_access_policy)

        # Create IAM role for Bedrock Knowledge Base
        knowledge_base_role = iam.Role(
            self, 'KnowledgeBaseRole',
            assumed_by=iam.ServicePrincipal('bedrock.amazonaws.com'),
        )

        # Add S3 permissions to the role
        knowledge_base_role.add_to_policy(iam.PolicyStatement(
            actions=[
                's3:GetObject',
                's3:ListBucket',
                's3:PutObject',
                's3:DeleteObject'
            ],
            resources=[
                document_bucket.bucket_arn,
                f'{document_bucket.bucket_arn}/*',
                supplemental_data_bucket.bucket_arn,
                f'{supplemental_data_bucket.bucket_arn}/*'
            ],
        ))

        # Add OpenSearch permissions to the role
        knowledge_base_role.add_to_policy(iam.PolicyStatement(
            actions=[
                'aoss:APIAccessAll'
            ],
            resources=[collection.attr_arn],
        ))

        # Add Bedrock permissions to the role
        knowledge_base_role.add_to_policy(iam.PolicyStatement(
            actions=[
                'bedrock:*'
            ],
            resources=['*'],
        ))

        # Create Data Access Policy for Bedrock Knowledge Base
        bedrock_data_access_policy = opensearchserverless.CfnAccessPolicy(
            self, 'BedrockDataAccessPolicy',
            name='bedrock-access-policy',
            type='data',
            description='Data access policy for development',
            policy=json.dumps([
                {
                    'Rules': [
                        {
                            'ResourceType': 'collection',
                            'Resource': [f'collection/{collection_name}'],
                            'Permission': [
                                'aoss:CreateCollectionItems',
                                'aoss:DeleteCollectionItems',
                                'aoss:UpdateCollectionItems',
                                'aoss:DescribeCollectionItems',
                                'aoss:*'
                            ]
                        },
                        {
                            'ResourceType': 'index',
                            'Resource': [f"index/{collection_name}/*"],
                            'Permission': [
                                'aoss:CreateIndex',
                                'aoss:DeleteIndex',
                                'aoss:UpdateIndex',
                                'aoss:DescribeIndex',
                                'aoss:ReadDocument',
                                'aoss:WriteDocument',
                                'aoss:*'
                            ]
                        }
                    ],
                    'Principal': [
                        knowledge_base_role.role_arn,
                        f'arn:aws:iam::{self.account}:root'
                    ],
                    'Description': 'Combined access policy for both collection and index operations'
                }
            ])
        )

        # Add dependencies
        bedrock_data_access_policy.add_dependency(collection)
        bedrock_data_access_policy.cfn_options.deletion_policy = CfnDeletionPolicy.DELETE

        # Create Knowledge Base
        knowledgebase_name = 'knowledge-base'
        knowledge_base = bedrock.CfnKnowledgeBase(
            self, 'BedrockKnowledgeBase',
            name=knowledgebase_name,
            role_arn=knowledge_base_role.role_arn,
            knowledge_base_configuration=bedrock.CfnKnowledgeBase.KnowledgeBaseConfigurationProperty(
                type='VECTOR',
                vector_knowledge_base_configuration=bedrock.CfnKnowledgeBase.VectorKnowledgeBaseConfigurationProperty(
                    embedding_model_arn=f'arn:aws:bedrock:{self.region}::foundation-model/amazon.titan-embed-text-v2:0',
                    supplemental_data_storage_configuration=bedrock.CfnKnowledgeBase.SupplementalDataStorageConfigurationProperty(
                        supplemental_data_storage_locations=[
                            bedrock.CfnKnowledgeBase.SupplementalDataStorageLocationProperty(
                                supplemental_data_storage_location_type="S3",
                                s3_location=bedrock.CfnKnowledgeBase.S3LocationProperty(
                                    uri=f"s3://{supplemental_data_bucket.bucket_name}"
                                )
                            )]
                    )
                )
            ),
            storage_configuration=bedrock.CfnKnowledgeBase.StorageConfigurationProperty(
                type='OPENSEARCH_SERVERLESS',
                opensearch_serverless_configuration=bedrock.CfnKnowledgeBase.OpenSearchServerlessConfigurationProperty(
                    collection_arn=collection.attr_arn,
                    field_mapping=bedrock.CfnKnowledgeBase.OpenSearchServerlessFieldMappingProperty(
                        metadata_field='metadata',
                        text_field='content',
                        vector_field='content_embedding',
                    ),
                    vector_index_name=knowledgebase_index_name,
                ),
            ),
        )

        knowledge_base.node.add_dependency(index_initializer)
        knowledge_base.node.add_dependency(index_init_trigger)
        knowledge_base.node.add_dependency(bedrock_data_access_policy)

        # Create data source that points to our document bucket
        data_source = bedrock.CfnDataSource(
            self, 'BedrockDataSource',
            data_source_configuration=bedrock.CfnDataSource.DataSourceConfigurationProperty(
                s3_configuration=bedrock.CfnDataSource.S3DataSourceConfigurationProperty(
                    bucket_arn=document_bucket.bucket_arn,
                ),
                type='S3'
            ),
            knowledge_base_id=knowledge_base.attr_knowledge_base_id,
            name='document-datasource',
            description='Data source for uploaded PDF documents',
            data_deletion_policy='RETAIN',
            vector_ingestion_configuration=bedrock.CfnDataSource.VectorIngestionConfigurationProperty(
                chunking_configuration=bedrock.CfnDataSource.ChunkingConfigurationProperty(
                    chunking_strategy='HIERARCHICAL',
                    hierarchical_chunking_configuration=bedrock.CfnDataSource.HierarchicalChunkingConfigurationProperty(
                        level_configurations=[
                            bedrock.CfnDataSource.HierarchicalChunkingLevelConfigurationProperty(
                                max_tokens=1000
                            ),
                            bedrock.CfnDataSource.HierarchicalChunkingLevelConfigurationProperty(
                                max_tokens=200
                            )
                        ],
                        overlap_tokens=60
                    )
                ),
                parsing_configuration=bedrock.CfnDataSource.ParsingConfigurationProperty(
                    parsing_strategy='BEDROCK_FOUNDATION_MODEL',
                    bedrock_foundation_model_configuration=bedrock.CfnDataSource.BedrockFoundationModelConfigurationProperty(
                        model_arn=f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
                        parsing_modality="MULTIMODAL"
                    )
                )
            )
        )

        data_source.add_dependency(knowledge_base)

        # Store outputs
        self.outputs: KnowledgebaseStackOutputs = {
            "opensearch_endpoint": collection.attr_collection_endpoint,
            "index_name": contextual_retrieval_index_name,
            "knowledgebase_id": knowledge_base.attr_knowledge_base_id,
            "document_cloudfront_url": document_distribution.distribution_domain_name,
            "data_source_id": data_source.attr_data_source_id,
            "document_bucket_name": document_bucket.bucket_name
        }

        # Output the collection endpoint, bucket names, and related infrastructure
        CfnOutput(self, "CollectionEndpoint", value=collection.attr_collection_endpoint)
        CfnOutput(self, "DataBucketName", value=document_bucket.bucket_name)
        CfnOutput(self, "DashboardsURL", value=f"https://{collection.attr_dashboard_endpoint}/_dashboards/")
        CfnOutput(self, "KnowledgeBaseId", value=knowledge_base.attr_knowledge_base_id)
        CfnOutput(self, "DataSourceId", value=data_source.attr_data_source_id)
        CfnOutput(
            self, 'DocumentCloudFrontUrl',
            value=document_distribution.distribution_domain_name,
            description='CloudFront URL for accessing documents',
        )