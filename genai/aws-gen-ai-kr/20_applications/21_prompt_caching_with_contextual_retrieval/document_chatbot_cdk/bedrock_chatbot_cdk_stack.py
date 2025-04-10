import aws_cdk as cdk
from constructs import Construct
from aws_cdk import (
    aws_lambda as lambda_,
    aws_apigatewayv2 as apigatewayv2,
    aws_apigatewayv2_integrations as integrations,
    aws_iam as iam,
    aws_s3 as s3,
    aws_s3_deployment as s3_deployment,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_dynamodb as dynamodb,
    aws_apigateway as apigateway,
    aws_sqs as sqs,
    aws_lambda_event_sources as lambda_event_sources,
    Duration,
    CustomResource
)
from aws_cdk.custom_resources import (
    Provider
)
import os
import json
from datetime import datetime


class BedrockChatbotStack(cdk.Stack):
    def __init__(self, scope: Construct, id: str, kb_id, kb_document_url, kb_outputs, **kwargs):
        super().__init__(scope, id, **kwargs)

        # Extract OpenSearch details and other info from knowledge base outputs
        opensearch_endpoint = kb_outputs.get("opensearch_endpoint")
        opensearch_index = kb_outputs.get("index_name")
        data_source_id = kb_outputs.get("data_source_id")
        document_bucket_name = kb_outputs.get("document_bucket_name")

        # Get the S3 bucket that was created in the knowledge base stack
        document_bucket = s3.Bucket.from_bucket_name(
            self, "ImportedDocumentBucket", document_bucket_name
        )

        # Create DynamoDB table to store WebSocket connections
        connections_table = dynamodb.Table(
            self, 'ConnectionsTable',
            partition_key=dynamodb.Attribute(
                name='connectionId',
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )

        # Create DynamoDB table to track document uploads and processing status
        document_table = dynamodb.Table(
            self, "DocumentTable",
            partition_key=dynamodb.Attribute(
                name="id",
                type=dynamodb.AttributeType.STRING
            ),
            removal_policy=cdk.RemovalPolicy.DESTROY,
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST
        )

        # Create Lambda functions for WebSocket routes
        connect_function = lambda_.Function(
            self, 'ConnectFunction',
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler='connect.handler',
            code=lambda_.Code.from_asset('lambda/websocket'),
            environment={
                'CONNECTIONS_TABLE': connections_table.table_name,
            },
        )

        disconnect_function = lambda_.Function(
            self, 'DisconnectFunction',
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler='disconnect.handler',
            code=lambda_.Code.from_asset('lambda/websocket'),
            environment={
                'CONNECTIONS_TABLE': connections_table.table_name,
            },
        )

        # Create Lambda layer for OpenSearch integration
        opensearch_layer = lambda_.LayerVersion(
            self, 'OpenSearchLayer',
            code=lambda_.Code.from_asset('layers/opensearch.zip'),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
            description='Layer containing the OpenSearch SDK'
        )

        # Create Bedrock Lambda function with both KnowledgeBase and OpenSearch access
        bedrock_function = lambda_.Function(
            self, 'BedrockLambdaFunction',
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler='message.handler',
            code=lambda_.Code.from_asset('lambda/websocket'),
            layers=[opensearch_layer],
            environment={
                'KNOWLEDGE_BASE_ID': kb_id,
                'REGION': 'us-west-2',
                'CONNECTIONS_TABLE': connections_table.table_name,
                'OPENSEARCH_ENDPOINT': opensearch_endpoint,
                'OPENSEARCH_INDEX': opensearch_index,
                'RESPONSE_LANGUAGE': 'Korean'
            },
            timeout=Duration.minutes(5),
            memory_size=1024
        )

        # Grant permissions to Lambda functions
        connections_table.grant_read_write_data(connect_function)
        connections_table.grant_read_write_data(disconnect_function)
        connections_table.grant_read_write_data(bedrock_function)

        # Grant Bedrock permissions to Lambda
        bedrock_function.add_to_role_policy(iam.PolicyStatement(
            actions=[
                'bedrock:RetrieveAndGenerate',
                'bedrock:Retrieve',
                'bedrock:InvokeModelWithResponseStream',
                'bedrock:InvokeModel',
                'bedrock:Rerank'
            ],
            resources=['*'],
        ))

        # Grant OpenSearch Serverless permissions
        bedrock_function.add_to_role_policy(iam.PolicyStatement(
            actions=['aoss:APIAccessAll'],
            resources=['*'],
        ))

        # Create WebSocket API
        websocket_api = apigatewayv2.WebSocketApi(
            self, 'BedrockWebSocketAPI',
            connect_route_options=apigatewayv2.WebSocketRouteOptions(
                integration=integrations.WebSocketLambdaIntegration(
                    'ConnectIntegration', connect_function
                )
            ),
            disconnect_route_options=apigatewayv2.WebSocketRouteOptions(
                integration=integrations.WebSocketLambdaIntegration(
                    'DisconnectIntegration', disconnect_function
                )
            ),
            default_route_options=apigatewayv2.WebSocketRouteOptions(
                integration=integrations.WebSocketLambdaIntegration(
                    'MessageIntegration', bedrock_function
                )
            ),
        )

        # Add permissions for Lambda to post to connections
        websocket_stage = apigatewayv2.WebSocketStage(
            self, 'BedrockWebSocketStage',
            web_socket_api=websocket_api,
            stage_name='prod',
            auto_deploy=True,
        )

        # Grant permission for Bedrock Lambda to manage WebSocket connections
        bedrock_function.add_to_role_policy(iam.PolicyStatement(
            actions=['execute-api:ManageConnections'],
            resources=[f'arn:aws:execute-api:{self.region}:{self.account}:{websocket_api.api_id}/*'],
        ))

        # Create S3 bucket for hosting React website
        website_bucket = s3.Bucket(
            self, 'ReactWebsiteBucket',
            removal_policy=cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        # CloudFront Origin Access Identity for S3
        origin_access_identity = cloudfront.OriginAccessIdentity(
            self, 'OAI',
            comment=f'OAI for {id} website'
        )

        # Grant read permissions to CloudFront
        website_bucket.grant_read(origin_access_identity)

        # Create CloudFront distribution
        distribution = cloudfront.Distribution(
            self, 'WebsiteDistribution',
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3Origin(
                    website_bucket,
                    origin_access_identity=origin_access_identity
                ),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
                origin_request_policy=cloudfront.OriginRequestPolicy.CORS_S3_ORIGIN,
            ),
            default_root_object="index.html",
            error_responses=[
                # For SPA routing, redirect all 404s to index.html
                cloudfront.ErrorResponse(
                    http_status=404,
                    response_http_status=200,
                    response_page_path="/index.html",
                )
            ],
        )

        # Create a queue for document processing
        document_processing_queue = sqs.Queue(
            self, "DocumentProcessingQueue",
            visibility_timeout=Duration.minutes(15),  # Match Lambda timeout
            retention_period=Duration.days(14)  # Keep messages for 14 days
        )

        # Create Lambda layers for document processing
        requests_layer = lambda_.LayerVersion(
            self, "RequestsLayer",
            code=lambda_.Code.from_asset("./layers/requests.zip"),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
            description="Layer containing the requests module"
        )

        pdfplumber_layer = lambda_.LayerVersion(
            self, "PdfplumberLayer",
            code=lambda_.Code.from_asset("./layers/pdfplumber_layer.zip"),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
            description="Layer containing pdfplumber and dependencies"
        )

        boto3_layer = lambda_.LayerVersion(
            self, "Boto3plumberLayer",
            code=lambda_.Code.from_asset("./layers/boto3.zip"),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
            description="Layer containing boto3 and dependencies"
        )

        # Use the existing KB sync Lambda with modifications
        kb_sync_lambda = lambda_.Function(
            self, 'KBSyncFunction',
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler='index.handler',
            code=lambda_.Code.from_asset('lambda/kb_sync'),
            environment={
                'KNOWLEDGE_BASE_ID': kb_id,
                'DATA_SOURCE_ID': data_source_id,
                'REGION': self.region,
                'DOCUMENT_TABLE': document_table.table_name
            },
            timeout=Duration.minutes(5)
        )

        # Grant permissions to sync the knowledge base
        kb_sync_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=[
                'bedrock:StartIngestionJob',
                'bedrock:GetIngestionJob',
                'bedrock:ListIngestionJobs',
            ],
            resources=['*'],
        ))

        # Grant access to DynamoDB for updating document status
        document_table.grant_read_write_data(kb_sync_lambda)

        # Lambda role with necessary permissions for document processor
        processor_role = iam.Role(
            self, "ProcessorLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com")
        )

        processor_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
        )

        # Add Bedrock permissions for embedding and context generation
        processor_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock-runtime:InvokeModel",
                    "bedrock-runtime:InvokeModelWithResponseStream"
                ],
                resources=["*"]
            )
        )

        # Add Lambda invoke permissions for the KB sync Lambda
        processor_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "lambda:InvokeFunction"
                ],
                resources=[kb_sync_lambda.function_arn]
            )
        )

        # Add OpenSearch permissions
        processor_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "aoss:APIAccessAll"
                ],
                resources=["*"]
            )
        )

        # Add S3 and DynamoDB permissions
        document_bucket.grant_read_write(processor_role)
        document_table.grant_read_write_data(processor_role)

        # Create document processor Lambda to handle document uploads and dual processing
        document_processor_lambda = lambda_.Function(
            self, "DocumentProcessorFunction",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="document_processor.handler",
            code=lambda_.Code.from_asset("./lambda/document_upload"),
            environment={
                "DOCUMENT_TABLE": document_table.table_name,
                "DOCUMENT_BUCKET": document_bucket.bucket_name,
                "REGION": self.region,
                "KNOWLEDGE_BASE_ID": kb_id,
                "DATA_SOURCE_ID": data_source_id,
                "KB_SYNC_LAMBDA_ARN": kb_sync_lambda.function_arn,
                "COLLECTION_ENDPOINT": opensearch_endpoint,
                "CR_INDEX_NAME": opensearch_index
            },
            timeout=Duration.minutes(15),  # Increased for document processing
            memory_size=1024,  # Increased for document processing
            layers=[requests_layer, pdfplumber_layer, boto3_layer],
            role=processor_role
        )

        # Grant permission to invoke the KB sync Lambda
        kb_sync_lambda.grant_invoke(document_processor_lambda)

        # Add SQS as event source for document processor
        document_processor_lambda.add_event_source(
            lambda_event_sources.SqsEventSource(
                document_processing_queue,
                batch_size=1  # Process one document at a time
            )
        )

        # Create Lambda for file upload
        upload_lambda = lambda_.Function(
            self, "UploadFunction",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="upload.handler",
            code=lambda_.Code.from_asset("./lambda/document_upload"),
            environment={
                "DOCUMENT_BUCKET": document_bucket.bucket_name,
                "DOCUMENT_TABLE": document_table.table_name,
                "PROCESSING_QUEUE_URL": document_processing_queue.queue_url
            },
            timeout=Duration.minutes(5),
            memory_size=256
        )

        # Create Lambda for document status
        status_lambda = lambda_.Function(
            self, "StatusFunction",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="status.handler",
            code=lambda_.Code.from_asset("./lambda/document_upload"),
            environment={
                "DOCUMENT_TABLE": document_table.table_name,
                "KNOWLEDGE_BASE_ID": kb_id,
                "DATA_SOURCE_ID": data_source_id
            },
            timeout=Duration.minutes(1),
            memory_size=256
        )

        # Grant permissions
        document_bucket.grant_read_write(upload_lambda)
        document_table.grant_read_write_data(upload_lambda)
        document_table.grant_read_data(status_lambda)
        document_processing_queue.grant_send_messages(upload_lambda)

        # Grant permissions to check Bedrock ingestion job status
        status_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=[
                'bedrock:GetIngestionJob',
                'bedrock:ListIngestionJobs',
            ],
            resources=['*'],
        ))

        # Create API Gateway for file upload and document status
        api = apigateway.RestApi(
            self, "DocumentManagementApi",
            rest_api_name="Document Management API",
            description="API for document upload and status tracking",
            default_cors_preflight_options=apigateway.CorsOptions(
                allow_origins=apigateway.Cors.ALL_ORIGINS,
                allow_methods=apigateway.Cors.ALL_METHODS
            )
        )

        # API Gateway endpoints
        uploads = api.root.add_resource("uploads")
        uploads.add_method("POST", apigateway.LambdaIntegration(upload_lambda))

        documents = api.root.add_resource("documents")
        documents.add_method("GET", apigateway.LambdaIntegration(status_lambda))

        website_deployment = s3_deployment.BucketDeployment(
            self, 'DeployWebsite',
            sources=[s3_deployment.Source.asset('./document_chatbot_ui/build')],
            destination_bucket=website_bucket,
            distribution=distribution,
            distribution_paths=['/*'],
            role=iam.Role(
                self, "DeploymentRole",
                assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
                managed_policies=[
                    iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
                ],
                inline_policies={
                    "CloudFrontInvalidation": iam.PolicyDocument(
                        statements=[
                            iam.PolicyStatement(
                                actions=[
                                    "cloudfront:CreateInvalidation",
                                    "cloudfront:GetInvalidation"
                                ],
                                resources=["*"]
                            )
                        ]
                    )
                }
            )
        )

        # Lambda to update config.json with actual values
        update_config_lambda = lambda_.Function(
            self, 'UpdateConfigFunction',
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler='index.handler',
            code=lambda_.Code.from_asset('lambda/update_config'),
            environment={
                'WEBSOCKET_URL': websocket_stage.url,
                'CLOUDFRONT_DOMAIN': kb_document_url,
                'UPLOAD_API_URL': api.url,
                'WEBSITE_BUCKET': website_bucket.bucket_name,
                'DISTRIBUTION_ID': distribution.distribution_id
            },
            timeout=Duration.minutes(5)
        )

        # Grant permissions
        website_bucket.grant_read_write(update_config_lambda)

        update_config_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=[
                "cloudfront:CreateInvalidation",
                "cloudfront:GetInvalidation"
            ],
            resources=["*"]
        ))

        # Custom resource to trigger Lambda after deployment
        config_updater = CustomResource(
            self, 'ConfigUpdaterResource',
            service_token=Provider(
                self, 'ConfigUpdaterProvider',
                on_event_handler=update_config_lambda
            ).service_token,
            properties={
                "Timestamp": datetime.now().isoformat()
            }
        )

        # Make sure this runs after the website is deployed
        config_updater.node.add_dependency(website_deployment)

        # Output the WebSocket API URL
        cdk.CfnOutput(
            self, 'WebSocketURL',
            value=websocket_stage.url,
            description='URL of the WebSocket API',
        )

        # Output the CloudFront distribution URL
        cdk.CfnOutput(
            self, 'WebsiteURL',
            value=f'https://{distribution.distribution_domain_name}',
            description='URL of the website',
        )

        # Output the Upload API URL
        cdk.CfnOutput(
            self, 'UploadApiURL',
            value=api.url,
            description='URL of the document upload API',
        )