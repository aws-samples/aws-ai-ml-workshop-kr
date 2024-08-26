from aws_cdk import (
    Duration, Aws,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_codecommit as codecommit,
    aws_events as events,
    aws_events_targets as targets,
    aws_s3 as s3,
)
from constructs import Construct


class DeployImage(Construct):

    def __init__(self, scope: Construct, id_: str,
                 vpc: ec2.Vpc, sg: ec2.SecurityGroup, source_repo_name: str, bucket: s3.Bucket) -> None:
        super().__init__(scope, id_)

        # MediaConvert Role
        self.mediaconvert_role = iam.Role(
            self,
            'MediaConvertRole',
            role_name='MediaConvert-GenAIDemo-Role',
            assumed_by=iam.ServicePrincipal('mediaconvert.amazonaws.com'),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name('AmazonAPIGatewayInvokeFullAccess'),
                iam.ManagedPolicy.from_aws_managed_policy_name('AmazonS3FullAccess')
            ]
        )

        # Deploy Image Lambda
        self.lambda_func_role = iam.Role(
            self,
            'LambdaFuncRole', 
            role_name='LambdaFuncRole',
            assumed_by=iam.ServicePrincipal('lambda.amazonaws.com'),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name('service-role/AWSLambdaBasicExecutionRole'),
                iam.ManagedPolicy.from_aws_managed_policy_name('service-role/AWSLambdaVPCAccessExecutionRole')
            ],
            inline_policies={
                'BedrockInvokeModel': iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=[
                                'bedrock:InvokeModel',
                                'bedrock:InvokeModelWithResponseStream'
                            ],
                            resources=['arn:aws:bedrock:us-west-2::foundation-model/*'],
                            effect=iam.Effect.ALLOW
                        )
                    ]
                ),
                'S3ObjectIO': iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=[
                                's3:Get*',
                                's3:Put*',
                                's3:List*'
                            ],
                            resources=[
                                bucket.bucket_arn,
                                bucket.arn_for_objects('*')
                            ],
                            effect=iam.Effect.ALLOW
                        )
                    ]
                ),
                'TranscribeJob': iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=[
                                'transcribe:StartTranscriptionJob',
                                'transcribe:GetTranscriptionJob'
                            ],
                            resources=[
                                f'arn:aws:transcribe:{Aws.REGION}:{Aws.ACCOUNT_ID}:transcription-job/*'
                            ],
                            effect=iam.Effect.ALLOW    
                        )
                    ]
                ),
                'MediaConvertJob': iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=[
                                "mediaconvert:*"
                            ],
                            resources=[
                                '*'
                            ],
                            effect=iam.Effect.ALLOW
                        ),
                        iam.PolicyStatement(
                            actions=[
                                "iam:PassRole"
                            ],
                            resources=[
                                self.mediaconvert_role.role_arn
                            ],
                            effect=iam.Effect.ALLOW
                        )
                    ]
                )
            }            
        )
        selection = vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)
        self.deploy_image_func = lambda_.Function(
            self,
            'DeployImageLambda',
            function_name='DeployImage',
            handler='lambda_function.lambda_handler',
            runtime=lambda_.Runtime.PYTHON_3_12,
            code=lambda_.Code.from_asset('bedrock_demo/codecontrol/runtimes/deploy_image'),
            timeout=Duration.seconds(30),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            security_groups=[sg],
            environment={
                'FUNC_ROLE': self.lambda_func_role.role_arn,
                'FUNC_SUBNET_ID': ','.join(selection.subnet_ids),
                'FUNC_SG_ID': sg.security_group_id,
                'REPO_NAME': source_repo_name
            }
        )      

        # LambdaUpdateCode
        self.deploy_image_func.add_to_role_policy(
            statement=iam.PolicyStatement(
                actions=[
                    'lambda:UpdateFunctionCode',
                    'lambda:CreateFunction'
                ],
                resources=[f'arn:aws:lambda:{Aws.REGION}:{Aws.ACCOUNT_ID}:function:*'],
                effect=iam.Effect.ALLOW                
            )
        )
        # Permissions for lambda creation
        self.deploy_image_func.add_to_role_policy(
            statement=iam.PolicyStatement(
                actions=[
                    'ec2:DescribeVpcs',
                    'ec2:DescribeSubnets',
                    'ec2:DescribeSecurityGroups',
                ],
                resources=['*'],
                effect=iam.Effect.ALLOW
            )
        )
        # Pass Role for Lambda Role
        self.deploy_image_func.add_to_role_policy(
            statement=iam.PolicyStatement(
                actions=[
                    'iam:PassRole'
                ],
                resources=[self.lambda_func_role.role_arn],
                effect=iam.Effect.ALLOW
            )
        )
        # CodeCommitRead
        self.deploy_image_func.add_to_role_policy(
            statement=iam.PolicyStatement(
                actions=[
                    'codecommit:GetFile'
                ],
                resources=[f'arn:aws:codecommit:{Aws.REGION}:{Aws.ACCOUNT_ID}:{source_repo_name}'],
                effect=iam.Effect.ALLOW
            )
        )
