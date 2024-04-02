from aws_cdk import (
    RemovalPolicy, Duration,
    aws_ec2 as ec2,
    aws_ecr as ecr,
    aws_lambda as lambda_,
    aws_ecr_assets as ecr_assets,
    aws_iam as iam,
    aws_elasticloadbalancingv2 as elbv2,
    aws_elasticloadbalancingv2_targets as targets,
)
from constructs import Construct
from . import ECR_LAMBDA_POLICY
from cdk_ecr_deployment import DockerImageName, ECRDeployment




class TestFunc(Construct):

    def __init__(self, scope: Construct, id_: str) -> None:
        super().__init__(scope, id_)

        vpc = ec2.Vpc(
            self,
            'BedrockDemoVPC',
            ip_addresses=ec2.IpAddresses.cidr('10.0.0.0/16'),
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name='Public',
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name='Private',
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24
                )
            ],
            nat_gateways=2,
            enable_dns_hostnames=True,
            enable_dns_support=True,
            max_azs=2,
            restrict_default_security_group=False
        )

        # Security Group for ALB
        sg_alb = ec2.SecurityGroup(
            self,
            'ALB_SG',
            vpc=vpc,
            allow_all_outbound=True,
            description='Security Group for ALB',
            security_group_name='alb-sg'
        )
        sg_alb.add_ingress_rule(
            peer=ec2.Peer.any_ipv4(),
            connection=ec2.Port.tcp(80),
            description="allow HTTP access from Internet"
        )

        alb = elbv2.ApplicationLoadBalancer(
            self,
            "BedrockALB",
            load_balancer_name='bedrock-alb',
            internet_facing=True,
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PUBLIC
            ),
            security_group=sg_alb
        )

        listener = alb.add_listener(
            "Listener",
            port=80,
            default_action=elbv2.ListenerAction.fixed_response(
                status_code=200,
                message_body="OK",
                content_type="text/plain"
            )
        )        

        # Comment Repo
        ecr_repo = ecr.Repository(
            self,
            'TestRepo',
            repository_name='test-func',
            removal_policy=RemovalPolicy.DESTROY,
            empty_on_delete=True
        )

        ecr_repo.add_to_resource_policy(
            statement=ECR_LAMBDA_POLICY
        )

        image_asset = ecr_assets.DockerImageAsset(
            self,
            'TestFuncDockerImage',
            directory='sample_source/containers/prompt/',
            platform=ecr_assets.Platform.LINUX_AMD64
        )
        image_asset.node.add_dependency(ecr_repo)

        ecr_deployment = ECRDeployment(
            self,
            'DeployTestFuncImage',
            src=DockerImageName(image_asset.image_uri),
            dest=DockerImageName(
                f'{ecr_repo.repository_uri}:{image_asset.asset_hash}'
            )
        )
        ecr_deployment.node.add_dependency(image_asset)

        lambda_func = lambda_.DockerImageFunction(
            self,
            'TestFunc',
            function_name='TestFunc',
            code=lambda_.DockerImageCode.from_ecr(
                repository=ecr_repo,
                tag_or_digest=image_asset.asset_hash
            ),
            memory_size=512,
            timeout=Duration.seconds(300),
            environment={
                'BEDROCK_REGION_NAME': 'us-west-2',
            }
        )
        lambda_func.node.add_dependency(ecr_deployment)

        target_group = listener.add_targets(
            'TestFuncTarget',
            target_group_name='test-func-tg',
            targets=[targets.LambdaTarget(lambda_func)],
            conditions=[
                elbv2.ListenerCondition.path_patterns(['/test'])
            ],
            priority=1
        )
        target_group.node.add_dependency(lambda_func)        



