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


class CommentFunc(Construct):

    def __init__(self, scope: Construct, id_: str, vpc: ec2.Vpc, sg: ec2.SecurityGroup,
                 role: iam.Role, listener: elbv2.ApplicationListener) -> None:
        super().__init__(scope, id_)

        # Comment Repo
        self.ecr_repo = ecr.Repository(
            self,
            'CommentRepo',
            repository_name='comment-func',
            removal_policy=RemovalPolicy.DESTROY,
            empty_on_delete=True
        )

        self.ecr_repo.add_to_resource_policy(
            statement=ECR_LAMBDA_POLICY
        )

        image_asset = ecr_assets.DockerImageAsset(
            self,
            'CommentFuncDockerImage',
            directory='sample_source/containers/comment/',
            platform=ecr_assets.Platform.LINUX_AMD64
        )
        image_asset.node.add_dependency(self.ecr_repo)

        ecr_deployment = ECRDeployment(
            self,
            'DeployCommentFuncImage',
            src=DockerImageName(image_asset.image_uri),
            dest=DockerImageName(
                f'{self.ecr_repo.repository_uri}:{image_asset.asset_hash}'
            )
        )
        ecr_deployment.node.add_dependency(image_asset)        

        lambda_func = lambda_.DockerImageFunction(
            self,
            'CommentFunc',
            function_name='CommentFunc',
            code=lambda_.DockerImageCode.from_ecr(
                repository=self.ecr_repo,
                tag_or_digest=image_asset.asset_hash
            ),
            memory_size=512,
            timeout=Duration.seconds(300),
            role=role,
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            security_groups=[sg],
            environment={
                'BEDROCK_REGION_NAME': 'us-west-2',
            }
        )
        lambda_func.node.add_dependency(ecr_deployment)

        target_group = listener.add_targets(
            'CommentFuncTarget',
            target_group_name='comment-func-tg',
            targets=[targets.LambdaTarget(lambda_func)],
            conditions=[
                elbv2.ListenerCondition.path_patterns(['/comment'])
            ],
            priority=1
        )
        target_group.node.add_dependency(lambda_func)