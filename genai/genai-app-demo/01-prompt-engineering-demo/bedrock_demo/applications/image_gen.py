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


class ImageGen(Construct):

    def __init__(self, scope: Construct, id_: str, vpc: ec2.Vpc, sg: ec2.SecurityGroup,
                 role: iam.Role, listener: elbv2.ApplicationListener, bucket_name: str) -> None:
        super().__init__(scope, id_)

        # Prompt Repo
        self.ecr_repo = ecr.Repository(
            self,
            'ImageGenRepo',
            repository_name='image-gen',
            removal_policy=RemovalPolicy.DESTROY,
            empty_on_delete=True
        )

        self.ecr_repo.add_to_resource_policy(
            statement=ECR_LAMBDA_POLICY
        )

        image_asset = ecr_assets.DockerImageAsset(
            self,
            'ImageGenDockerImage',
            directory='sample_source/containers/image_gen/',
            platform=ecr_assets.Platform.LINUX_AMD64
        )
        image_asset.node.add_dependency(self.ecr_repo)

        ecr_deployment = ECRDeployment(
            self,
            'DeployImageGenImage',
            src=DockerImageName(image_asset.image_uri),
            dest=DockerImageName(
                f'{self.ecr_repo.repository_uri}:{image_asset.asset_hash}'
            )
        )
        ecr_deployment.node.add_dependency(image_asset)        

        lambda_func = lambda_.DockerImageFunction(
            self,
            'ImageGen',
            function_name='ImageGen',
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
                'BUCKET_NAME': bucket_name,
            }
        )
        lambda_func.node.add_dependency(ecr_deployment)

        target_group = listener.add_targets(
            'ImageGenTarget',
            target_group_name='image-gen-tg',
            targets=[targets.LambdaTarget(lambda_func)],
            conditions=[
                elbv2.ListenerCondition.path_patterns(['/image'])
            ],
            priority=7
        )
        target_group.node.add_dependency(lambda_func)