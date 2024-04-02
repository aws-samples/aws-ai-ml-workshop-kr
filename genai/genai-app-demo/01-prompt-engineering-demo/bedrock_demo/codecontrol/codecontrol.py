from aws_cdk import (
    Aws,
    aws_codecommit as codecommit,
    aws_codebuild as codebuild,
    aws_iam as iam,
)
from constructs import Construct


class CodeControl(Construct):

    def __init__(self, scope: Construct, id_: str, repo_name: str, project_name: str) -> None:
        super().__init__(scope, id_)

        # CodeCommit
        self.repo = codecommit.Repository(
            self,
            'BedrockDemo',
            repository_name=repo_name,
            description='Bedrock Demo Source Code',
            code=codecommit.Code.from_directory('sample_source/', 'main')
        )

        # CodeBuild
        builder = codebuild.Project(
            self,
            'ContainerBuild',
            project_name=project_name,
            source=codebuild.Source.code_commit(
                repository=self.repo,
                branch_or_ref='main'
            ),
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.AMAZON_LINUX_2_5,
                privileged=True,
                environment_variables={
                    'AWS_DEFAULT_REGION': codebuild.BuildEnvironmentVariable(value=Aws.REGION),
                    'AWS_ACCOUNT_ID': codebuild.BuildEnvironmentVariable(value=Aws.ACCOUNT_ID),
                    'IMAGE_REPO_NAME': codebuild.BuildEnvironmentVariable(value=''),
                    'IMAGE_TAG': codebuild.BuildEnvironmentVariable(value='latest'),
                    'CONTAINER_FOLDER': codebuild.BuildEnvironmentVariable(value='')
                }
            ),
            build_spec=codebuild.BuildSpec.from_source_filename('buildspec.yml')
        )

        # ECR Push Permission
        builder.add_to_role_policy(
            statement=iam.PolicyStatement(
                actions=[
                    'ecr:BatchCheckLayerAvailability',
                    'ecr:CompleteLayerUpload',
                    'ecr:InitiateLayerUpload',
                    'ecr:PutImage',
                    'ecr:UploadLayerPart'
                ],
                resources=[f'arn:aws:ecr:{Aws.REGION}:{Aws.ACCOUNT_ID}:repository/*'],
                effect=iam.Effect.ALLOW
            )
        )

        # ECR Login Permission
        builder.add_to_role_policy(
            statement=iam.PolicyStatement(
                actions=[
                    'ecr:GetAuthorizationToken'
                ],
                resources=['*'],
                effect=iam.Effect.ALLOW
            )
        )
