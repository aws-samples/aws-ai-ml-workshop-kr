from aws_cdk import (
    Duration, Aws,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_codecommit as codecommit,
    aws_events_targets as targets,
)
from constructs import Construct


class InvokeBuild(Construct):

    def __init__(self, scope: Construct, id_: str,
                 vpc: ec2.Vpc, sg: ec2.SecurityGroup, source_repo: codecommit.Repository,
                 parameter_name: str, project_name: str) -> None:
        super().__init__(scope, id_)

        # Invoke Build Lambda
        invoke_build_func = lambda_.Function(
            self,
            'InvokeBuildLambda',
            function_name='InvokeBuild',
            handler='lambda_function.lambda_handler',
            runtime=lambda_.Runtime.PYTHON_3_12,
            code=lambda_.Code.from_asset('bedrock_demo/codecontrol/runtimes/invoke_build'),
            timeout=Duration.seconds(30),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            security_groups=[sg],
            environment={
                'REPO_NAME': source_repo.repository_name,
                'LAST_COMMIT_ID': parameter_name,
                'BUILD_PROJECT_NAME': project_name,
                'DIR_FILTER': 'containers/'
            }
        )

        # ParameterStore
        invoke_build_func.add_to_role_policy(
            statement=iam.PolicyStatement(
                actions=['ssm:GetParameter', 'ssm:PutParameter'],
                resources=[f'arn:aws:ssm:{Aws.REGION}:{Aws.ACCOUNT_ID}:parameter/{parameter_name}'],
                effect=iam.Effect.ALLOW
            )
        )
        # CodeBuildStart
        invoke_build_func.add_to_role_policy(
            statement=iam.PolicyStatement(
                actions=['codebuild:StartBuild'],
                resources=[f'arn:aws:codebuild:{Aws.REGION}:{Aws.ACCOUNT_ID}:project/{project_name}'],
                effect=iam.Effect.ALLOW
            )
        )
        # CodeCommitRead
        invoke_build_func.add_to_role_policy(
            statement=iam.PolicyStatement(
                actions=[
                    'codecommit:GetRepository', 'codecommit:GetBranch',
                    'codecommit:GetCommit', 'codecommit:GetDifferences',
                    'codecommit:GetFile'
                ],
                resources=[f'arn:aws:codecommit:{Aws.REGION}:{Aws.ACCOUNT_ID}:{source_repo.repository_name}'],
                effect=iam.Effect.ALLOW
            )
        )

        source_repo.on_reference_updated(
            'BuildImageTrigger',
            description='Build Image',
            rule_name='BuildImageTrigger',
            target=targets.LambdaFunction(invoke_build_func)
        )