from aws_cdk import Stack 
from constructs import Construct

from typing import Type
from bedrock_demo.config import BaseConfig

from bedrock_demo.network.networking import Networking
from bedrock_demo.storage.storage import Storage
from bedrock_demo.codecontrol.codecontrol import CodeControl
from bedrock_demo.codecontrol.invoke_build import InvokeBuild
from bedrock_demo.codecontrol.deploy_image import DeployImage
from bedrock_demo.applications.prompt_func import PromptFunc
from bedrock_demo.applications.shortform_func import ShortformFunc
from bedrock_demo.applications.summary_func import SummaryFunc
from bedrock_demo.applications.identify_func import IdentifyFunc
from bedrock_demo.applications.comment_func import CommentFunc
from bedrock_demo.applications.trigger import DeployImageTrigger


class BedrockDemo(Stack):
    def __init__(self, scope: Construct, id_: str, conf: Type[BaseConfig], **kwargs) -> None:
        super().__init__(scope, id_, **kwargs)

        networking = Networking(
            self,
            'Networking',
            cidr=conf.VPC_CIDR,
        )

        storage = Storage(
            self,
            'Storage'
        )

        codecontrol = CodeControl(
            self,
            'CodeControl',
            repo_name=conf.SOURCE_REPO_NAME,
            project_name=conf.BUILD_PROJECT_NAME
        )

        InvokeBuild(
            self,
            'InvokeBuild',
            vpc=networking.vpc,
            sg=networking.sg_lambda,
            source_repo=codecontrol.repo,
            parameter_name=conf.SSM_PARAMETER_NAME,
            project_name=conf.BUILD_PROJECT_NAME
        )

        deploy_func = DeployImage(
            self,
            'DeployImage',
            vpc=networking.vpc,
            sg=networking.sg_lambda,
            source_repo_name=conf.SOURCE_REPO_NAME,
            bucket=storage.bucket,
        )

        # Bedrock Demo Functions
        comment_func = CommentFunc(
            self,
            'CommentFunc',
            vpc=networking.vpc,
            sg=networking.sg_lambda,
            listener=networking.listener,
            role=deploy_func.lambda_func_role
        )

        identify_func = IdentifyFunc(
            self,
            'IdentifyFunc',
            vpc=networking.vpc,
            sg=networking.sg_lambda,
            listener=networking.listener,
            role=deploy_func.lambda_func_role,
            bucket_name=storage.bucket.bucket_name
        )

        prompt_func = PromptFunc(
            self,
            'PromptFunc',
            vpc=networking.vpc,
            sg=networking.sg_lambda,
            listener=networking.listener,
            role=deploy_func.lambda_func_role            
        )

        shortform_func = ShortformFunc(
            self,
            'ShortformFunc',
            vpc=networking.vpc,
            sg=networking.sg_lambda,
            listener=networking.listener,
            role=deploy_func.lambda_func_role,
            bucket_name=storage.bucket.bucket_name,
            mediaconvert_role_arn=deploy_func.mediaconvert_role.role_arn,            
        )

        summary_func = SummaryFunc(
            self,
            'SummaryFunc',
            vpc=networking.vpc,
            sg=networking.sg_lambda,
            listener=networking.listener,
            role=deploy_func.lambda_func_role            
        )

        DeployImageTrigger(
            self,
            'DeployImageTrigger',
            deploy_image_func=deploy_func.deploy_image_func,
            ecr_repo_names=[
                comment_func.ecr_repo.repository_name,
                identify_func.ecr_repo.repository_name,
                prompt_func.ecr_repo.repository_name,
                shortform_func.ecr_repo.repository_name,
                summary_func.ecr_repo.repository_name
            ]
        )








