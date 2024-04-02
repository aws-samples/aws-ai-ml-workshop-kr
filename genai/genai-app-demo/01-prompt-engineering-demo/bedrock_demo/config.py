import os
import aws_cdk as cdk


class BaseConfig:
    # VPC CIDR
    VPC_CIDR = '10.0.0.0/16'

    SOURCE_REPO_NAME = 'BedrockDemo'
    BUILD_PROJECT_NAME = 'Build-Image'
    SSM_PARAMETER_NAME = 'last_commit_id'


app_config = {
    'config': BaseConfig
}
