import time
import json
import boto3
import os
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow._utils import _RegisterModelStep
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)


class ModelDeployment(StepCollection):
    """custom step to deploy model as SageMaker Endpoint"""

    def __init__(
        self,
        model_name: str,
        registered_model: _RegisterModelStep,
        endpoint_instance_type,
        sagemaker_endpoint_role: str,
        autoscaling_policy: dict = None,
    ):
        self.name = "sagemaker-pipelines-model-deployment"
        self.model_package_arn = registered_model.properties.ModelPackageArn
        self.lambda_role = self.create_lambda_role(self.name)
        #        Use the current time to define unique names for the resources created
        current_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())

        steps = []
        lambda_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deploy_handler.py")
        # Lambda helper class can be used to create the Lambda function
        self.func = Lambda(
            function_name=f"{self.name}-{current_time}",
            execution_role_arn=self.lambda_role,
            script=lambda_file,
            handler="deploy_handler.lambda_handler",
            timeout=600,
            memory_size=256,
        )

        # The dictionary retured by the Lambda function is captured by LambdaOutput, each key in the dictionary corresponds to a
        # LambdaOutput

        output_param_1 = LambdaOutput(output_name="statusCode", output_type=LambdaOutputTypeEnum.String)
        output_param_2 = LambdaOutput(output_name="body", output_type=LambdaOutputTypeEnum.String)
        output_param_3 = LambdaOutput(output_name="other_key", output_type=LambdaOutputTypeEnum.String)

        # The inputs provided to the Lambda function can be retrieved via the `event` object within the `lambda_handler` function
        # in the Lambda
        lambda_step = LambdaStep(
            name="HuggingFaceModelDeployment",
            lambda_func=self.func,
            inputs={
                "model_name": model_name + current_time,
                "endpoint_config_name": model_name + current_time,
                "endpoint_name": model_name,
                "endpoint_instance_type": endpoint_instance_type,
                "model_package_arn": self.model_package_arn,
                "role": sagemaker_endpoint_role,
            },
            outputs=[output_param_1, output_param_2, output_param_3],
        )
        steps.append(lambda_step)
        self.steps = steps

    def create_lambda_role(self, name):
        """
        Create a role for the Lambda function
        """
        role_name = f"{name}-role"
        iam = boto3.client("iam")
        try:
            response = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {"Service": "lambda.amazonaws.com"},
                                "Action": "sts:AssumeRole",
                            }
                        ],
                    }
                ),
                Description="Role for Lambda to call ECS Fargate task",
            )

            role_arn = response["Role"]["Arn"]

            response = iam.attach_role_policy(
                RoleName=role_name, PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
            )

            response = iam.attach_role_policy(
                PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess", RoleName=role_name
            )

            return role_arn

        except iam.exceptions.EntityAlreadyExistsException:
            print(f"Using ARN from existing role: {role_name}")
            response = iam.get_role(RoleName=role_name)
            return response["Role"]["Arn"]
