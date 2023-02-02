"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession

BASE_DIR = os.path.dirname(os.path.realpath(__file__))



def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )


import os

def print_files_in_dir(root_dir, prefix):
    files = os.listdir(root_dir)
    for file in files:
        path = os.path.join(root_dir, file)
        print(prefix + path)
        if os.path.isdir(path):
            print_files_in_dir(path, prefix + "    ")

            
def get_pipeline(
#    s3_input_data_uri,    
    project_prefix,
    region,
    endpoint_name = None, # Sagemaker Endpoint Name
    role=None, # SAGEMAKER_PIPELINE_ROLE_ARN 이 넘어옴.
    default_bucket=None,
    model_package_group_name= None,
    pipeline_name= None,
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    ##################################
    ## 입력 파라미터 확인
    ##################################        
    print("######### get_pipeline() input parameter ###############")
    print(f"### BASE_DIR: {BASE_DIR}")    
#    print(f"s3_input_data_uri: {s3_input_data_uri}")        
    print(f"project_prefix: {project_prefix}")            
    # print(f"sagemaker_project_arn: {sagemaker_project_arn}")            
    print(f"role: {role}")            
    print(f"default_bucket: {default_bucket}")            
    print(f"model_package_group_name: {model_package_group_name}")            
    print(f"endpoint_name: {endpoint_name}")                
    print(f"pipeline_name: {pipeline_name}")            
    ##################################
    ## 현재 폴더 기준으로 하위 폴더 및 파일 보기
    ##################################        
    print("######### Look at subfolder and files #########")    
    print_files_in_dir(root_dir =".", prefix="")                    
    
    ##################################
    ## 환경 초기화
    ##################################        
    
    
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
        
    print("role: ", role) # SAGEMAKER_PIPELINE_ROLE_ARN 이 넘어옴.         

    pipeline_session = get_pipeline_session(region, default_bucket)

        
    ##################################
    ## 파이프라인 파라미터 정의
    ##################################        
    
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="Approved"
    )
    
    endpoint_instance_type = ParameterString(
        name="endpoint_instance_type", default_value="ml.g4dn.xlarge"
    )
    
    
    

    ##################################
    ## 모델 Approval 스텝 생성
    ##################################    
    
    from sagemaker.lambda_helper import Lambda
    from sagemaker.workflow.lambda_step import (
        LambdaStep,
        LambdaOutput,
        LambdaOutputTypeEnum,
    )

    function_name = "sagemaker-lambda-step-approve-model-deployment"
    print("function_name: \n", function_name)

    approval_lambda_script_path = f'{BASE_DIR}/iam_change_model_approval.py'
    print("approval_lambda_script_path: \n", approval_lambda_script_path)
    

    # Lambda helper class can be used to create the Lambda function
    func_approve_model = Lambda(
        function_name=function_name,
        execution_role_arn=role,
        script=approval_lambda_script_path,        
        handler="iam_change_model_approval.lambda_handler",
    )

    output_param_1 = LambdaOutput(output_name="statusCode", output_type=LambdaOutputTypeEnum.String)
    output_param_2 = LambdaOutput(output_name="body", output_type=LambdaOutputTypeEnum.String)
    output_param_3 = LambdaOutput(output_name="image_uri_approved", output_type=LambdaOutputTypeEnum.String)
    output_param_4 = LambdaOutput(output_name="ModelDataUrl_approved", output_type=LambdaOutputTypeEnum.String)
    
    step_approve_lambda = LambdaStep(
    name="LambdaApproveModelStep",
    lambda_func=func_approve_model,
    inputs={
        "model_package_group_name" : model_package_group_name,
        "ModelApprovalStatus": model_approval_status,
    },
    outputs=[output_param_1, output_param_2, output_param_3, output_param_4],
    )
    

    ##################################
    ## 배포할 세이지 메이커 모델 스텝 생성
    ##################################    

    from sagemaker.workflow.model_step import ModelStep
    from sagemaker.model import Model
    import time    
    
    import time 

    current_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    model_name = f'{endpoint_name}-{current_time}'    
    print("model_name: \n", model_name)    


    model = Model(
        image_uri= step_approve_lambda.properties.Outputs["image_uri_approved"],
        model_data = step_approve_lambda.properties.Outputs["ModelDataUrl_approved"],    
        role=role,
        name = model_name, # SageMaker Model Name
        sagemaker_session=pipeline_session,
    )
    
#    instance_type = 'ml.g4dn.xlarge' # $ 0.906 per Hour

    step_model_create = ModelStep(
       name="MyModelCreationStep",
       step_args=model.create(instance_type = endpoint_instance_type)
    )    
    
    ##################################
    ## 모델 앤드 포인트 배포를 위한 람다 스텝 생성
    ##################################    



    endpoint_config_name = f'{endpoint_name}-{current_time}'
    # endpoint_name = "lambda-deploy-endpoint-" + current_time

    # function_name = "sagemaker-lambda-step-endpoint-deploy-" + current_time
    function_name = "sagemaker-lambda-step-endpoint-deployment"


    print("endpoint_config_name: \n", endpoint_config_name)
    print("endpoint_name: \n", endpoint_name)
    print("function_name: \n", function_name)

    create_endpoint_lambda_script_path = f'{BASE_DIR}/iam_create_endpoint.py'
    print("create_endpoint_lambda_script_path: \n", create_endpoint_lambda_script_path)
    
    # Lambda helper class can be used to create the Lambda function
    func_deploy_model = Lambda(
        function_name=function_name,
        execution_role_arn=role,
        script=create_endpoint_lambda_script_path,
        handler="iam_create_endpoint.lambda_handler",
        timeout = 600, # 디폴트는 120초 임. 10분으로 연장
    )

    output_param_1 = LambdaOutput(output_name="statusCode", output_type=LambdaOutputTypeEnum.String)
    output_param_2 = LambdaOutput(output_name="body", output_type=LambdaOutputTypeEnum.String)
    output_param_3 = LambdaOutput(output_name="other_key", output_type=LambdaOutputTypeEnum.String)

    step_deploy_lambda = LambdaStep(
        name="LambdaDeployStep",
        lambda_func=func_deploy_model,
        inputs={
            "model_name": step_model_create.properties.ModelName,
            "endpoint_config_name": endpoint_config_name,
            "endpoint_name": endpoint_name,
            "instance_type": endpoint_instance_type,        
        },
        outputs=[output_param_1, output_param_2, output_param_3],
    )
    

    
    ##################################
    # pipeline creation
    ##################################    
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            model_approval_status,
            endpoint_instance_type
        ],
        steps=[step_approve_lambda, step_model_create, step_deploy_lambda],    
        sagemaker_session=pipeline_session
    )
    
    return pipeline


