import json
import boto3


def lambda_handler(event, context):
    """
    모델 레지스트리에서 최신 버전의 모델 승인 상태를 변경하는 람다 함수.
    """
    
    try:
        sm_client = boto3.client("sagemaker")

        ##############################################
        # 람다 함수는 두개의 입력 인자를 event 개체를 통해서 받습니다.
        # 모델 패키지 이름과 모델 승인 유형을 받습니다.
        ##############################################   
        
        model_package_group_name = event["model_package_group_name"]
        ModelApprovalStatus = event["ModelApprovalStatus"]        
        print("model_package_group_name: \n", model_package_group_name)
        print("ModelApprovalStatus: \n", ModelApprovalStatus)        

        
        # 해당 모델 패키지에서 가장 최근의 버전의 모델을 가져옵니다.
        response = sm_client.list_model_packages(ModelPackageGroupName= model_package_group_name)
        ModelPackageArn = response['ModelPackageSummaryList'][0]['ModelPackageArn']
        sm_client.describe_model_package(ModelPackageName=ModelPackageArn)

        ##############################################                
        # 최근 모델의 모델에  승인 상태를 변경 합니다.
        ##############################################                

        
        # 최근 모델의 모델 승인 상태를 가지고 있는 사전 변수를 선언합니다.
        model_package_update_input_dict = {
            "ModelPackageArn" : ModelPackageArn,
            "ModelApprovalStatus" : ModelApprovalStatus
        }
        
        # 모델 승인 상태 변경
        model_package_update_response = sm_client.update_model_package(**model_package_update_input_dict)
        respone = sm_client.describe_model_package(ModelPackageName=ModelPackageArn)        

        return_msg = f"Success"
        
        ##############################################        
        # 람다 함수의 리턴 정보를 구성하고 리턴 합니다.
        ##############################################        

        return {
            "statusCode": 200,
            "body": json.dumps(return_msg),
            "other_key": "s3://sagemaker-us-east-1-057716757052/pytorch-training-2022-10-14-13-19-13-618/output/model.tar.gz",


        }

    except BaseException as error:
        return_msg = f"There is no model_package_group_name{model_package_group_name}"                
        error_msg = f"An exception occurred: {error}"
        print(error_msg)    
        return {
            "statusCode": 500,
            "body": json.dumps(return_msg),
            "other_key": "s3://sagemaker-us-east-1-057716757052/pytorch-training-2022-10-14-13-19-13-618/output/model.tar.gz",
        }        
        

