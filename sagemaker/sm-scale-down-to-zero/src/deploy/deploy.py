import os
import sys
import boto3
import shutil
import argparse
import sagemaker
#import subprocess
#from distutils.dir_util import copy_tree

from sagemaker import ModelPackage
#from sagemaker.pytorch.model import PyTorchModel
#from sagemaker.serializers import CSVSerializer
#from sagemaker.deserializers import JSONDeserializer
#from sagemaker.serializers import CSVSerializer, NumpySerializer, JSONSerializer
from sagemaker.workflow.pipeline_context import PipelineSession, LocalPipelineSession
    
class deploy():
    
    def __init__(self, args):
         
        self.args = args
        print (self.args)
        
    def _create_endpoint(self,):
                    
        sagemaker_session = sagemaker.Session() 
        sm_client = boto3.client('sagemaker')
        
        model_package_arn = sm_client.list_model_packages(
            ModelPackageGroupName=self.args.model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending"
        )['ModelPackageSummaryList'][0]['ModelPackageArn']
        
        print ("model_package_arn: ", model_package_arn)
        print (f"Endpoint-name: {self.args.endpoint_name}")
        print ("sagemaker_session", sagemaker_session) 
        print ("self.args.execution_role", self.args.execution_role)
        
        # 모델 생성
        model = ModelPackage(
            role=self.args.execution_role, 
            model_package_arn=model_package_arn, 
            sagemaker_session=sagemaker_session
        )
        
        model.deploy(
            initial_instance_count=1,
            instance_type=self.args.depoly_instance_type,
            endpoint_name=self.args.endpoint_name
        )

    def execution(self, ):
        
        self._create_endpoint() 
        
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_deploy", type=str, default="/opt/ml/processing/")
    parser.add_argument("--region", type=str, default="ap-northeast-2")
    parser.add_argument("--instance_type", type=str, default="ml.g4dn.xlarge")    
    parser.add_argument("--depoly_instance_type", type=str, default="ml.g4dn.xlarge")    
    parser.add_argument("--model_package_group_name", type=str, default="model_package_group_name")
    parser.add_argument("--endpoint_name", type=str, default="endpoint_name")
    parser.add_argument("--execution_role", type=str, default="execution_role")
    #parser.add_argument("--local_mode", type=str, default="local_mode")
    
    args, _ = parser.parse_known_args()
           
    print("Received arguments {}".format(args))
    os.environ['AWS_DEFAULT_REGION'] = args.region
    
    dep = deploy(args)
    dep.execution()