import time
import boto3
import argparse
import sys, os

import logging
import logging.handlers

def _get_logger():
    '''
    로깅을 위해 파이썬 로거를 사용
    # https://stackoverflow.com/questions/17745914/python-logging-module-is-printing-lines-multiple-times
    '''
    loglevel = logging.DEBUG
    l = logging.getLogger(__name__)
    if not l.hasHandlers():
        l.setLevel(loglevel)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))        
        l.handler_set = True
    return l  

logger = _get_logger()



# Parse argument variables passed via the DeployModel processing step
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
parser.add_argument('--region', type=str, default="ap-northeast-2")
parser.add_argument('--endpoint_instance_type', type=str, default='ml.t3.medium')
parser.add_argument('--endpoint_name', type=str)
args = parser.parse_args()

logger.info("#############################################")
logger.info(f"args.model_name: {args.model_name}")
logger.info(f"args.region: {args.region}")    
logger.info(f"args.endpoint_instance_type: {args.endpoint_instance_type}")        
logger.info(f"args.endpoint_name: {args.endpoint_name}")    

region = args.region
instance_type = args.endpoint_instance_type
model_name = args.model_name


boto3.setup_default_session(region_name=region)
sagemaker_boto_client = boto3.client('sagemaker')

#name truncated per sagameker length requirememnts (63 char max)
endpoint_config_name=f'{args.model_name[:56]}-config'
existing_configs = sagemaker_boto_client.list_endpoint_configs(NameContains=endpoint_config_name)['EndpointConfigs']

if not existing_configs:
    create_ep_config_response = sagemaker_boto_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'InstanceType': instance_type,
            'InitialVariantWeight': 1,
            'InitialInstanceCount': 1,
            'ModelName': model_name,
            'VariantName': 'AllTraffic'
        }]
    )

existing_endpoints = sagemaker_boto_client.list_endpoints(NameContains=args.endpoint_name)['Endpoints']

if not existing_endpoints:
    logger.info(f"Creating endpoint")        
    create_endpoint_response = sagemaker_boto_client.create_endpoint(
        EndpointName=args.endpoint_name,
        EndpointConfigName=endpoint_config_name)
else:
    logger.info(f"Endpoint exists")            
    

endpoint_info = sagemaker_boto_client.describe_endpoint(EndpointName=args.endpoint_name)
endpoint_status = endpoint_info['EndpointStatus']


logger.info(f'Endpoint status is creating')    
while endpoint_status == 'Creating':
    endpoint_info = sagemaker_boto_client.describe_endpoint(EndpointName=args.endpoint_name)
    endpoint_status = endpoint_info['EndpointStatus']
    logger.info(f'Endpoint status: {endpoint_status}')
    if endpoint_status == 'Creating':
        time.sleep(30)