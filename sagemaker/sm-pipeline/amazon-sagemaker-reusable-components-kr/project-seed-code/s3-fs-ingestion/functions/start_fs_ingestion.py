import json
import os
import boto3
import logging
from time import gmtime, strftime

logger = logging.getLogger(__name__)
logging.root.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# configuration settings
SM_PIPELINE_NAME = os.environ.get("SM_PIPELINE_NAME", "")

s3 = boto3.resource('s3')
sm = boto3.client('sagemaker')

def lambda_handler(event, context):
    try:
        operation = event["detail"]["eventName"]
        obj_key = event["detail"]["requestParameters"]["key"]
        bucket_name = event["detail"]["requestParameters"]["bucketName"]

        logger.info(f"Got the event: {operation} for the object: {bucket_name}/{obj_key}")

        logger.info(f"Starting pipeline {SM_PIPELINE_NAME}")

        start_pipeline = sm.start_pipeline_execution(
                PipelineName=SM_PIPELINE_NAME,
                PipelineExecutionDisplayName=f"{obj_key.split('/')[-1].replace('_','').replace('.csv','')}-{strftime('%d-%H-%M-%S', gmtime())}",
                PipelineParameters=[
                    {
                        'Name': 'InputDataUrl',
                        'Value': f"s3://{bucket_name}/{obj_key}"
                    },
                ],
                PipelineExecutionDescription=obj_key
                )

        logger.info(f"start_pipeline_execution returned {start_pipeline}")

    except Exception as e:
        logger.error(f"Exception in start_fs_ingestion function: {str(e)}")
        return
