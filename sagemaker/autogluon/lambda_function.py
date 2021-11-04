import json
import os
import boto3
import datetime
from urllib.parse import unquote_plus
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def lambda_handler(event, context):
  for record in event['Records']:
    bucket = record['s3']['bucket']['name']
    key = unquote_plus(record['s3']['object']['key'])
    tmpkey = key.replace('/', '')
  logging.info(key)
  logging.info(tmpkey)
  filename = key.split('/')[-1]
  print(filename)
  dataset = filename.split('_')[0]
  print(dataset)
  
  now = datetime.datetime.now
  str_time = now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]
  sm = boto3.Session().client('sagemaker')
  training_job_params = {
    'TrainingJobName': dataset + '-autogluon-' + str_time,
    'HyperParameters': {
      'filename':json.dumps(filename),
      'sagemaker_container_log_level': '20',
      'sagemaker_enable_cloudwatch_metrics': 'false',
      'sagemaker_program': 'autogluon-tab-with-test.py',
      'sagemaker_region': os.environ['AWS_REGION'],
      'sagemaker_submit_directory': 's3://' + bucket + '/source/sourcedir.tar.gz',
      's3-output': os.environ['S3_OUTPUT_PATH']
    },
    'AlgorithmSpecification': {
      'TrainingImage': '763104351884.dkr.ecr.' + os.environ['AWS_REGION'] + '.amazonaws.com/mxnet-training:1.6.0-cpu-py3',
      'TrainingInputMode': 'File',
      'EnableSageMakerMetricsTimeSeries': False
    },
    'RoleArn': os.environ['SAGEMAKER_ROLE_ARN'],
    'InputDataConfig': [
      {
        'ChannelName': 'training',
        'DataSource': {
          'S3DataSource': {
            'S3DataType': 'S3Prefix',
            'S3Uri': os.environ['S3_TRIGGER_PATH'],
            'S3DataDistributionType': 'FullyReplicated'
          }
        },
        'CompressionType': 'None',
        'RecordWrapperType': 'None'
      }
    ],
    'OutputDataConfig': {
      'KmsKeyId': '',
      'S3OutputPath': os.environ['S3_OUTPUT_PATH']
    },
    'ResourceConfig': {
      'InstanceType': os.environ['AG_INSTANCE_TYPE'],
      'InstanceCount': 1,
      'VolumeSizeInGB': 30
    },
    'StoppingCondition': {
      'MaxRuntimeInSeconds': 86400
    },
    'EnableNetworkIsolation': False,
    'EnableInterContainerTrafficEncryption': False,
    'EnableManagedSpotTraining': False,
  }
  
  response = sm.create_training_job(**training_job_params)
  
  return {
    'statusCode': 200,
    'body': json.dumps(key)
  }