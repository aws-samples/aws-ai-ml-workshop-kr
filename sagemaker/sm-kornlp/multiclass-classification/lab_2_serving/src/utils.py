import os
import json
from sagemaker.s3 import S3Uploader 
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONLinesSerializer
from sagemaker.deserializers import JSONLinesDeserializer


def print_outputs(outputs):
    jsonlines = outputs.split('\n')

    for jsonline in jsonlines:
        print(json.loads(jsonline))
        
        
def prepare_model_artifact(model_path,
                           model_artifact_path='model_and_code', 
                           model_artifact_name='model.tar.gz'):
    
    os.system(f'rm -rf {model_artifact_path}')
    os.system(f'mkdir {model_artifact_path} {model_artifact_path}/code')
    os.system(f'cp {model_path}/*.* {model_artifact_path}')
    os.system(f'cp ./src/* {model_artifact_path}/code')
    os.system(f'tar cvzf {model_artifact_name} -C {model_artifact_path}/ .') 
    os.system(f'rm -rf {model_artifact_path}')
    print(f'Archived {model_artifact_name}')
    
        
def upload_model_artifact_to_s3(model_variant, model_path, bucket, prefix,
                                model_artifact_path='model_and_code', 
                                model_artifact_name='model.tar.gz'):
    prepare_model_artifact(model_path, model_artifact_path, model_artifact_name)
    model_s3_uri = S3Uploader.upload(model_artifact_name,'s3://{}/{}/{}'.format(bucket, prefix, model_variant))
    os.system(f'rm -rf {model_artifact_name}')
    print(f'Uploaded to {model_s3_uri}')
    
    return model_s3_uri


class NLPPredictor(Predictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super().__init__(
            endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=JSONLinesSerializer(),
            deserializer=JSONLinesDeserializer(),
        )