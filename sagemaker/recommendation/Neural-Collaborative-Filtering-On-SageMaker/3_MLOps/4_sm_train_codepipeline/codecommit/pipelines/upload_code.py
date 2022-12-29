from __future__ import absolute_import

import logging
import argparse
import json
import sys
import traceback
import os
import json
import sagemaker

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
# logger.setLevel(logging.WARNING)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler(sys.stdout))  
    

def parser_args():
    parser = argparse.ArgumentParser(
        "Upload training code for the pipeline script."
    )

    parser.add_argument(
        "-code-repository-name",
        "--code-repository-name",
        dest="code_repository_name",
        type=str,
        help="The source codecommit repository name",
    )
    
    parser.add_argument(
        "-bucket",
        "--bucket",
        dest="bucket",
        type=str,
        help="bucket for the source codecommit repository name",
    )

    

    args = parser.parse_args()

    return args

def clone_code_repository(args):
    '''
    codecommit repository name 을 얻고, git clone 을 함.
    '''
    logger.info("###### Code Repository Name ############:")    
    logger.info(f"code_repository_name: {args.code_repository_name}")
    
    code_repo_name = args.code_repository_name
    
    import boto3

    client = boto3.client('codecommit')    
    response = client.get_repository(repositoryName=code_repo_name)
    
    logger.debug(f"response , {response}")
    
    # Store cloneUrlHttp
    cloneUrlHttp = response['repositoryMetadata']["cloneUrlHttp"]
    
    logger.info(f"###### cloneUrlHttp ############: {cloneUrlHttp}")
    # Run git clone
    os.system(f"git clone {cloneUrlHttp}")    
    
    # Show folder
    logger.info(f"###### Showing folder after cloning repo ############")    
    for dirpath, dirs, files in os.walk("./"):
        print(dirpath)
    


    
def upload_code_s3(args):
    '''
    훈련 코드를와 디펜던시 파일을 압축하여 S3에 업로드하고 S3 URL 을 제공
    '''
    
    logger.info(f"###### tar soruce.tar.gz ############")    
    # train 훈련 코드가 있는 곳 (train.py and others)
    package_dir = "pipelines/ncf/src"
    # source.tar.gz 로 압축
    os.system(f"cd {package_dir} && pwd && tar -czvf source.tar.gz *") 
    
    logger.info(f"bucket name : {args.bucket}")
    bucket = args.bucket
    prefix='ncf'

    code_path = os.path.join(package_dir, 'source.tar.gz')
    logger.info(f"source.tar.gz path : {code_path}")

    # S3로 source.tar.gz 업로딩
    sagemaker_session  = sagemaker.session.Session()
    s3_code_uri = sagemaker_session.upload_data(code_path, bucket, prefix)
    logger.info(f"s3_code_uri : {s3_code_uri}")    
    
    return s3_code_uri

def store_s3_code_uri_json(s3_code_uri):
    '''
    json_file_name 안에 S3_URL 을 저장
    '''
 
    # Data to be written
    dictionary = {
        "s3_code_uri": s3_code_uri,
    }

    # Serializing json
    json_object = json.dumps(dictionary, indent=4)

    json_file_name = "code_location.json"    
    # Writing to sample.json
    with open(json_file_name, "w") as outfile:
        outfile.write(json_object)

    
    return json_file_name



def main(args):  
    
    clone_code_repository(args)
    s3_code_uri = upload_code_s3(args)
    store_s3_code_uri_json(s3_code_uri)
    
        

if __name__ == "__main__":
    args = parser_args()    
    main(args)
