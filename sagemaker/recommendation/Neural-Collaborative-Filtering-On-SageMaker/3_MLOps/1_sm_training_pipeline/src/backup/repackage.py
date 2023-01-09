import json
import pathlib
import tarfile

import joblib
import numpy as np
import pandas as pd
import argparse
import os
import sys

import logging
# import logging.handlers

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

def show_files_folder(folder):
    # Traverse all files
    for file in os.walk(folder):
        logger.info(f"{file}")

    return None

def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', type=str, default= "/opt/ml/processing")    
    parser.add_argument('--source_path', type=str, default= "/opt/ml/processing/source/source.tar.gz")    
    parser.add_argument('--model_path', type=str, default= "/opt/ml/processing/model/model.tar.gz")
    parser.add_argument('--repackage_dir', type=str, default="/opt/ml/processing/repackage")

    args = parser.parse_args()

    return args

def upload_code_s3(upload_file):
    '''
    훈련 코드를와 디펜던시 파일을 압축하여 S3에 업로드하고 S3 URL 을 제공
    '''
    import sagemaker

    bucket = sagemaker.Session().default_bucket()

        
    logger.info(f"bucket name : {bucket}")

    prefix='ncf'

    code_path = upload_file
    logger.info(f"source.tar.gz path : {code_path}")

    # S3로 source.tar.gz 업로딩
    sagemaker_session  = sagemaker.session.Session()
    s3_code_uri = sagemaker_session.upload_data(code_path, bucket, prefix)
    logger.info(f"s3_code_uri : {s3_code_uri}")    
    
    return s3_code_uri

def check_final_folder_structure(base_dir, upload_file):
    try:
        with tarfile.open(upload_file) as tar:
            temp2_dir = f"{base_dir}/temp2"
            os.makedirs(temp2_dir, exist_ok=True)
            tar.extractall(path= temp2_dir)

        logger.info("\n########## In temp2, untaring source artifact, repackage_dir is: ")
        show_files_folder(temp2_dir)            
    except Exception:
        print(traceback.format_exc())


def main(args):
    '''
    model.tar.gz 와 source.tar.gz 를 모두 압축 해제한 후에, 모두 합쳐서 다시 model.tar.gz 로 생성
    '''
    
    logger.info("#############################################")
    logger.info(f"args.base_dir: {args.base_dir}")    
    logger.info(f"args.source_path: {args.source_path}")    
    logger.info(f"args.model_path: {args.model_path}")
    logger.info(f"args.repackage_dir: {args.repackage_dir}")        

    source_path = args.source_path
    model_path = args.model_path    
    repackage_dir = args.repackage_dir
    base_dir = args.base_dir
    
    ####################################
    ## 기본 폴더 이하의 파일 폴더 구조 확인 하기
    ####################################    
    logger.info("\n### Initial Folder Status ")    
    show_files_folder(base_dir)
    
    

    ####################################
    ## Temp folder 생성
    ####################################    

    temp_dir = f"{base_dir}/temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    
    ####################################
    ## 모델 아티펙트의 압축 해제
    ####################################    
    
    with tarfile.open(model_path) as tar:
        tar.extractall(path= temp_dir)
    
    logger.info("\n### Folder Status After untaring model artifact ")    
    show_files_folder(temp_dir)    

    ####################################
    ## repackage_dir 밑에 code 폴더 생성
    ####################################    

    code_dir = f"{temp_dir}/code"
    os.makedirs(code_dir, exist_ok=True)
    
    ####################################
    ## 소스 아티펙트의 압축 해제
    ####################################    
    
    try:
        with tarfile.open(source_path) as tar:
            tar.extractall(path= code_dir)

        logger.info("\n########## After untaring source artifact, repackage_dir is: ")
        show_files_folder(temp_dir)            
    except:
        print("##########Error occureed")


    ####################################
    ## temp 의 파일을 압추갛여 repackage 에 저장
    ####################################    
    os.system(f"tar -czf {repackage_dir}/model.tar.gz -C {temp_dir} .")
    logger.info("\n########## After taring temp folder, final repackage_dir is: ")    
    show_files_folder(repackage_dir)                

    ####################################
    ## 옵션: 최종 압축 파일의 폴더 구조 확인 
    ####################################        
    upload_file = f"{repackage_dir}/model.tar.gz"
    check_final_folder_structure(base_dir, upload_file)

    
    
    

    
        # 권한 추가    
    # os.system(f"chmod -R 755 {repackage_dir}")

        
    return None

if __name__ == "__main__":
    args = parser_args()    
    main(args)

    

    

    

#     pathlib.Path(output_evaluation_dir).mkdir(parents=True, exist_ok=True)
    
#     evaluation_path = f"{output_evaluation_dir}/evaluation.json"
#     with open(evaluation_path, "w") as f:
#         f.write(json.dumps(report_dict))
#         for file in os.walk(output_evaluation_dir):
#             logger.info(f"{file}")

                
#     logger.info(f"evaluation_path \n: {evaluation_path}")                
#     logger.info(f"report_dict \n: {report_dict}")                    
        

        
        
                