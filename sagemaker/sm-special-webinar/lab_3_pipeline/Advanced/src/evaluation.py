import json
import pathlib
import pickle
import tarfile

import joblib
import numpy as np
import pandas as pd
import argparse
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost'])

import xgboost

from sklearn.metrics import roc_auc_score

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', type=str, default= "/opt/ml/processing")    
    parser.add_argument('--model_path', type=str, default= "/opt/ml/processing/model/model.tar.gz")
    parser.add_argument('--test_path', type=str, default= "/opt/ml/processing/test/test.csv")
    parser.add_argument('--output_evaluation_dir', type=str, default="/opt/ml/processing/evaluation")

    
    ####################################
    ## 기본 커맨드 인자 파싱
    ####################################    
        
    # parse arguments
    args = parser.parse_args()     
    
    logger.info("#############################################")
    logger.info(f"args.model_path: {args.model_path}")
    logger.info(f"args.test_path: {args.test_path}")    
    logger.info(f"args.output_evaluation_dir: {args.output_evaluation_dir}")        

    model_path = args.model_path
    test_path = args.test_path    
    output_evaluation_dir = args.output_evaluation_dir
    base_dir = args.base_dir
    
    ####################################
    ## 기본 폴더 이하의 파일 폴더 구조 확인 하기
    ####################################    
    
    # Traverse all files
    logger.info(f"****** All folder and files under {base_dir} ****** ")
    for file in os.walk(base_dir):
        logger.info(f"{file}")
    logger.info(f"************************************************* ")        

    ####################################
    ## 모델 아티펙트의 압축 해제하고, 모델 로딩하기
    ####################################    
    
    with tarfile.open(model_path) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=".")
    
    model = pickle.load(open("xgboost-model", "rb"))
    logger.info(f"model is loaded")    
    

    ####################################
    ## 테스트 데이터 로딩하기
    ####################################        
    
    df = pd.read_csv(test_path)
    logger.info(f"test df sample \n: {df.head(2)}")        
    
    y_test = df.iloc[:, 0].astype('int')    
    df.drop(df.columns[0], axis=1, inplace=True)

    X_test = xgboost.DMatrix(df)
    print("Payload: \n", df.values)
    
    ####################################
    ## 모델 추론하기
    ####################################            
    
    predictions_prob = model.predict(X_test)
    

    ####################################
    ## 스코어 값을 0.5 를 기준으로 레이블 0, 1로 구분 
    ####################################            

    
    # if predictions_prob is greater than 0.5, it is 1 as a fruad, otherwise it is 0 as a non-fraud
    threshold = 0.5
    predictions = [1 if e >= 0.5 else 0 for e in predictions_prob ] 
    
    ####################################
    ## 평가 지표 보여주기
    ####################################            

    
    print(f"{classification_report(y_true=y_test, y_pred = predictions)}")
    cm = confusion_matrix(y_true= y_test, y_pred= predictions)    
    print(cm)

    roc_score = round(roc_auc_score(y_true = y_test, y_score = predictions_prob ), 4)
    print("roc_auc_score : ", roc_score)
    
    ####################################
    ## JSON 으로 모델 평가 결과를 저장하기
    ####################################                
    # 참고: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
    
    report_dict = {
        "binary_classification_metrics": {
            "auc": {
                "value": roc_score,
                "standard_deviation" : "NaN",
            },
        },
    }


    pathlib.Path(output_evaluation_dir).mkdir(parents=True, exist_ok=True)
    
    evaluation_path = f"{output_evaluation_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
        for file in os.walk(output_evaluation_dir):
            logger.info(f"{file}")

    logger.info(f"evaluation_path \n: {evaluation_path}")                
    logger.info(f"report_dict \n: {report_dict}")                    
        

        
        
                