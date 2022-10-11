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

from sklearn.metrics import mean_squared_error

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
    parser.add_argument('--output_evaluation_dir', type=str, default="/opt/ml/processing/output")

    
        
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
    
    
    # Traverse all files
    logger.info(f"****** All folder and files under {base_dir} ****** ")
    for file in os.walk(base_dir):
        logger.info(f"{file}")
    logger.info(f"************************************************* ")        

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
    
    model = xgboost.XGBRegressor()
    model.load_model("xgboost-model")
    logger.info(f"model is loaded")    
    

    df = pd.read_csv(test_path)
    logger.info(f"test df sample \n: {df.head(2)}")        
    
#    y_test = df.iloc[:, 0].astype('int').to_numpy()
    y_test = df.iloc[:, 0].astype('int')    
    df.drop(df.columns[0], axis=1, inplace=True)
    
#     X_test = xgboost.DMatrix(df.values)
    X_test = df.values
    
    predictions_prob = model.predict(X_test)
    
    # if predictions_prob is greater than 0.5, it is 1 as a fruad, otherwise it is 0 as a non-fraud
    threshold = 0.5
    predictions = [1 if e >= 0.5 else 0 for e in predictions_prob ] 
    
#     print("y_test: ", y_test)
#     print("y_test length: ", len(y_test))    
#     print("predctions length: ", len(predictions))
#     print("predctions: ", predictions)    
#     print("predictions: ", predictions)
    # logging.info(f"{classification_report(y_true=y_test, y_pred = predictions)}")
    print(f"{classification_report(y_true=y_test, y_pred = predictions)}")
    cm = confusion_matrix(y_true= y_test, y_pred= predictions)    
    print(cm)

    mse = mean_squared_error(y_test, predictions)
    std = np.std(y_test - predictions)
    report_dict = {
        "regression_metrics": {
            "mse": {
                "value": mse,
                "standard_deviation": std
            },
        },
    }


    pathlib.Path(output_evaluation_dir).mkdir(parents=True, exist_ok=True)
    
    evaluation_path = f"{output_evaluation_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
        
    logger.info(f"evaluation_path \n: {evaluation_path}")                
    logger.info(f"report_dict \n: {report_dict}")                    
        
                