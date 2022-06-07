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

from autogluon.tabular import TabularPredictor

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
        tar.extractall(path="./autogluon_model/")
    
    model = TabularPredictor.load("./autogluon_model")
    logger.info(f"model is loaded")    
    

    data = pd.read_csv(test_path)
    logger.info(f"test df sample \n: {data.head(2)}")        
    
#    y_test = df.iloc[:, 0].astype('int').to_numpy()
    y_true = data.iloc[:, 0].astype('int')    
    data.drop(data.columns[0], axis=1, inplace=True)

    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    prediction = pd.concat([pred, pred_proba], axis=1)

    print(f"{(pred==y_true).astype(int).sum()}/{len(pred)} are correct")
    # report = classification_report(y_true=y_true, y_pred=pred, target_names=['No fraud','fraud'])
    # # report_df = pd.DataFrame(report).transpose()
    # print(report)
    # cm = confusion_matrix(y_true=y_true, y_pred=pred)    
    # print(cm)

    pathlib.Path(output_evaluation_dir).mkdir(parents=True, exist_ok=True)
    
    evaluation_path = f"{output_evaluation_dir}/prediction.json"
    with open(evaluation_path, "w") as f:
        f.write(prediction.to_json())
        
    logger.info(f"evaluation_path \n: {evaluation_path}")              
        
                