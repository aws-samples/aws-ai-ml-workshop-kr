import argparse
import os
import requests
import tempfile
import subprocess, sys

import pandas as pd
import numpy as np
from glob import glob

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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


def split_train_test(df, test_ratio=0.1):
    '''
    두 개의 데이터 세트로 분리
    '''
    total_rows = df.shape[0]
    train_end = int(total_rows * (1 - test_ratio))
    
    train_df = df[0:train_end]
    test_df = df[train_end:]
    
    return train_df, test_df


def get_dataframe(base_preproc_input_dir, file_name_prefix ):    
    '''
    파일 이름이 들어가 있는 csv 파일을 모두 저장하여 데이터 프레임을 리턴
    '''
    
    input_files = glob('{}/{}*.csv'.format(base_preproc_input_dir, file_name_prefix))
    #claim_input_files = glob('{}/dataset*.csv'.format(base_preproc_input_dir))    
    logger.info(f"input_files: \n {input_files}")    
    
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(base_preproc_input_dir, "train"))
        
    raw_data = [ pd.read_csv(file, index_col=0) for file in input_files ]
    df = pd.concat(raw_data)
   
    logger.info(f"dataframe shape \n {df.shape}")    
    logger.info(f"dataset sample \n {df.head(2)}")        
    #logger.info(f"df columns \n {df.columns}")    
    
    return df


def convert_type(raw, cols, type_target):
    '''
    해당 데이터 타입으로 변경
    '''
    df = raw.copy()
    
    for col in cols:
        df[col] = df[col].astype(type_target)
    
    return df
    

if __name__ =='__main__':
    
    ################################
    #### 커맨드 인자 파싱   
    #################################        
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_output_dir', type=str, default="/opt/ml/processing/output")
    parser.add_argument('--base_preproc_input_dir', type=str, default="/opt/ml/processing/input")   
    parser.add_argument('--split_rate', type=float, default=0.1)       
    parser.add_argument('--label_column', type=str, default="fraud")       
    # parse arguments
    args = parser.parse_args()     
    
    logger.info("######### Argument Info ####################################")
    logger.info(f"args.base_output_dir: {args.base_output_dir}")
    logger.info(f"args.base_preproc_input_dir: {args.base_preproc_input_dir}")    
    logger.info(f"args.label_column: {args.label_column}")        
    logger.info(f"args.split_rate: {args.split_rate}")            

    base_output_dir = args.base_output_dir
    base_preproc_input_dir = args.base_preproc_input_dir
    label_column = args.label_column    
    split_rate = args.split_rate

    #################################        
    #### 두개의 파일(claim, customer) 을 로딩하여 policy_id 로 조인함  ########
    #################################    
    
    logger.info(f"\n### Loading Claim Dataset")
    claim_df = get_dataframe(base_preproc_input_dir,file_name_prefix='claim' )        
    
    logger.info(f"\n### Loading Customer Dataset")    
    customer_df = get_dataframe(base_preproc_input_dir,file_name_prefix='customer' )            
    
    df = customer_df.join(claim_df, how='left')
    logger.info(f"### dataframe merged with customer and claim: {df.shape}")


    #################################    
    #### 카테고리 피쳐를 원핫인코딩  
    #################################    
    
    logger.info(f"\n ### Encoding: Category Features")    
    categorical_features = df.select_dtypes(include=['object']).columns.values.tolist()    
    #categorical_features = ['driver_relationship']    
    logger.info(f"categorical_features: {categorical_features}")            

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )
    
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features)
        ],
        sparse_threshold = 0, # dense format 으로 제공
    )

    X_pre_category = preprocess.fit_transform(df)
    

    # 원핫인코딩한 컬럼의 이름 로딩
    # Ref: Sklearn Pipeline: Get feature names after OneHotEncode In ColumnTransformer,  https://stackoverflow.com/questions/54646709/sklearn-pipeline-get-feature-names-after-onehotencode-in-columntransformer
    
    processed_category_features = preprocess.transformers_[0][1].named_steps['onehot'].get_feature_names(categorical_features)
    #logger.info(f"processed_category_features: {processed_category_features}")
#    print(X_pre)
    
    ###############################
    ### 숫자형 변수 전처리 
    ###############################
    
    logger.info(f"\n ### Encoding: Numeric Features")        
    
    float_cols = df.select_dtypes(include=['float64']).columns.values
    int_cols = df.select_dtypes(include=['int64']).columns.values
    numeric_features = np.concatenate((float_cols, int_cols), axis=0).tolist()
    
    logger.info(f"int_cols: \n{int_cols}")    
    logger.info(f"float_cols: \n{float_cols}")        
    #logger.info(f"numeric_features: \n{numeric_features}")

    # 따로 스케일링은 하지 않고, 미싱 값만 중간값을 취함
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
           # ("scaler", StandardScaler())
        ]
    )

    numeric_preprocessor = ColumnTransformer(
        transformers=[
            ("cat", numeric_transformer, numeric_features)
        ],
        sparse_threshold = 0,
    )

    X_pre_numeric = numeric_preprocessor.fit_transform(df)    

    
    ###############################
    ### 전처리 결과 결합 ####
    ###############################
    
    logger.info(f"\n ### Handle preprocess results")            
    
    # 전처리 결과를 데이터 프레임으로 생성
    category_df = pd.DataFrame(data=X_pre_category, columns=processed_category_features)
    numeric_df = pd.DataFrame(data=X_pre_numeric, columns=numeric_features)    

    full_df = pd.concat([numeric_df, category_df ], axis=1)
    
    # float 타입을 int 로 변경
    full_df = convert_type(full_df, cols=int_cols, type_target='int')
    full_df = convert_type(full_df, cols=processed_category_features, type_target='int')    
    
    # label_column을 맨 앞으로 이동 시킴
    full_df = pd.concat([full_df[label_column], full_df.drop(columns=[label_column])], axis=1)
    
    ###############################    
    # 훈련, 테스트 데이터 세트로 분리 및 저장
    ###############################
    
    train_df, test_df = split_train_test(full_df, test_ratio=split_rate)    
    train_df.to_csv(f"{base_output_dir}/train/train.csv", index=False)
    test_df.to_csv(f"{base_output_dir}/test/test.csv", index=False)    

    logger.info(f"preprocessed train shape \n {train_df.shape}")        
    logger.info(f"preprocessed test shape \n {test_df.shape}")            

    # logger.info(f"preprocessed train path \n {base_output_dir}/train/train.csv")
    logger.info(f"\n ### Final result for train dataset ")    
    logger.info(f"preprocessed train sample \n {train_df.head(2)}")


    
