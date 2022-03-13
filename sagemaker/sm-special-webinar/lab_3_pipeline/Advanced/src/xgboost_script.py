import os
import sys
import pickle
import xgboost as xgb
import argparse
import pandas as pd
import json

import pandas as pd
pd.options.display.max_rows=20
pd.options.display.max_columns=10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ###################################
    ## 커맨드 인자 처리
    ###################################    
    
    # Hyperparameters are described here
    parser.add_argument('--scale_pos_weight', type=int, default=50)    
    parser.add_argument('--num_round', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--min_child_weight', type=float, default=1.0)    
    parser.add_argument('--alpha', type=float, default=0.0)        
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--nfold', type=int, default=5)
    parser.add_argument('--early_stopping_rounds', type=int, default=10)
    parser.add_argument('--train_data_path', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))

    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))

    args = parser.parse_args()
    
    ###################################
    ## 데이터 세트 로딩 및 변환
    ###################################        

    data = pd.read_csv(f'{args.train_data_path}/train.csv')
    train = data.drop('fraud', axis=1)
    label = pd.DataFrame(data['fraud'])
    dtrain = xgb.DMatrix(train, label=label)
    
    ###################################
    ## 하이퍼파라미터 설정
    ###################################        
    
    params = {'max_depth': args.max_depth, 
              'eta': args.eta, 
              'num_round' : args.num_round,
              'min_child_weight' : args.min_child_weight,
              'alpha' : args.alpha,
              'objective': args.objective, 
              'scale_pos_weight': args.scale_pos_weight}
    
    num_boost_round = args.num_round
    nfold = args.nfold
    early_stopping_rounds = args.early_stopping_rounds

    ###################################
    ## Cross-Validation으로 훈련하여, 훈련 및 검증 메트릭 추출
    ###################################            
    
    cv_results = xgb.cv(
        params = params,
        dtrain = dtrain,
        num_boost_round = num_boost_round,
        nfold = nfold,
        early_stopping_rounds = early_stopping_rounds,
        metrics = ('auc'),
        stratified = True, # 레이블 (0,1) 의 분포에 따라 훈련 , 검증 세트 분리
        seed = 0
        )
    
    ###################################
    ## 훈련 및 검증 데이터 세트의 roc-auc 값을 metrics_data 에 저장
    ###################################            

    
    print("cv_results: ", cv_results)

    # Select the best score
    train_auc_mean = round(cv_results.iloc[-1]['train-auc-mean'],4)
    validation_auc_mean = round(cv_results.iloc[-1]['test-auc-mean'],4)
    
    # XGBoost Framework 에는 이미 정의된 metric definition이 있습니다. 
    # 참조: Monitor and Analyze Training Jobs Using Metrics (https://docs.amazonaws.cn/en_us/sagemaker/latest/dg/training-metrics.html)
    # 아래와 같은 형태로 print() 를 하게 되면, metric 이 클라우드 워치 혹은 HPO에서 사용이 가능합니다.
    # Name				Regex
    # validation:auc	.*\[[0-9]+\].*#011validation-auc:([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*
    # train:auc	.*\[[0-9]+\].*#011train-auc:([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*
    
    print("Refer to CloudWatch to see the following metrics")
    print(f"[0]#011train-auc:{train_auc_mean}")
    print(f"[1]#011validation-auc:{validation_auc_mean}")

    
    metrics_data = {
        'classification_metrics': {
            'validation:auc': { 'value': train_auc_mean},
            'train:auc': {'value': validation_auc_mean}
        }
    }
    
    ###################################
    ## 오직 훈련 데이터 만으로 훈련하여 모델 생성
    ###################################            

    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=len(cv_results))

    ###################################
    ## 모델 아티펙트 및 훈련/검증 지표를 저장
    ###################################            
    
    # Save the model to the location specified by ``model_dir``
    metrics_location = args.output_data_dir + '/metrics.json'
    model_location = args.model_dir + '/xgboost-model'
    
    
    with open(metrics_location, 'w') as f:
        json.dump(metrics_data, f)
    
    with open(model_location, 'wb') as f:
        pickle.dump(model, f)

