###########################################################################################################################
'''
훈련 코드는 크게 아래와 같이 구성 되어 있습니다.
- 커맨드 인자로 전달된 변수 내용 확인
- 훈련 데이터 로딩 
- xgboost의 cross-validation(cv) 로 훈련
- 훈련 및 검증 데이터 세트의 roc-auc 값을 metrics_data 에 저장
    - 모델 훈련시 생성되는 지표(예: loss 등)는 크게 두가지 방식으로 사용 가능
        - 클라우드 워치에서 실시간으로 지표 확인
        - 하이퍼 파라미터 튜닝(HPO) 에서 평가 지표로 사용 (예: validation:roc-auc)
        - 참조 --> https://docs.amazonaws.cn/en_us/sagemaker/latest/dg/training-metrics.html
        - 참조: XGBoost Framework 에는 이미 디폴트로 정의된 metric definition이 있어서, 정의된 규칙에 따라서 모델 훈련시에 print() 를 하게 되면, 
               metric 이 클라우드 워치 혹은 HPO에서 사용이 가능
           
Name                Regex
validation:auc	.*\[[0-9]+\].*#011validation-auc:([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*
train:auc	    .*\[[0-9]+\].*#011train-auc:([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*

실제 코드에 위의 Regex 형태로 print() 반영
print(f"[0]#011train-auc:{train_auc_mean}")
print(f"[1]#011validation-auc:{validation_auc_mean}")
    
- 훈련 성능을 나타내는 지표를 저장 (metrics.json)
- 훈련이 모델 아티펙트를 저장

'''
###########################################################################################################################



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

def train_sagemaker(args):
    if os.environ.get('SM_CURRENT_HOST') is not None:
        args.train_data_path = os.environ.get('SM_CHANNEL_TRAIN')
        args.model_dir = os.environ.get('SM_MODEL_DIR')
        args.output_data_dir = os.environ.get('SM_OUTPUT_DATA_DIR')
    return args

def main():
    parser = argparse.ArgumentParser()

    ###################################
    ## 커맨드 인자 처리
    ###################################    
    
    # Hyperparameters are described here
    parser.add_argument('--scale_pos_weight', type=int, default=50)    
    parser.add_argument('--num_round', type=int, default=999)
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--nfold', type=int, default=5)
    parser.add_argument('--early_stopping_rounds', type=int, default=10)
    parser.add_argument('--train_data_path', type=str, default='../dataset')

    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--model-dir', type=str, default='../model')
    parser.add_argument('--output-data-dir', type=str, default='../output')

    args = parser.parse_args()
    
    ## Check Training Sagemaker
    args = train_sagemaker(args)
    
    ###################################
    ## 데이터 세트 로딩 및 변환
    ###################################        

    data = pd.read_csv(f'{args.train_data_path}/train/train.csv')
    train = data.drop('fraud', axis=1)
    label = pd.DataFrame(data['fraud'])
    dtrain = xgb.DMatrix(train, label=label)
    
    ###################################
    ## 하이퍼파라미터 설정
    ###################################        
    
    params = {'max_depth': args.max_depth, 'eta': args.eta, 'objective': args.objective, 'scale_pos_weight': args.scale_pos_weight}
    
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
    
#     for i in cv_results.index:
#         train_auc_mean = cv_results['train-auc-mean'][i]
#         train_auc_std = cv_results['train-auc-std'][i]
#         test_auc_mean = cv_results['test-auc-mean'][i]
#         test_auc_std = cv_results['test-auc-std'][i]

#         print(f" train_auc_mean : {train_auc_mean}, train_auc_std : {train_auc_std}, test_auc_mean : {test_auc_mean}, test_auc_std : {test_auc_std}, ")

    # Select the best score
    print(f"[0]#011train-auc:{cv_results.iloc[-1]['train-auc-mean']}")
    print(f"[1]#011validation-auc:{cv_results.iloc[-1]['test-auc-mean']}")
    
    metrics_data = {
        'classification_metrics': {
            'validation:auc': { 'value': cv_results.iloc[-1]['test-auc-mean']},
            'train:auc': {'value': cv_results.iloc[-1]['train-auc-mean']}
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
    os.makedirs(args.output_data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    metrics_location = args.output_data_dir + '/metrics.json'
    model_location = args.model_dir + '/xgboost-model'
    
    
    with open(metrics_location, 'w') as f:
        json.dump(metrics_data, f)
    
    model.save_model(model_location)



if __name__ == '__main__':
    main()


        
