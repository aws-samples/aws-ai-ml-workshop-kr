import argparse
import os
from pprint import pprint

import yaml
from autogluon.tabular import TabularDataset, TabularPredictor


def train_sagemaker(args):
    if os.environ.get('SM_CURRENT_HOST') is not None:
        args.train_data_path = os.environ.get('SM_CHANNEL_INPUTDATA')
        args.model_dir = os.environ.get('SM_MODEL_DIR')
        args.output_data_dir = os.environ.get('SM_OUTPUT_DATA_DIR')
        args.ag_config = os.environ.get('SM_CHANNEL_CONFIG')
    return args


def get_input_path(path):
    file = os.listdir(path)[0]
    print(f"file : {file}")
    if len(os.listdir(path)) > 1:
        print(f"WARN: more than one file is found in {path} directory")
    print(f"Using {file}")
    filename = f"{path}/{file}"
    return filename


def main():
    
    # Disable Autotune
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    
    # ------------------------------------------------------------ Args parsing
    print("Starting AG")
    parser = argparse.ArgumentParser()

    ###################################
    ## 커맨드 인자 처리
    ###################################    
    
    # Hyperparameters are described here
    parser.add_argument('--train_data_path', type=str, default='../../data/dataset')
    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--model-dir', type=str, default='../model')
    parser.add_argument('--output-data-dir', type=str, default='../output')
    parser.add_argument("--ag_config", type=str, default="")
    parser.add_argument("--config_name", type=str, default="config-med.yaml")

    args = parser.parse_args()
    
    ## Check Training Sagemaker
    args = train_sagemaker(args)

    # config_file = get_input_path(args.ag_config)
    config_file = args.ag_config + f"/{args.config_name}"
    print(f"args.ag_config : {config_file}")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)  # AutoGluon-specific config    
    
    ###################################
    ## 데이터 세트 로딩 및 변환
    ###################################
    # train_file = get_input_path(args.train_data_path)
    train_file = args.train_data_path + "/train.csv"
    train_data = TabularDataset(train_file)

    ###################################
    ## 훈련
    ###################################
    
    ag_predictor_args = config["ag_predictor_args"]
    ag_predictor_args["path"] = args.model_dir
    ag_fit_args = config["ag_fit_args"]

    predictor = TabularPredictor(**ag_predictor_args).fit(train_data, **ag_fit_args)

    # --------------------------------------------------------------- Inference
    
    ###################################
    ## 검증, 모델 훈련/검증 지표를 저장
    ###################################        
    # Save the model to the location specified by ``model_dir``
    os.makedirs(args.output_data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    

    # test_file = get_input_path(args.test_data_path)
    test_file = args.train_data_path + "/test.csv"
    test_data = TabularDataset(test_file)

    # Predictions
    y_pred_prob = predictor.predict_proba(test_data)
    if config.get("output_prediction_format", "csv") == "parquet":
        y_pred_prob.to_parquet(f"{args.output_data_dir}/predictions.parquet")
    else:
        y_pred_prob.to_csv(f"{args.output_data_dir}/predictions.csv")

    # Leaderboard
    if config.get("leaderboard", False):
        lb = predictor.leaderboard(test_data, silent=False)
        lb.to_csv(f"{args.output_data_dir}/leaderboard.csv")

    # Feature importance
    if config.get("feature_importance", False):
        feature_importance = predictor.feature_importance(test_data)
        feature_importance.to_csv(f"{args.output_data_dir}/feature_importance.csv")
            

    ###################################
    ## 모델 저장
    ###################################            
#     model_location = args.model_dir + '/xgboost-model'
    
#     model.save_model(model_location)



if __name__ == '__main__':
    main()


        
