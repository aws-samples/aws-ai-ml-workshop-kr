import logging, sys, os
import numpy as np

import io

import torch
# import torch.distributed as dist
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.parallel
# import torch.optim
# import torch.utils.data
# import torch.utils.data.distributed

# import subprocess

# import sys
# sys.path.append('.')

# subprocess.call(['pip', 'install', 'sagemaker_inference'])
# from sagemaker_inference import content_types, decoder

import logging
import json
import traceback
from common_utils import load_json
import model as model
import numpy as np    


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


# 파이토치 서브의 디폴트 model_fn, input_fn 코드
# https://github.com/aws/sagemaker-pytorch-inference-toolkit/blob/master/src/sagemaker_pytorch_serving_container/default_pytorch_inference_handler.py


def model_fn(model_dir):
    print("######## Staring model_fn() ###############")
    logger.info("--> model_dir : {}".format(model_dir))
    
    try:
        # 디바이스 할당
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 컨피그 파일 로딩
        model_config_path = os.path.join(os.path.dirname(__file__), 'model_config.json')
        logger.info(f"model_config_path: :  {model_config_path}")                 
    
        model_config_dict = load_json(model_config_path)        
    
        model_user_num = int(model_config_dict["user_num"])
        model_item_num = int(model_config_dict["item_num"])
        model_factor_num = int(model_config_dict["factor_num"])
        model_num_layers = int(model_config_dict["num_layers"])
        model_dropout = float(model_config_dict["dropout"])
        model_type = model_config_dict["model_type"]

        # 모델 네트워크 로딩
        inf_model = model.NCF(model_user_num, model_item_num, model_factor_num, 
                              model_num_layers, 
                              model_dropout, model_type, GMF_model=None, MLP_model=None)

        logger.info("--> model network is loaded")    
        
        # 모델 아티펙트 경로       
        model_file_path = os.path.join(model_dir, "NeuMF-end.pth")
        logger.info("model_file_path: :  {model_file_path}")                      
        
        # 모델 가중치 로딩    
        with open(model_file_path, "rb") as f:
              inf_model.load_state_dict(torch.load(f))            
        logger.info(f"####### Model is loaded #########")        

    except Exception:
        logger.info("---> ########## Failure loading a Model #######")                
        print(traceback.format_exc())

            
    return inf_model.to(device)


def input_fn(input_data, content_type):
    '''
    content_type == 'application/x-npy' 일시에 토치 텐서의 변환 작업 수행
    '''
    logger.info("#### input_fn starting ######")
    logger.info(f"content_type: {content_type}")    

    try:
        # 디바이스 할당
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        logger.info(f"#### type of input data: {type(input_data)}")                              
        if isinstance(input_data, str):
            pass
        elif isinstance(input_data, io.BytesIO):
            input_data = input_data.read()
            input_data = bytes.decode(input_data)        
        elif isinstance(input_data, bytes):
            input_data = input_data.decode()

        data = json.loads(input_data)

        user_np = np.asarray(data['user'])
        item_np = np.asarray(data['item'])   
        
        user = torch.from_numpy(user_np)
        item = torch.from_numpy(item_np)
        
        user = user.to(device)
        item = item.to(device)
        
        payload = [user, item]        
    
    except Exception:
        print(traceback.format_exc())        
    
        
    return payload


    

def predict_fn(data, model):
    '''
    모델의 추론 함수
    '''
    logger.info("#### predict_fn starting ######")    
    # 디바이스 할당
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    
        logger.info(f"#### type of input data: {type(data)}")                                  
    
        user = data[0]
        item = data[1]
        
        # 모델 추론
        model.eval()
        with torch.no_grad():
            predictions = model(user, item)

        predictions_numpy = predictions.detach().cpu().numpy()
        # print("predictions: ", predictions_numpy.shape)

        top_k = 10
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(item, indices).cpu().numpy().tolist()
        logger.info(f"recommends:  {recommends}")
        
    except Exception:
        print(traceback.format_exc())        
        
    
    return recommends




def predict_fn2(data, model):
    '''
    모델의 추론 함수
    '''
    logger.info("#### predict_fn starting ######")    
    # 디바이스 할당
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("#### Test 0 ######")    
    print("#### type of data initially, ", type(data))    
    
    data = json.loads(data)

    print("#### Test 1 ######")      
    
    print("#### type of data, ", type(data))
    print("##### data: ", data)
    
    user_np = np.asarray(data['user'])
    item_np = np.asarray(data['item'])   
    
    print("#### Test 2 ######")    
    
    user = torch.from_numpy(user_np)
    item = torch.from_numpy(item_np)
    
    print("#### Test 3 ######")        
    
    user = user.to(device)
    item = item.to(device)
    
    print("#### Test ######")
    # 모델 추론
    model.eval()
    with torch.no_grad():
        predictions = model(user, item)
        
    predictions_numpy = predictions.detach().cpu().numpy()
    # print("predictions: ", predictions_numpy.shape)

    top_k = 10
    _, indices = torch.topk(predictions, top_k)
    recommends = torch.take(item, indices).cpu().numpy().tolist()
    print("recommends: ", recommends)
    
    return recommends


    

# Serialize the prediction result into the response content type
# def output_fn(prediction, accept='application/json'):
#     logger.info("############# Staring output_fn() #################")
#     logger.info(f"prediction type : {type(prediction)}")
#     if type(prediction) == torch.Tensor:    
#         result = prediction.cpu().detach().numpy()
    
#     result = prediction
    
#     prediction_dict = {'prediction': result}
#     return json.dumps(prediction_dict)
    
# def output_fn(prediction, accept):
#     LOGGER.info("############# Staring output_fn() #################")
#     LOGGER.info(f"accept : {accept}")
    
#     if type(prediction) == torch.Tensor:
#             prediction = prediction.detach().cpu().numpy().tolist()
    
#     encoded_prediction = encoder.encode(prediction, accept)
    
#     return encoded_prediction    


