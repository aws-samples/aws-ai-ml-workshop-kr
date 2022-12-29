import argparse
import os
import json
import sys
sys.path.append('./src')

from train_lib import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##################################
    #### 세이지 메이커 프레임워크의 도커 컨테이너 환경 변수 인자
    ##################################

    parser.add_argument('--train-data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    parser.add_argument('--test-data-dir', type=str, default=os.environ['SM_CHANNEL_TEST'])

    
        
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])    
           
    
    ##################################
    #### 사용자 정의 커맨드 인자
    ##################################

    parser.add_argument("--lr", 
        type=float, 
        default=0.001, 
        help="learning rate")
    parser.add_argument("--dropout", 
        type=float,
        default=0.0,  
        help="dropout rate")
    parser.add_argument("--batch_size", 
        type=int, 
        default=256, 
        help="batch size for training")
    parser.add_argument("--epochs", 
        type=int,
        default=20,  
        help="training epoches")
    parser.add_argument("--top_k", 
        type=int, 
        default=10, 
        help="compute metrics@top_k")
    parser.add_argument("--factor_num", 
        type=int,
        default=32, 
        help="predictive factors numbers in the model")
    parser.add_argument("--num_layers", 
        type=int,
        default=3, 
        help="number of layers in MLP model")
    parser.add_argument("--num_ng", 
        type=int,
        default=4, 
        help="sample negative items for training")
    parser.add_argument("--test_num_ng", 
        type=int,
        default=99, 
        help="sample part of negative items for testing")
    parser.add_argument("--out", 
        default=True,
        help="save model or not")
    parser.add_argument("--gpu", 
        type=str,
        default="0",  
        help="gpu card ID")
    args = parser.parse_args()
    

    ##################################
    #### 훈련 함수 콜
    ##################################
    
    train(args)

