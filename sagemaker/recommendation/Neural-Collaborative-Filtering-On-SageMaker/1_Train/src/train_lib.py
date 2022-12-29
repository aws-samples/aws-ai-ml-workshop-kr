import os
import time
import numpy as np
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
 
# Custom Module
import model
import config
import evaluate
from evaluate import ndcg, hit
import data_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.WARNING)
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(args):
    '''
    1. args 를 받아서 입력 데이터 로딩
    2. 데이터 세트 생성
    3. 모델 네트워크 생성
    4. 훈련 푸프 실행
    5. 모델 저장
    '''
    #######################################
    ## 환경 확인     
    #######################################
    
    logger.info("##### Args: \n {}".format(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug("device: ".format(device))    

    
    #######################################
    ## 데이타 저장 위치 및 모델 경로 저장 위치 확인
    #######################################
    
    #### Parsing argument  ##########################        
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir    
    model_dir = args.model_dir        
    
    logger.info("args.train_data_dir: ".format(train_data_dir))
    logger.info("args.test_data_dir: ".format(test_data_dir))  
    logger.info("args.model_dir: ".format(model_dir))  
    
    train_rating_path = os.path.join(train_data_dir, 'ml-1m.train.rating')
    test_negative_path = os.path.join(train_data_dir, 'ml-1m.test.negative')

    
    #######################################
    ## 데이터 로딩 및 데이터 세트 생성 
    #######################################

    logger.info("=====> data loading <===========")        
    
    train_loader, user_num, item_num = _get_train_data_loader(args, train_rating_path, test_negative_path)
    test_loader =  _get_test_data_loader(args, train_rating_path, test_negative_path)
    

    #######################################
    ## 모델 네트워크 생성
    #######################################
    
    NCF_model = load_model_network(user_num, item_num, args)            
    NCF_model.to(device)    
    
    #######################################
    ## 손실 함수 및 옵티마이저 정의
    #######################################    
    
    if config.model == 'NeuMF-pre':
        optimizer = optim.SGD(NCF_model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(NCF_model.parameters(), lr=args.lr)


    #######################################
    ## 훈련 루프 실행
    #######################################
    
    count, best_hr = 0, 0
    train_loader.dataset.ng_sample()    
    print("=====> Starting New Traiing <===========")

    for epoch in range(args.epochs):
        start_time = time.time()

        train_epoch(NCF_model, train_loader, optimizer, epoch, device, sampler=None)            
        
        elapsed_time = time.time() - start_time    
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
                    time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))

        
        best_hr, best_ndcg, best_epoch = test(args, NCF_model, epoch, test_loader, best_hr, model_dir)
    
        
        
    print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
                                        best_epoch, best_hr, best_ndcg))

        

def _get_train_data_loader(args, train_rating_path, test_negative_path):
    '''
    데이터 로더를 제공
        1. 훈련, 테스트 데이터 로딩
        2. 커스텀 데이터 셋 생성
        3. 데이터 로더 생성
            - 데이터 로더의 Shuffle = False 는 추후에 데이터 셋을 Shuffle 할 예정이기에 False 로 설정.
    '''
    logger.info("Get train data sampler and data loader")
    
    train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all_script(train_rating_path, test_negative_path)
    

    train_dataset = data_utils.NCFData(
            train_data, item_num, train_mat, args.num_ng, True)


    train_loader = data.DataLoader(train_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    
    return train_loader, user_num, item_num

def _get_test_data_loader(args, train_rating_path, test_negative_path):
    '''
    train_loader 와 비슷하게 test_loader 를 생성
    '''
    logger.info("Get test data sampler and data loader")    
    
    train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all_script(train_rating_path, test_negative_path)
    
    test_dataset = data_utils.NCFData(
            test_data, item_num, train_mat, 0, False)
    
        
    test_loader = data.DataLoader(test_dataset,
            batch_size=args.test_num_ng+1, shuffle=False, num_workers=1)

    
    return test_loader

        
def load_model_network(user_num, item_num, args):        
    '''
    모델 네트워크를 로딩
    '''
    if config.model == 'NeuMF-pre':
        assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
        assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
        GMF_model = torch.load(config.GMF_model_path)
        MLP_model = torch.load(config.MLP_model_path)
        logger.info("Pretrained model is used")        
    else:
        GMF_model = None
        MLP_model = None
        logger.info("Pretrained model is NOT used")            

    NCF_model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
                            args.dropout, config.model, GMF_model, MLP_model)

    return NCF_model


def test(args, model, epoch, test_loader, best_hr, model_dir):
    '''
    테스트 데이타로 추론하여 평가
    '''
    model.eval()
    with torch.no_grad():
        HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)
        print("HR={:.3f}; \t NDCG={:.3f};".format(np.mean(HR), np.mean(NDCG)))
        
    if HR > best_hr:
        print("best_hr: ", HR)        
        best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
        if args.out:
            if not os.path.exists(config.model_path):
                os.mkdir(config.model_path)
            ### Save Model 을 다른 곳에 저장
            _save_model(model, model_dir, f'{config.model}.pth')  
    else:
        best_ndcg , best_epoch = NDCG, epoch


    return best_hr, best_ndcg, best_epoch
    
    
def train_epoch(model, train_loader, optimizer, epoch, device, sampler=None):
    if sampler:
        sampler.set_epoch(epoch)
    
    start_time = time.time()
    model.train()
    
    for batch_idx, (user, item, label) in enumerate(train_loader,1):
        user = user.to(device)
        item = item.to(device)
        label = label.float().to(device)

        model.zero_grad()
        prediction = model(user, item)
        
        loss_function = nn.BCEWithLogitsLoss()        
        loss = loss_function(prediction, label)
        loss.backward()
        optimizer.step()
            
        if batch_idx % 1000 == 0:
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)] Loss={:.6f};".format(
                    epoch,
                    batch_idx * len(user),
                    len(train_loader.sampler),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    

def _save_model(model, model_dir, model_weight_file_name):
    path = os.path.join(model_dir, model_weight_file_name)
    print(f"the model is saved at {path}")    
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.state_dict(), path)

