import argparse
import os
import json
import time
import numpy as np
import logging


import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR

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


def train_sm_ddp(args):
    '''
    1. args 를 받아서 입력 데이터 로딩
    2. 데이터 세트 생성
    3. 모델 네트워크 생성
    4. 훈련 푸프 실행
    5. 모델 저장
    '''
    ######################    
    # DDP 코드: 1. 라이브러리 임포트
    ######################    

    import smdistributed.dataparallel.torch.distributed as dist
    from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP

    dist.init_process_group(backend='smddp')    


    ######################    
    # DDP 코드 : 2. 배치 사이즈 결정       
    # SageMaker data parallel: Scale batch size by world size
    ######################        
    
    batch_size = args.batch_size        
    batch_size //= dist.get_world_size()
    batch_size = max(batch_size, 1)
    if dist.get_rank() == 0:
        logger.info("################################")            
        logger.info(f"Global batch size: {args.batch_size}")            
        logger.info(f"each gpu batch_size: {batch_size}")

        
    ######################    
    # DDP 코드 : 3. 각 GPU 를 DDP LIb 프로세스에 할당      
    # SageMaker data parallel: Pin each GPU to a single library process.
    ######################        
    import os
    
    local_rank = dist.get_local_rank()    
    torch.cuda.set_device(local_rank) 
    
    if dist.get_rank() == 0:    
        logger.info(f"world size: {dist.get_world_size()}")



        
    #######################################
    ## 입력 매개 변수 및 환경 확인     
    #######################################
    if dist.get_rank() == 0:
        logger.info("##### Args: \n {}".format(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    #######################################
    ## 데이타 저장 위치 및 모델 경로 저장 위치 확인
    #######################################
    
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir    
    model_dir = args.model_dir        
    
    if dist.get_rank() == 0:
        logger.info("args.train_data_dir: ".format(train_data_dir))
        logger.info("args.test_data_dir: ".format(test_data_dir))  
        logger.info("args.model_dir: ".format(model_dir))  
    
    train_rating_path = os.path.join(train_data_dir, 'ml-1m.train.rating')
    test_negative_path = os.path.join(train_data_dir, 'ml-1m.test.negative')

    
    #######################################
    ## 데이터 로딩 및 데이터 세트 생성 
    #######################################

    if dist.get_rank() == 0:
        logger.info("=====> data loading <===========")        
    
    train_kwargs = {'num_workers': 4, 'pin_memory': True}
    test_kwargs = {'num_workers': 1, 'pin_memory': True}    
    
    train_loader, train_sampler, user_num, item_num = _get_train_data_loader(dist, args, train_rating_path, test_negative_path, **train_kwargs)
    test_loader =  _get_test_data_loader(dist, args, train_rating_path, test_negative_path, **test_kwargs)
    
    if dist.get_rank() == 0:    
        logger.info(
            "Processes {}/{} ({:.0f}%) of train data".format(
                len(train_loader.sampler),
                len(train_loader.dataset),
                100.0 * len(train_loader.sampler) / len(train_loader.dataset),
            )
        )

        logger.info(
            "Processes {}/{} ({:.0f}%) of test data".format(
                len(test_loader.sampler),
                len(test_loader.dataset),
                100.0 * len(test_loader.sampler) / len(test_loader.dataset),
            )
        )


    
    #######################################
    ## 모델 네트워크 생성
    #######################################
    
    NCF_model = load_model_network(dist, user_num, item_num, args)            
    NCF_model = DDP(NCF_model.to(device))
    NCF_model.cuda(local_rank)
    
    if dist.get_rank() == 0:
        logger.info("### Model loaded")    
    
    
    #######################################
    ## 손실 함수 및 옵티마이저 정의
    #######################################    
    
    if config.model == 'NeuMF-pre':
        optimizer = optim.SGD(NCF_model.parameters(), lr=args.lr)
    else:
        # 모델이 "NeuMF-End" 이기에 else 문이 선택 됨.
        optimizer = optim.Adam(NCF_model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    
        

    #######################################
    ## 훈련 루프 실행
    #######################################
    
    count, best_hr = 0, 0
    # Negative 샘플 생성 함
    train_loader.dataset.ng_sample()    

    for epoch in range(args.epochs):
        start_time = time.time()

        # 훈련 루프 실행
        train_epoch(dist, args, NCF_model, train_loader, optimizer, epoch, device, sampler=train_sampler)
        scheduler.step()    

        elapsed_time = time.time() - start_time    
        
        if dist.get_rank() == 0:
            print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
                    time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))

            best_hr, best_ndcg, best_epoch = test(args, NCF_model, epoch, test_loader, best_hr, model_dir)
    
        
    if dist.get_rank() == 0:
        print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
                                        best_epoch, best_hr, best_ndcg))

        

def _get_train_data_loader(dist, args, train_rating_path, test_negative_path, **kwargs):
    '''
    훈련 데이터 셋 생성 및 데이터 로더 생성
    '''
    if dist.get_rank() == 0:
        logger.info("Get train data sampler and data loader")
    
    train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all_script(train_rating_path, test_negative_path)
    

    train_dataset = data_utils.NCFData(
            train_data, item_num, train_mat, args.num_ng, True)


    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=dist.get_world_size(), rank= dist.get_rank()
    )

    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs 
    )

    
    return train_loader, train_sampler, user_num, item_num

def _get_test_data_loader(dist, args, train_rating_path, test_negative_path, **kwargs):
    '''
    테스트 데이터 셋 생성 및 데이터 로더 생성
    [중요] 테스트 로더에서는 sampler를 생성을 하지 않았음. 샘플러 생성시에 결과(HR, NDCG) 의 값이 낮게 나와서 삭제 함. (현재 이유를 정확히 모르겠음.)
    '''
    if dist.get_rank() == 0:
        logger.info("Get test data sampler and data loader")    
    
    train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all_script(train_rating_path, test_negative_path)
    

    test_dataset = data_utils.NCFData(
            test_data, item_num, train_mat, 0, False)
    
        
    test_loader = data.DataLoader(test_dataset,
            batch_size=args.test_num_ng+1, shuffle=False, **kwargs)

    
    return test_loader

        
def load_model_network(dist, user_num, item_num, args):        
    '''
    모델 네트워크 로딩 현재 Neuf-end 이기에 항상 else 문이 실행 됨
    '''
    if config.model == 'NeuMF-pre':
        assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
        assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
        GMF_model = torch.load(config.GMF_model_path)
        MLP_model = torch.load(config.MLP_model_path)
        if dist.get_rank() == 0:
            logger.info("Pretrained model is used")        
    else:
        GMF_model = None
        MLP_model = None
        if dist.get_rank() == 0:
            logger.info("Pretrained model is NOT used")            

    NCF_model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
                            args.dropout, config.model, GMF_model, MLP_model)

    return NCF_model


    

def test(args, model, epoch, test_loader, best_hr, model_dir):
    '''
    테스트 데이터로 모델 평가를 함.
    best_hr 이 나오면 모델 저장을 함.
    '''
    model.eval()
    with torch.no_grad():
        HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)
        print("HR={:.3f}; \t NDCG={:.3f};".format(HR, NDCG))        
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
    
    
def train_epoch(dist, args, model, train_loader, optimizer, epoch, device, sampler=None):
    '''
    훈련 루프 실행
    '''
    # Horovod: set epoch to sampler for shuffling.    
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

        if (dist.get_rank() == 0) & (batch_idx % args.log_interval == 0):
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
    '''
    model_dir (예: /opt/ml/model) 에 모델 저장
    '''
    path = os.path.join(model_dir, model_weight_file_name)
    logger.info(f"the model is saved at {path}")    
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.state_dict(), path)



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
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    

    
    args = parser.parse_args()
    

    ##################################
    #### 훈련 함수 콜
    ##################################
    
    train_sm_ddp(args)

