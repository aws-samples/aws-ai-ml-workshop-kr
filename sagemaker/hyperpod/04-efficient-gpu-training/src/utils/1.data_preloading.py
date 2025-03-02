#!/usr/bin/env python
# coding=utf-8

import os
import time
import yaml
import argparse
import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed
import pandas as pd
import matplotlib.pyplot as plt
import torch.distributed as dist

from utils.train_utils import print_gpu_utilization, is_main_process, main_process_print
from custom_dataset import StorageConfig, SequenceDataset, StorageAwareDataLoader, PrefetchDataLoader
from model import get_model
from transformers import AutoModelForSequenceClassification


def benchmark_dataloader(dataset, batch_size, num_workers, pin_memory, prefetch_factor, num_epochs, 
                         use_fp16=False, use_dataloader_prefetch_cuda_steam=False, gpu_iterations=3):
    """
    특정 DataLoader 설정에 대한 성능을 벤치마킹합니다.
    
    Parameters:
    -----------
    dataset : Dataset
        벤치마킹에 사용할 데이터셋
    batch_size : int
        배치 크기
    num_workers : int
        DataLoader의 worker 수
    pin_memory : bool
        pin_memory 옵션 사용 여부
    prefetch_factor : int
        prefetch_factor 값 (num_workers > 0일 때만 유효)
    num_epochs : int
        학습할 에폭 수
    use_fp16 : bool
        FP16 혼합 정밀도 사용 여부
        
    Returns:
    --------
    dict
        벤치마크 결과 (처리량, 메모리 사용량, 소요 시간 등)
    """
    # DataLoader 설정
    # prefetch_factor는 num_workers가 0보다 클 때만 사용
    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': True,
        'prefetch_factor': prefetch_factor
    }
    
    # DataLoader 구성
    dataloader = StorageAwareDataLoader(dataset, **dataloader_kwargs)
    
    # Accelerator 초기화
    accelerator = Accelerator(mixed_precision='fp16' if use_fp16 else 'no')
    
    # 모델 초기화
    #model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased", hidden_dropout_prob=0.2)
    model = get_model()
    
    # 옵티마이저 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # 옵티마이저 준비
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Accelerator prepare 이후에 prefetch 래퍼 적용 - CUDA Streaming (CPU -> GPU prefetch, prefetch_factor와는 개념이 다름)
    if use_dataloader_prefetch_cuda_steam:
        dataloader = PrefetchDataLoader(dataloader, accelerator.device)

    # 초기 GPU 메모리 측정
    initial_memory = print_gpu_utilization()
    
    # 학습 루프 시간 측정
    model.train()
    start_time = time.time()
    processed_samples = 0
    total_batches = min(len(dataloader), 20)  # 20 배치만 테스트
    
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            if step >= total_batches:
                break

            # GPU 연산량 증가를 위한 반복
            accumulated_loss = 0
            for i in range(gpu_iterations):
                # 순전파
                outputs = model(**batch)
                current_loss = outputs.loss.detach()  # 그래디언트 기록 중단
                accumulated_loss += current_loss
            
                # 메모리 정리
                del outputs
                torch.cuda.empty_cache()  # GPU 캐시 비우기   

            # 순전파
            outputs = model(**batch)
            loss = outputs.loss
            
            # 역전파
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            processed_samples += batch_size * accelerator.num_processes
    
    # 종료 시간 및 메모리 측정
    end_time = time.time()
    final_memory = print_gpu_utilization()
    
    # 결과 계산
    elapsed_time = end_time - start_time
    throughput = processed_samples / elapsed_time if elapsed_time > 0 else 0
    
    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "throughput": throughput,
        "initial_memory_mb": initial_memory,
        "final_memory_mb": final_memory,
        "memory_increase_mb": final_memory - initial_memory,
        "elapsed_time": elapsed_time
    }


def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description="DataLoader Performance Benchmark")
    parser.add_argument("--config", type=str, required=True, help="설정 파일 경로")
    parser.add_argument("--output", type=str, default="dataloader_benchmark_results.csv",
                      help="결과를 저장할 CSV 파일 경로")
    parser.add_argument("--local_rank", type=int, default=-1,
                      help="분산 훈련을 위한 로컬 랭크 (일반적으로 자동으로 설정됨)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 설정 파일 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if is_main_process():
        main_process_print(config)
    
    # 재현성을 위한 랜덤 시드 설정
    set_seed(42)
    
    # 전통적인 PyTorch Dataset 클래스를 사용한 데이터셋 생성
    storage_config = StorageConfig(
        storage_type="luster",
        base_path="datasets"
    )

    # C/GPU 연산량 조절
    cpu_iterations = config['cpu_iterations'] # CPU 연산량 조절용 파라미터
    gpu_iterations = config['gpu_iterations'] # GPU 연산량 조절용 파라미터

    dataset = SequenceDataset(
        storage_config=storage_config,
        max_seq_length=config['max_seq_length'],
        cpu_iterations=cpu_iterations,
        use_cache=config.get('dataloader_cache', False) # dataloader cache 옵션 
    )
    
    # 테스트할 worker 수 범위
    worker_counts = [0, 2, 4, 8] if torch.cuda.is_available() else [0, 1, 2]
    
    # pin_memory 옵션
    pin_memory_options = [False, True]
    
    # 배치 크기
    batch_size = config['per_device_train_batch_size']

    # num_epochs 설정 (config에서 가져오거나 기본값 사용)
    num_epochs = config.get('num_epochs', 1)

    # dataloader_prefetch_cuda_steam
    dataloader_prefetch_cuda_steam = config.get('dataloader_prefetch_cuda_steam', False)
    
    # 결과 저장할 리스트
    results = []
    
    # 모든 설정 조합에 대해 벤치마크 실행
    if is_main_process():
        main_process_print(f"{'=' * 70}")
        main_process_print(f"DataLoader 성능 벤치마크 시작 (num_epochs: {num_epochs}, batch_size: {batch_size})")
        main_process_print(f"테스트 파라미터: workers, pin_memory")
        main_process_print(f"{'=' * 70}")
    
    for num_workers in worker_counts:
        for pin_memory in pin_memory_options:

            if is_main_process():
                main_process_print("테스트 설정: ")
                main_process_print(f'  num_workers={num_workers}')
                main_process_print(f'  pin_memory={pin_memory}')
                
            # 벤치마크 실행
            result = benchmark_dataloader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=(2 if num_workers > 0 else None), # defalut 2, 그아래로 설정 안 됨, workers > 0 에 None 으로 설정해도 2로 셋팅 
                num_epochs=num_epochs,
                use_fp16=config['fp16'],
                use_dataloader_prefetch_cuda_steam=dataloader_prefetch_cuda_steam,
                gpu_iterations=gpu_iterations
            )
            
            # 결과 출력 - 메인 프로세스에서만
            if is_main_process():
                main_process_print(f"  처리량: {result['throughput']:.2f} samples/sec")
                main_process_print(f"  소요 시간: {result['elapsed_time']:.2f} seconds")
                main_process_print(f"  메모리 사용량: {result['final_memory_mb']} MB")
            
            # 결과 저장 - 메인 프로세스에서만
            if is_main_process():
                results.append(result)
            
    
if __name__ == "__main__":
    # 분산 환경 초기화를 보장
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
    # if "SM_TRAINING_ENV" in os.environ:
    #     import smdistributed.dataparallel.torch.torch_smddp
    #     dist.init_process_group(backend="smddp")
    # else:
    #     dist.init_process_group(backend="nccl")
    main()