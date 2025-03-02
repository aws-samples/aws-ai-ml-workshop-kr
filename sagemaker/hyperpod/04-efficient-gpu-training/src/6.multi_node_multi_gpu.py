#!/usr/bin/env python
# coding=utf-8

import os
import time
import yaml
import torch
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.multiprocessing as mp

from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from model import get_model
from utils.train_utils import print_gpu_utilization, is_main_process, main_process_print
from custom_dataset import StorageConfig, SequenceDataset, StorageAwareDataLoader, PrefetchDataLoader

def benchmark_put_it_all_together(
    dataset,
    batch_size,
    num_workers,
    pin_memory,
    prefetch_factor,
    num_epochs,
    gradient_checkpointing,
    gradient_accumulation_steps,
    mixed_precision,
    use_dataloader_prefetch_cuda_steam=False,
    gpu_iterations=3
    ):
    """
    특정 DataLoader 설정에 대한 성능을 벤치마킹합니다.
    
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
    dataloader = StorageAwareDataLoader(
        dataset=dataset,
        distributed=(True if os.environ["DISTRIBUTED"] == "True" else False),
        **dataloader_kwargs
    )
    
    # Accelerator 초기화
    accelerator = Accelerator(mixed_precision=mixed_precision)

    # 모델 초기화
    #model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased", hidden_dropout_prob=0.2)
    model = get_model()

    #model = torch.compile(model)

    # gradient_checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
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
        for step, batch in enumerate(dataloader, start=1):
            if step >= total_batches:
                break

            # GPU 연산량 증가를 위한 반복 (연산 자체는 아무 의미 없음, GPU 연산량을 늘리기 위한 용도)
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
            
            # gradient_accumulation
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            # 역전파
            accelerator.backward(loss)
            
            # gradient_accumulation
            if gradient_accumulation_steps > 1:
                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
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
        main_process_print(f'config" {config}')
    
    # 재현성을 위한 랜덤 시드 설정
    set_seed(42)
    
    # 테스트할 worker 수 범위
    num_workers = config.get('num_workers', 1)
    
    # pin_memory 옵션
    pin_memory = config.get('pin_memory', True)

    # data_loader prefetch_factor 
    prefetch_factor = config.get('prefetch_factor', 2)
    
    # 배치 크기
    batch_size = config['per_device_train_batch_size']

    # num_epochs 설정 (config에서 가져오거나 기본값 사용)
    num_epochs = config.get('num_epochs', 1)

    # dataloader_prefetch_cuda_steam
    dataloader_prefetch_cuda_steam = config.get('dataloader_prefetch_cuda_steam', False)
    
    # Graident trick options
    gradient_checkpointing = config.get('gradient_checkpointing', False)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1) # > 1 then enalble, ==1 then disable

    # Mixed precision
    mixed_precision = config.get("mixed_precision", "no")

    # Tensor Float 32
    tf32 = config.get("tf32", False)
    torch.backends.cuda.matmul.allow_tf32 = True if tf32 else False
    torch.backends.cudnn.allow_tf32 = True if tf32 else False

    # Storage
    storage_type = config.get("storage_type", "lustre")

    # 결과 저장할 리스트
    results = []

    # 전통적인 PyTorch Dataset 클래스를 사용한 데이터셋 생성
    storage_config = StorageConfig(
        storage_type="lustre",  # "local", "s3", "lustre"
        base_path="datasets",
        #s3_bucket="efficient-gpu-training"
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

    # default setting
    # num_workers = 0
    # pin_memory = False
    # prefetch_factor = None
    # dataloader_prefetch_cuda_steam = False
    # gradient_checkpointing = False
    # gradient_accumulation_steps = 1
    # mixed_precision = "no"
    # tf32 = False
    # storage_type = "s3"
    # torch.backends.cuda.matmul.allow_tf32 = True if tf32 else False
    # torch.backends.cudnn.allow_tf32 = True if tf32 else False

    # 모든 설정 조합에 대해 벤치마크 실행
    if is_main_process():
        main_process_print(f"{'=' * 70}")
        main_process_print(f"DataLoader 성능 벤치마크 시작 (num_epochs: {num_epochs}, batch_size: {batch_size})")
        main_process_print(f"테스트 파라미터:")
        main_process_print(f'  num_workers={num_workers}')
        main_process_print(f'  pin_memory={pin_memory}')
        main_process_print(f'  prefetch_factor={prefetch_factor}')
        main_process_print(f'  dataloader_prefetch_cuda_steam={dataloader_prefetch_cuda_steam}')
        main_process_print(f'  gradient_checkpointing={gradient_checkpointing}')
        main_process_print(f'  gradient_accumulation_steps={gradient_accumulation_steps}')
        main_process_print(f'  mixed_precision={mixed_precision}')
        main_process_print(f'  tf32={tf32}')
        main_process_print(f'  storage_type={storage_type}')
        main_process_print(f"{'=' * 70}")
        
    # 벤치마크 실행
    result = benchmark_put_it_all_together(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=(2 if num_workers > 0 else None), # defalut 2, 그아래로 설정 안 됨, workers > 0 에 None 으로 설정해도 2로 셋팅 
        num_epochs=num_epochs,
        mixed_precision=mixed_precision,
        use_dataloader_prefetch_cuda_steam=dataloader_prefetch_cuda_steam,
        gpu_iterations=gpu_iterations,
        gradient_checkpointing=gradient_checkpointing,
        gradient_accumulation_steps=gradient_accumulation_steps
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
    
    mp.set_start_method('spawn')  # 프로그램 시작 시 설정

    # 분산 환경 초기화를 보장
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        os.environ["DISTRIBUTED"] = "True" # str
    else: 
        os.environ["DISTRIBUTED"] = "False" # str

    
    # if "SM_TRAINING_ENV" in os.environ:
    # if args.distributed_backend == "smddp":
    #    import smdistributed.dataparallel.torch.torch_smddp  # pylint: disable=unused-import
    #     import smdistributed.dataparallel.torch.torch_smddp
    #     dist.init_process_group(backend="smddp")
    # else:
    #     dist.init_process_group(backend="nccl")


    #     # This is the only change needed to enable SMDDP in an FSDP script
    # try:
    #     backend = "smddp"
    #     import smdistributed.dataparallel.torch.torch_smddp
    # except ModuleNotFoundError:
    #     backend = "nccl"
    # print(f"using backend: {backend}")

    # Install SMDDP wheel (only run for cuda11.8)
# SMDDP_WHL="smdistributed_dataparallel-2.0.2-cp310-cp310-linux_x86_64.whl" \
#   && wget -q https://smdataparallel.s3.amazonaws.com/binary/pytorch/2.0.1/cu118/2023-12-07/${SMDDP_WHL} \
#   && pip install --force ${SMDDP_WHL} \
#   && rm ${SMDDP_WHL}

    print ('os.environ["DISTRIBUTED"]', os.environ["DISTRIBUTED"])
    main()