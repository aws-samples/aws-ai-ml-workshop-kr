# Methods and tools for efficient training on GPU

- 이 Lab은 `효율적인(연산시간 및 메모리 감소) GPU 학습`을 위한 Method 및 Tool에 대해 다루고 있으며 `SageMaker HyperPod` 기반 실습 코드를 제공합니다.
--- 

## 1. Contents
- **Data pre-loading**
    - num_workers
    - pin_memory
    - prefetch
        - pre-computation (prefetch_factor)
        - cpu -> gpu transfer (by cuda stream)
    - caching (dataloader)
- **Gradient tricks**
    - Gradient checkpointing
    - Gradient accumulation
- **Mixed precision** and **Tensor Float 32**
    - FP16, BF16 and TF32
- **Storage**
    - FSx Lustre vs S3 File System
- **Put it all together**
- **Multi Node Multi GPUs**

## 0. 선수 항목
- [Amazon SageMaker HyperPod](https://catalog.workshops.aws/sagemaker-hyperpod/en-US/01-cluster/option-a-easy-cluster-setup/00-easy-cluster-setup) 워크샵의 아래 항목을 먼저 진행 해야 함.
    - 0. Prerequisites
    - 1. Cluster Setup - Computing node 로서 ml.g5.12xlarge 4대
    - 2. [VS Code Integration](https://catalog.workshops.aws/sagemaker-hyperpod/en-US/05-advanced/05-vs-code)

## 1. 훈련 Git Repo 준비 (Head Node)
- Cluster head node 접근 (using vscode or ssh)
- Cluster head node 에서 훈련 코드가 있는 Git Repo 및 작업할 Git Repo 다운로드
  
    ```
    git clone https://github.com/aws-samples/aws-ai-ml-workshop-kr.git
    cd aws-ai-ml-workshop-kr/sagemaker/hyperpod/04-efficient-gpu-training/
    ```
## 2. 콘다 가상 환경 설치 및 실행 (Head Node)
- 0.create_conda_env.sh 수행 (efficient_gpu_training 가상 작업환경 생성)
  
    ```
    bash 0.create_conda_env.sh
    ```

## 3. 데이터셋 준비
- Synthetic dataset 생성
    - src/generate_sample_dataset.py 수행 (dataset dir 내 1,000개 파일 생성)
      
        ```
        ./efficient_gpu_training/bin/python ./src/generate_sample_dataset.py
        
        ``` 
- S3 data upload (S3 File System 테스트 용)
    - Step 1. S3 bucket 생성: efficient-gpu-training
    - (local vs code 활용 시) Step 2. aws configure 셋팅 
    - Step 3. upload 코드 수행
      
        ```
        bash ./src/upload_data_to_s3.sh
        ```  


## 4. Job 수행 (Slum 기반 스케쥴링)
- 1. Data pre-loading
    - sbatch | script | yaml [1.data_preloading.sbatch]
- [2.gradient_trick.sbatch]
- [3.mixed_precision.sbatch]
- [4.storage.sbatch]
- [5.put_it_all_together.sbatch]
- [6.multi_node_multi_gpu.sbatch]


## 5. 실행 및 결과
- 아래와 같이 명령어로 실행
    ```
    ubuntu@ip-10-1-92-35:~/lab/07-llama3-korean-news-summary-lora-hyperpod$ pwd
    /fsx/ubuntu/lab/07-llama3-korean-news-summary-lora-hyperpod
    ubuntu@ip-10-1-92-35:~/lab/07-llama3-korean-news-summary-lora-hyperpod$ sbatch 1.distributed-training-llama3-qLora.sbatch 
    ```
- 실행 결과
    - 실행 로그
        ```
        ubuntu@ip-10-1-92-35:~/lab/07-llama3-korean-news-summary-lora-hyperpod/logs$ ls -al
        -rw-rw-r--  1 ubuntu ubuntu 11976 Feb 16 12:05 FSDP_60.err
        -rw-rw-r--  1 ubuntu ubuntu 43633 Feb 16 12:05 FSDP_60.out
        ```
    - checkpoint file
        ```
        ubuntu@ip-10-1-92-35:~/lab/07-llama3-korean-news-summary-lora-hyperpod/outputs$ ls -al
        drwxrwxr-x  2 ubuntu ubuntu    33280 Feb 16 09:03 checkpoint-1
        drwxrwxr-x  2 ubuntu ubuntu    33280 Feb 16 09:03 checkpoint-2
        drwxrwxr-x  2 ubuntu ubuntu    33280 Feb 16 09:04 checkpoint-3
        drwxrwxr-x  2 ubuntu ubuntu    33280 Feb 16 09:05 checkpoint-4
        drwxrwxr-x  2 ubuntu ubuntu    33280 Feb 16 09:05 checkpoint-5
        drwxrwxr-x 30 ubuntu ubuntu    33280 Feb 16 12:02 runs
        ```
- 작업 디렉토리 기준으로 생성된 파일 및 결과
    - 아래는 위이 결과를 실행하고 나온 최종 작업 디렉토리 입니다.
    - ![hyperpod_working_directory.png](img/hyperpod_working_directory.png)        
