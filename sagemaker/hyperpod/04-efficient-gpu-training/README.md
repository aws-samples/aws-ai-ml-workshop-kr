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

## 1. 훈련 Git Repo 준비
- Cluster head node 에서 훈련 코드가 있는 Git Repo 및 작업할 Git Repo 다운로드
    - 훈련 코드 리포:  https://github.com/aws-samples/aws-ai-ml-workshop-kr
    - 워크샵 리포: awsome-distributed-training, Cluster 세팅하면 기본 Git Repo 로 설치 됨.
- Workding 디렉토리 이동 (아래는 예시이고, 작업 폴더를 하나 생성하고 시작해도 됩니다.)
    ```
     cd lab/07-llama3-korean-news-summary-lora-hyperpod/
    ubuntu@ip-10-1-92-35:~/lab/07-llama3-korean-news-summary-lora-hyperpod$ pwd
    /fsx/ubuntu/lab/07-llama3-korean-news-summary-lora-hyperpod
    ```
## 2. 환경 생성 및 sbatch 파일 준비
- 워크샵 리포로 부터 필요한 셀 파일을 작업폴더에 카피
    * 소스: /awsome-distributed-training/3.test_cases/10.FSD
        * 0.create_conda_env.sh 
        * 1.distributed-training-llama2-5-steps.sbatch

## 3. 콘다 가상 환경 설치 및 실행
- requirements.txt 파일을 추가하고, 필요한 python packaage 를 추가함.
    - torch 를 설치시에 cuda 버전과 호환되는 지 확인 하세요.
    - Compute Node 의 cuda 버전 확인 : (Torch 호환성 확인)
        ```
        Head Node 에서 ssh compute node hostname (i.e. ssh ip-10-1-92-127)
        ```
- 0.create_conda_env.sh 수정
    * 가상 환경 이름 수정 (예: llama3_lora )
    * pip install -r requirements.txt 내용 추가
* ./0.create_conda_env.sh 실행
    * 버전 확인
        * source ./miniconda3/bin/activate
        * source activate ./llama_lora/
        * pip list | grep torch

## 4. 데이타 준비
- data_preparation 폴더 생성
    * data_preparation 폴더에 prepare_data.py 생성 및 [기존의 데이터 준비 로직](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/genai/aws-gen-ai-kr/30_fine_tune/05-fine-tune-llama3/llama3/archive/fine-tune-llama3/notebook/01-naver-news-fsdp-QLoRA/01-Prepare-Dataset-KR-News.ipynb)을 카피해서 정리
    * prepare_data.py  동작 테스트
        * 아래 처럼 가상 환경에 진입한 후에 테스트
        ```
            source ./miniconda3/bin/activate
            source activate ./llama_lora/
            python prepare_data.py
        ```
## 5. 훈련 스크립트 준비
- train_script 폴더 생성 및 스크립트 카피
    * 파일 카피: train_script
        * [local_run_fsdp_qlora.py](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/genai/aws-gen-ai-kr/30_fine_tune/05-fine-tune-llama3/llama3/archive/fine-tune-llama3/scripts/local_run_fsdp_qlora.py)
        * 신규 train_qLora.py 로 복사
    * 파일 카피: [local_llama_3_8b_fsdp_qlora.yaml](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/genai/aws-gen-ai-kr/30_fine_tune/05-fine-tune-llama3/llama3/archive/fine-tune-llama3/notebook/01-naver-news-fsdp-QLoRA/accelerator_config/local_llama_3_8b_fsdp_qlora.yaml)
    
- train_script 수정  
    - local_llama_3_8b_fsdp_qlora.yaml 설정 파일 수정
        * 데이터 경로: 위의 prepare_data.py 에서 생성된 경로로 수정
    * train.py 스크립트 필요시 수정 

## 6. Slurm Batch (sbatch) 파일 준비                 
- sbatch 스크립트 수정 (예: [1.distributed-training-llama3-qLora.sbatch](https://github.com/gonsoomoon-ml/lab/blob/main/07-llama3-korean-news-summary-lora-hyperpod/1.distributed-training-llama3-qLora.sbatch) )
    - 아래는 torchrun , train_lora.py, hyperpod_llama_3_8b_fsdp_qlora.yaml 경로 수정
    ```
    export TORCHRUN=llama_lora/bin/torchrun
    export TRAIN_SCRIPT=train_script/train_lora.py
    export CONFIG_PATH=train_script/hyperpod_llama_3_8b_fsdp_qlora.yaml
    ```

## 7. 훈련 실행 및 결과
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

        
## 8. 트러벌 슈팅
- 1.distributed-training-llama3-qLora.sbatch  파일에 몇 개의 노드로 훈련을 할지의 파라미터 입니다. 현재 1 만이 작동하고, 2 이상일 경우에는 에러가 발생 합니다. (원인 파악 중 입니다.)
```
#SBATCH --nodes=1 # number of nodes to use
```