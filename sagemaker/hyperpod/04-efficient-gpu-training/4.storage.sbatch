#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --nodes=1 # number of nodes to use
#SBATCH --job-name=4.STORAGE # name of your job
#SBATCH --output=logs/%x_%j.out # logfile for stdout
#SBATCH --error=logs/%x_%j.err # logfile for stderr, remove it to merge both outputs
#SBATCH --exclusive # job has exclusive use of the resource, no sharing

set -ex; # 스크립트 실행 중 에러가 발생하면 즉시 종료, 실행되는 모든 명령어를 화면에 출력

###########################
###### User Variables #####
###########################

GPUS_PER_NODE=1 # 4 for G5.12x, 8 for P4/P5 (사용할 GPU 수, 노드당)

###########################
## 환경 변수 설정 ##
###########################

# 분산 환경 로그 제어
export NCCL_DEBUG=WARN
export TORCH_CPP_LOG_LEVEL=ERROR

## TORCHRUN path and 
export TORCHRUN=./efficient_gpu_training/bin/torchrun
export TRAIN_SCRIPT=./src/4.storage.py
export CONFIG_PATH=./src/4.config_storage.yaml

###########################
####### Torch Dist  #######
###########################

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=$GPUS_PER_NODE
    --nnodes=$SLURM_JOB_NUM_NODES
    --rdzv_id=$SLURM_JOB_ID
    --rdzv_backend=c10d
    --rdzv_endpoint=$(hostname)
)

############################
# Training Configuration  ##
############################

declare -a TRAINING_ARGS=(
    "--config"
    "${CONFIG_PATH}"
)

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi

echo "========================================================"
echo "데이터로더 최적화 실험 시작"
echo "========================================================"

# 기본 설정으로 실행
srun ${AUTO_RESUME} -l ${TORCHRUN} "${TORCHRUN_ARGS[@]}" $TRAIN_SCRIPT "${TRAINING_ARGS[@]}"

echo "========================================================"
echo "데이터로더 최적화 실험 완료"
echo "========================================================"