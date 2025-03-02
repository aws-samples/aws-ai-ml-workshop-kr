#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --nodes=1 # number of nodes to use
#SBATCH --job-name=FSDP # name of your job
#SBATCH --output=logs/%x_%j.out # logfile for stdout
#SBATCH --error=logs/%x_%j.err # logfile for stderr, remove it to merge both outputs
#SBATCH --exclusive # job has exclusive use of the resource, no sharing

set -ex; # 스크립트 실행 중 에러가 발생하면 즉시 종료, 실행되는 모든 명령어를 화면에 출력

###########################
###### User Variables #####
###########################

GPUS_PER_NODE=8 # 4 for G5.12x, 8 for P4/P5

###########################
## FSDP Environment Variables ##
###########################
# export TORCH_DISTRIBUTED_DEBUG=INFO
export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1
export FSDP_SHARDING_STRATEGY=1 

###########################
## EFA Environment Variables ##
###########################

## Plenty of EFA level variables for p4, p5
## For G4dn and other G5, comment out all
#export FI_LOG_LEVEL=warn
#export NCCL_DEBUG=INFO
#export FI_PROVIDER=efa
#export FI_EFA_USE_HUGE_PAGE=0    # Set to 0 when you see os.fork() causes OSError: Cannot allocate memory.  Disabling huge page causes minor performance hit.
## Switching SYNC_MEMOPS to zero can boost throughput with FSDP
## Disables CU_POINTER_ATTRIBUTE_SYNC_MEMOPS
## Reduces memory synchronizations
## https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html
#export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
# LD_PRELOAD is required for PyTorch to find the NCCL library
# This path assumes you are using the Deep Learning AMI
# If you are not using the DLAMI, you may need to update this path
#export LD_PRELOAD=/usr/local/cuda-12.1/lib/libnccl.so
#export NCCL_SOCKET_IFNAME=^docker,lo,veth_def_agent,eth


## TORCHRUN path and 
export TORCHRUN=llama_lora/bin/torchrun
export TRAIN_SCRIPT=train_script/train_lora.py
export CONFIG_PATH=train_script/hyperpod_llama_3_8b_fsdp_qlora.yaml

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
srun ${AUTO_RESUME} -l ${TORCHRUN} "${TORCHRUN_ARGS[@]}" $TRAIN_SCRIPT "${TRAINING_ARGS[@]}"
