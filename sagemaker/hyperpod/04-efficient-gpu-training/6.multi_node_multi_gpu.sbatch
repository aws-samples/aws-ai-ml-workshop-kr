#!/bin/bash

#SBATCH --nodes=4                                    # 4개 노드로 변경 (원하는 노드 수로 조정 가능)
#SBATCH --job-name=6.MULTI-NODE-MULTI-GPU
#SBATCH --output=logs/%x_%j.out 
#SBATCH --error=logs/%x_%j.err  
#SBATCH --exclusive             

set -ex;

###########################
###### User Variables #####
###########################

GPUS_PER_NODE=4                                      # 노드당 GPU 수 (G5.12x 기준)
TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))  # 전체 GPU 수 계산

###########################
## 환경 변수 설정 ##
###########################

# 분산 학습 환경 최적화를 위한 환경변수 추가
export NCCL_DEBUG=WARN
export TORCH_CPP_LOG_LEVEL=ERROR
export NCCL_SOCKET_IFNAME=ens  # 네트워크 인터페이스 지정
export NCCL_IB_DISABLE=0       # InfiniBand 활성화
export NCCL_P2P_DISABLE=0      # P2P 통신 활성화

## TORCHRUN path and scripts
export TORCHRUN=./efficient_gpu_training/bin/torchrun
export TRAIN_SCRIPT=./src/6.multi_node_multi_gpu.py
export CONFIG_PATH=./src/6.config_multi_node_multi_gpu.yaml

###########################
####### Torch Dist  #######
###########################

# 첫 번째 노드의 IP 주소를 master로 사용
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=$GPUS_PER_NODE
    --nnodes=$SLURM_JOB_NUM_NODES
    --node_rank=$SLURM_NODEID
    --master_addr=$MASTER_ADDR
    --master_port=$MASTER_PORT
    --rdzv_id=$SLURM_JOB_ID
    --rdzv_backend=c10d
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT
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
echo "멀티 노드 분산 학습 시작"
echo "노드 수: $SLURM_JOB_NUM_NODES"
echo "노드당 GPU 수: $GPUS_PER_NODE"
echo "총 GPU 수: $TOTAL_GPUS"
echo "마스터 노드: $MASTER_ADDR"
echo "========================================================"

# srun으로 실행
srun ${AUTO_RESUME} -l ${TORCHRUN} "${TORCHRUN_ARGS[@]}" $TRAIN_SCRIPT "${TRAINING_ARGS[@]}"

echo "========================================================"
echo "멀티 노드 분산 학습 완료"
echo "========================================================"