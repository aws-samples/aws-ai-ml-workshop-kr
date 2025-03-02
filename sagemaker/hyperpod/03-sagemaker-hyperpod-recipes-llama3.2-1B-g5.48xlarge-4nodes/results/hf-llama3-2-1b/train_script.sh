#!/bin/bash
set -ex
export NCCL_DEBUG=WARN
export FI_PROVIDER=efa
export NCCL_SOCKET_IFNAME=^lo,docker0,veth_def_agent
export NCCL_IGNORE_DISABLED_P2P=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DIST_INIT_BARRIER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
MASTER_ADDR=$(head -n 1 /fsx/ubuntu/sagemaker-hyperpod-recipes/results/hf-llama3-2-1b/hostname)
NODEID=$(($(grep -nx -o "\b$(hostname)\b" /fsx/ubuntu/sagemaker-hyperpod-recipes/results/hf-llama3-2-1b/hostname | cut -d ":" -f 1) - 1))
NNODES=4
PROCESSES_PER_NODE=8
MASTER_PORT=41000

DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE --nnodes $NNODES --rdzv_endpoint=$MASTER_ADDR --rdzv_id=100 --rdzv_backend=c10d"

# For greater env stability, grab hostname from `hostname`
# https://sim.amazon.com/issues/P162624109
LAUNCHER_HOSTNAME="$(hostname)"

mkdir -p $HOME/tmp
GIT_CLONE_DIR="$HOME/tmp/$LAUNCHER_HOSTNAME"
[[ -d $GIT_CLONE_DIR ]] && rm -rf $GIT_CLONE_DIR
git clone https://github.com/aws/sagemaker-hyperpod-training-adapter-for-nemo.git $GIT_CLONE_DIR
GIT_CLONE_DIR=${GIT_CLONE_DIR}/
cd $GIT_CLONE_DIR
rm -rf __pycache__

unset SLURM_NTASKS

torchrun $DISTRIBUTED_ARGS  \
  examples/llama/llama_pretrain.py \
  --config-path=/fsx/ubuntu/sagemaker-hyperpod-recipes/results/hf-llama3-2-1b --config-name=hf-llama3-2-1b_hydra.yaml