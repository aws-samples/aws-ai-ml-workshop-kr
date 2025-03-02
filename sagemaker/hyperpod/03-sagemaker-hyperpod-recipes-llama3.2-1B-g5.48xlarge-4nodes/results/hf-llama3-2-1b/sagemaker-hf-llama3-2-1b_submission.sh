#!/bin/bash

# Parameters
#SBATCH --exclusive
#SBATCH --job-name=sagemaker-hf-llama3-2-1b
#SBATCH --mem=0
#SBATCH --nodes=4
#SBATCH --output=/fsx/ubuntu/sagemaker-hyperpod-recipes/results/hf-llama3-2-1b/log-sagemaker-hf-llama3-2-1b_%j.out
#SBATCH --time=6-00:00:00

# setup
export NCCL_DEBUG=WARN
export FI_PROVIDER=efa
export NCCL_SOCKET_IFNAME=^lo,docker0,veth_def_agent
export NCCL_IGNORE_DISABLED_P2P=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DIST_INIT_BARRIER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1


# Prepare distributed files
srun -l bash -c "scontrol show hostnames | sort > /fsx/ubuntu/sagemaker-hyperpod-recipes/results/hf-llama3-2-1b/hostname"

# command 1
srun --output /fsx/ubuntu/sagemaker-hyperpod-recipes/results/hf-llama3-2-1b/log-sagemaker-hf-llama3-2-1b_%j.out --container-image /fsx/ubuntu/sagemaker-hyperpod-recipes/smdistributed-modelparallel.sqsh --container-mounts /fsx/ubuntu/sagemaker-hyperpod-recipes/launcher/nemo/nemo_framework_launcher/launcher_scripts:/fsx/ubuntu/sagemaker-hyperpod-recipes/launcher/nemo/nemo_framework_launcher/launcher_scripts,/fsx/ubuntu/sagemaker-hyperpod-recipes/results:/fsx/ubuntu/sagemaker-hyperpod-recipes/results,/var/log/aws/clusters:/var/log/aws/clusters bash -c "
  bash /fsx/ubuntu/sagemaker-hyperpod-recipes/results/hf-llama3-2-1b/train_script.sh "
