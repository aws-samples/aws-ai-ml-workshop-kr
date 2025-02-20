#!/usr/bin/env bash
set -ex

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -f -p ./miniconda3

source ./miniconda3/bin/activate

conda create -y -p ./llama_lora python=3.10.14

source activate ./llama_lora/

pip install -r requirements.txt
# conda install -y pytorch=2.4.1 torchvision torchaudio transformers datasets fsspec=2023.9.2 pytorch-cuda=12.1 "numpy=1.*" -c pytorch -c nvidia

# Create checkpoint dir
mkdir checkpoints
