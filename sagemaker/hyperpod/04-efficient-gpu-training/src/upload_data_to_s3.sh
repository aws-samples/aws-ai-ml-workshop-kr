#!/usr/bin/env bash
set -ex

# Step 1. S3 bucket 생성: efficient-gpu-training
# Step 2. aws configure 셋팅
# Step 3. 21.efficient-gpu-training/src/ 로 change dir 한 후 아래 수행 
aws s3 cp ./datasets s3://efficient-gpu-training/datasets/ --recursive
