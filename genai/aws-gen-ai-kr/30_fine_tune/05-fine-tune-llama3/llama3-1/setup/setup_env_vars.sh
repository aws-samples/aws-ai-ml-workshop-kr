#!/bin/bash

# 환경 변수 설정
CACHE_PATH=/home/ec2-user/SageMaker/.cache
HF_DATASETS_CACHE=/home/ec2-user/SageMaker/.cache
HF_HOME=/home/ec2-user/SageMaker/.cache

# .bashrc 파일 경로
BASHRC_FILE="$HOME/.bashrc"

# 환경 변수 설정 문장
EXPORT_STATEMENT_CACHE_PATH="export TRANSFORMERS_CACHE=$CACHE_PATH"
EXPORT_STATEMENT_HF_DATASETS_CACHE="export HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
EXPORT_STATEMENT_HF_HOME="export HF_HOME=$HF_HOME"

# .bashrc 파일에 해당 문장이 있는지 확인
if grep -q "$EXPORT_STATEMENT_CACHE_PATH" "$BASHRC_FILE"; then
    echo "TRANSFORMERS_CACHE is already set in .bashrc"
else
    # .bashrc 파일에 환경 변수 설정 추가
    echo "$EXPORT_STATEMENT_CACHE_PATH" is $EXPORT_STATEMENT_CACHE_PATH
    echo $EXPORT_STATEMENT_CACHE_PATH >> "$BASHRC_FILE"
    echo $EXPORT_STATEMENT_HF_DATASETS_CACHE >> "$BASHRC_FILE"    
    echo $EXPORT_STATEMENT_HF_HOME >> "$BASHRC_FILE"        
    echo "Added TRANSFORMERS_CACHE, HF_DATASETS_CACHE, HF_HOME to .bashrc"
fi

# 변경사항 적용
source "$BASHRC_FILE"

echo "TRANSFORMERS_CACHE is now set to: $TRANSFORMERS_CACHE"
echo "HF_DATASETS_CACHE is now set to: $HF_DATASETS_CACHE"
echo "HF_HOME is now set to: $HF_HOME"