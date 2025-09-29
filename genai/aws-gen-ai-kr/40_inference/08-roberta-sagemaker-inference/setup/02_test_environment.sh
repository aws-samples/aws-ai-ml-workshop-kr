#!/bin/bash

# KLUE RoBERTa SageMaker Inference 환경 테스트 스크립트

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_NAME="klue-roberta-inference"

echo "=========================================="
echo "KLUE RoBERTa 환경 테스트"
echo "=========================================="

cd "$SCRIPT_DIR"

# 가상환경 활성화
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ 가상환경 활성화됨"
else
    echo "❌ 가상환경을 찾을 수 없습니다. 01_setup_environment.sh를 먼저 실행하세요."
    exit 1
fi

echo ""
echo "=========================================="
echo "1. Python 환경 확인"
echo "=========================================="

echo "Python 경로: $(which python)"
echo "Python 버전: $(python --version)"

echo ""
echo "=========================================="
echo "2. 핵심 패키지 버전 확인"
echo "=========================================="

python << 'EOF'
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("❌ PyTorch not installed")

try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except ImportError:
    print("❌ Transformers not installed")

try:
    import sagemaker
    print(f"✅ SageMaker: {sagemaker.__version__}")
except ImportError:
    print("❌ SageMaker not installed")

try:
    import boto3
    print(f"✅ Boto3: {boto3.__version__}")
except ImportError:
    print("❌ Boto3 not installed")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError:
    print("❌ NumPy not installed")
EOF


echo ""
echo "=========================================="
echo "3. Jupyter 커널 확인"
echo "=========================================="

echo "등록된 Jupyter 커널:"
jupyter kernelspec list

echo ""
if jupyter kernelspec list | grep -q "$VENV_NAME"; then
    echo "✅ KLUE RoBERTa 커널이 등록되어 있습니다."
else
    echo "⚠️  KLUE RoBERTa 커널이 등록되지 않았습니다."
    echo "다음 명령으로 수동 등록하세요:"
    echo "python -m ipykernel install --user --name=$VENV_NAME --display-name='KLUE RoBERTa Inference (Python 3.10)'"
fi


echo ""
echo "=========================================="
echo "테스트 완료!"
echo "=========================================="
echo ""
echo "✅ 모든 테스트가 통과하면 환경 설정이 완료된 것입니다."
echo ""
echo "📝 다음 단계:"
echo "1. 가상환경 활성화: source .venv/bin/activate"
echo "2. Jupyter Lab 실행: jupyter lab"
echo "3. step3_sagemaker_inference.ipynb 노트북 열기"