#!/bin/bash

# KLUE RoBERTa SageMaker Inference 환경 설정 스크립트
# uv를 사용한 Python 가상환경 생성 및 패키지 설치

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_NAME="klue-roberta-inference"
PYTHON_VERSION="3.11"

echo "=========================================="
echo "KLUE RoBERTa SageMaker 추론 환경 설정"
echo "=========================================="
echo "프로젝트 경로: $SCRIPT_DIR"
echo "가상환경 이름: $VENV_NAME"
echo "Python 버전: $PYTHON_VERSION"
echo ""

# setup 디렉토리로 이동
cd "$SCRIPT_DIR"

# 기존 가상환경 제거 (있다면)
if [ -d ".venv" ]; then
    echo "🗑️  기존 가상환경을 제거합니다..."
    rm -rf .venv
fi

# uv가 설치되어 있는지 확인
if ! command -v uv &> /dev/null; then
    echo "❌ uv가 설치되지 않았습니다. 먼저 00_install_uv.sh를 실행하세요."
    exit 1
fi

echo "=========================================="
echo "1. Python 가상환경 생성 중..."
echo "=========================================="

# Python 가상환경 생성 (uv 사용)
uv venv --python $PYTHON_VERSION

echo "✅ Python $PYTHON_VERSION 가상환경이 생성되었습니다."

echo "=========================================="
echo "2. pyproject.toml을 사용한 패키지 설치 중..."
echo "=========================================="

# 가상환경 활성화
source .venv/bin/activate

echo "📦 프로젝트 의존성 설치 중..."
# pyproject.toml을 사용하여 설치 (PyTorch CUDA 포함)
uv pip install -e . --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu121

echo "✅ 패키지 설치가 완료되었습니다."

echo "=========================================="
echo "3. Jupyter 커널 등록 중..."
echo "=========================================="

# Jupyter 커널 등록
python -m ipykernel install --user --name=$VENV_NAME --display-name="KLUE RoBERTa Inference (Python 3.10)"

echo "✅ Jupyter 커널이 등록되었습니다."

echo "=========================================="
echo "4. 환경 정보 출력 중..."
echo "=========================================="

echo "Python 버전:"
python --version

echo ""
echo "PyTorch 버전 및 CUDA 지원:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Transformers 버전:"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo ""
echo "SageMaker 버전:"
python -c "import sagemaker; print(f'SageMaker: {sagemaker.__version__}')"

echo ""
echo "설치된 패키지 목록:"
uv pip list

echo ""
echo "=========================================="
echo "🎉 환경 설정이 완료되었습니다!"
echo "=========================================="
echo ""
echo "📝 사용 방법:"
echo "1. 가상환경 활성화:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Jupyter Lab 실행:"
echo "   jupyter lab"
echo ""
echo "3. Jupyter에서 커널 선택:"
echo "   'KLUE RoBERTa Inference (Python 3.10)' 커널을 선택하세요."
echo ""
echo "4. 테스트 실행:"
echo "   python test_local_model.py"
echo ""
echo "5. 환경 비활성화:"
echo "   deactivate"
echo ""
echo "📦 가상환경 위치: $SCRIPT_DIR/.venv"
echo "📋 Jupyter 커널명: $VENV_NAME"