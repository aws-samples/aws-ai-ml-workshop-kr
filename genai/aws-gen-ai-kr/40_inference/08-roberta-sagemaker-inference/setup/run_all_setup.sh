#!/bin/bash

# KLUE RoBERTa SageMaker Inference - 전체 설정 자동화 스크립트
# 한 번에 모든 환경을 설정합니다.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 KLUE RoBERTa SageMaker 추론 환경 자동 설정"
echo "================================================="
echo "스크립트 위치: $SCRIPT_DIR"
echo "프로젝트 위치: $PROJECT_ROOT"
echo ""

# 사용자 확인
read -p "전체 환경 설정을 시작하시겠습니까? (y/N): " confirm
if [[ $confirm != [yY] ]]; then
    echo "설정이 취소되었습니다."
    exit 0
fi

echo ""
echo "⏱️  예상 소요 시간: 3-5분"
echo ""

# 단계 1: UV 설치
echo "================================================="
echo "1단계: UV 패키지 매니저 설치"
echo "================================================="
$SCRIPT_DIR/00_install_uv.sh

# PATH 업데이트 (현재 세션용)
export PATH="$HOME/.cargo/bin:$PATH"

echo ""
echo "✅ UV 설치 완료"
sleep 2

# 단계 2: 환경 설정
echo ""
echo "================================================="
echo "2단계: Python 가상환경 및 패키지 설치"
echo "================================================="
$SCRIPT_DIR/01_setup_environment.sh

echo ""
echo "✅ 환경 설정 완료"
sleep 2

# 단계 3: 환경 테스트
echo ""
echo "================================================="
echo "3단계: 환경 테스트"
echo "================================================="
$SCRIPT_DIR/02_test_environment.sh

echo ""
echo "================================================="
echo "🎉 전체 설정이 완료되었습니다!"
echo "================================================="
echo ""
echo "📝 다음 단계:"
echo ""
echo "1. 가상환경 활성화:"
echo "   cd $PROJECT_ROOT"
echo "   source .venv/bin/activate"
echo ""
echo "2. Jupyter Lab 실행:"
echo "   jupyter lab"
echo ""
echo "3. 노트북 열기:"
echo "   step3_sagemaker_inference.ipynb"
echo ""
echo "4. 커널 선택:"
echo "   'KLUE RoBERTa Inference (Python 3.11)'"
echo ""
echo "5. 모델 테스트 (선택사항):"
echo "   python test_local_model.py"
echo "   python test_inference.py"
echo ""
echo "================================================="
echo "설정 정보"
echo "================================================="
echo "프로젝트: $PROJECT_ROOT"
echo "가상환경: $PROJECT_ROOT/.venv"
echo "Jupyter 커널: klue-roberta-inference"
echo "Python 버전: 3.11"
echo "PyTorch 버전: 2.5.0 (CUDA 12.4)"
echo "================================================="