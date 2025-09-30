#!/bin/bash

# KLUE RoBERTa SageMaker 추론 환경 설정 - Conda 버전
# SageMaker 노트북 인스턴스에서 확실히 작동하는 설정

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_NAME="klue_roberta"
PYTHON_VERSION="3.11"

echo "=========================================="
echo "KLUE RoBERTa Conda 환경 설정"
echo "=========================================="
echo "프로젝트 경로: $PROJECT_ROOT"
echo "Conda 환경명: $ENV_NAME"
echo "Python 버전: $PYTHON_VERSION"
echo ""

# 사용자 확인
read -p "환경 설정을 시작하시겠습니까? (y/N): " confirm
if [[ $confirm != [yY] ]]; then
    echo "설정이 취소되었습니다."
    exit 0
fi

echo ""
echo "⏱️  예상 소요 시간: 5-10분 (패키지 다운로드 포함)"
echo ""

# 1. 기존 환경 정리
echo "1. 기존 환경 정리 중..."

# 기존 conda 환경 제거
conda env remove -n $ENV_NAME -y 2>/dev/null || true

# 기존 커널 제거
jupyter kernelspec uninstall -y $ENV_NAME 2>/dev/null || true
jupyter kernelspec uninstall -y klue-roberta-inference 2>/dev/null || true

echo "✅ 기존 환경 정리 완료"
echo ""

# 2. Conda 환경 생성
echo "2. Conda 환경 생성 중 (Python $PYTHON_VERSION)..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

echo "✅ Conda 환경 생성 완료"
echo ""

# 3. 환경 활성화 및 pip 업그레이드
echo "3. 환경 활성화 및 기본 설정..."
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME

# pip 업그레이드
pip install --upgrade pip setuptools wheel

echo "✅ 기본 설정 완료"
echo ""

# 4. requirements.txt를 사용한 패키지 설치
echo "4. 패키지 설치 중 (requirements.txt)..."
cd "$SCRIPT_DIR"

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ 패키지 설치 완료"
else
    echo "❌ requirements.txt를 찾을 수 없습니다!"
    exit 1
fi
echo ""

# 5. Jupyter 커널 등록
echo "5. Jupyter 커널 등록 중..."

# 절대 경로로 Python 지정
CONDA_PYTHON="/home/ec2-user/anaconda3/envs/$ENV_NAME/bin/python"

# 커널 등록
$CONDA_PYTHON -m ipykernel install --user --name=$ENV_NAME --display-name="KLUE RoBERTa (Python $PYTHON_VERSION)"

echo "✅ Jupyter 커널 등록 완료"
echo ""

# 6. 설치 확인
echo "6. 설치 확인 중..."
echo "---"
echo "Python 경로: $($CONDA_PYTHON --version)"
echo "Python 위치: $(which python)"
echo ""

$CONDA_PYTHON -c "
import sys
print(f'Python 실행 경로: {sys.executable}')
print('')

try:
    import torch
    print(f'✅ PyTorch {torch.__version__} 설치됨')
    print(f'   CUDA 사용 가능: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   CUDA 버전: {torch.version.cuda}')
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
except ImportError as e:
    print(f'❌ PyTorch 설치 실패: {e}')

try:
    import transformers
    print(f'✅ Transformers {transformers.__version__} 설치됨')
except ImportError as e:
    print(f'❌ Transformers 설치 실패: {e}')

try:
    import sagemaker
    print(f'✅ SageMaker {sagemaker.__version__} 설치됨')
except ImportError as e:
    print(f'❌ SageMaker 설치 실패: {e}')
"
echo "---"
echo ""

# 7. 커널 확인
echo "7. 등록된 커널 확인..."
jupyter kernelspec list | grep -E "$ENV_NAME|python" || true
echo ""

echo "=========================================="
echo "🎉 Conda 환경 설정 완료!"
echo "=========================================="
echo ""
echo "📝 다음 단계:"
echo ""
echo "1. Jupyter Lab/Notebook 재시작:"
echo "   sudo initctl restart jupyter-server"
echo "   또는"
echo "   sudo supervisorctl restart notebook"
echo ""
echo "2. 브라우저 캐시 삭제 및 새로고침:"
echo "   - Ctrl+Shift+R (강력 새로고침)"
echo "   - 또는 시크릿/프라이빗 창에서 열기"
echo ""
echo "3. 노트북에서 커널 선택:"
echo "   Kernel → Change kernel → 'KLUE RoBERTa (Python $PYTHON_VERSION)'"
echo ""
echo "4. 첫 번째 셀에서 확인:"
echo "   !which python"
echo "   !python --version"
echo "   import torch"
echo "   print(torch.__version__)"
echo ""
echo "5. 터미널에서 환경 활성화:"
echo "   conda activate $ENV_NAME"
echo ""
echo "=========================================="
echo "환경 정보:"
echo "---"
echo "Conda 환경명: $ENV_NAME"
echo "Python 버전: $PYTHON_VERSION"
echo "프로젝트 경로: $PROJECT_ROOT"
echo "설정 경로: $SCRIPT_DIR"
echo "=========================================="