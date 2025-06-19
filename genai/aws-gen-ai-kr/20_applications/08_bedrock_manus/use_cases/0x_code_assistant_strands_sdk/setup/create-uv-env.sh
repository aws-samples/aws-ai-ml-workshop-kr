#!/bin/bash

# UV 환경 설정 및 Jupyter 커널 등록 스크립트
# 사용법: ./setup_uv_jupyter.sh <환경이름> [python버전]

set -e  # 에러 발생 시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 함수: 출력 메시지
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 사용법 출력
usage() {
    echo "사용법: $0 <환경이름> [python버전]"
    echo ""
    echo "예시:"
    echo "  $0 myproject"
    echo "  $0 myproject 3.11"
    echo "  $0 myproject 3.11.5"
    echo ""
    echo "옵션:"
    echo "  환경이름     : 생성할 환경의 이름 (필수)"
    echo "  python버전   : 사용할 Python 버전 (선택, 기본값: 3.11)"
    exit 1
}

# 인수 검증
if [ $# -lt 1 ]; then
    print_error "환경 이름이 필요합니다."
    usage
fi

ENV_NAME=$1
PYTHON_VERSION=${2:-3.11}
VENV_PATH=".venv"  # 명시적으로 가상 환경 경로 설정

print_info "환경 설정을 시작합니다..."
print_info "환경 이름: $ENV_NAME"
print_info "Python 버전: $PYTHON_VERSION"
print_info "가상 환경 경로: $VENV_PATH"

# UV 설치 확인 및 자동 설치
install_uv() {
    print_info "UV를 설치합니다..."
    
    # 공식 설치 스크립트 사용 (권장)
    print_info "공식 설치 스크립트를 사용합니다..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # PATH 업데이트 (가능한 설치 경로들 추가)
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    
    # 환경 파일이 있다면 source
    if [ -f "$HOME/.local/bin/env" ]; then
        source "$HOME/.local/bin/env"
    fi
    
    # 설치 확인
    if command -v uv &> /dev/null; then
        print_success "UV가 성공적으로 설치되었습니다!"
        uv --version
    else
        print_error "UV 설치에 실패했습니다."
        print_info "수동 설치 방법:"
        echo "  1. 공식 스크립트: curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo "  2. pip: pip install uv"
        echo "  3. pipx: pipx install uv"
        exit 1
    fi
}

if ! command -v uv &> /dev/null; then
    print_warning "UV가 설치되어 있지 않습니다."
    read -p "UV를 자동으로 설치하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_uv
    else
        print_error "UV가 필요합니다. 수동으로 설치해주세요."
        print_info "설치 방법:"
        echo "  1. 공식 스크립트 (권장): curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo "  2. pip: pip install uv"
        echo "  3. pipx: pipx install uv"
        exit 1
    fi
fi

# 1. 가상 환경 생성 확인
print_info "가상 환경 확인 중..."
if [ ! -d "$VENV_PATH" ]; then
    print_info "가상 환경을 생성합니다..."
    uv venv
    print_success "가상 환경이 생성되었습니다."
else
    print_warning "이미 $VENV_PATH 가상 환경이 존재합니다. 기존 환경을 사용합니다."
fi

# 2. 프로젝트 초기화
print_info "프로젝트 초기화 중..."
if [ ! -f "pyproject.toml" ]; then
    uv init
    print_success "프로젝트가 초기화되었습니다."
else
    print_warning "이미 pyproject.toml이 존재합니다. 기존 프로젝트를 사용합니다."
fi

# 3. Python 버전 설정
print_info "Python $PYTHON_VERSION 설정 중..."
uv python pin $PYTHON_VERSION
print_success "Python $PYTHON_VERSION이 설정되었습니다."

# 4. 의존성 설치
print_info "필수 패키지 설치 중..."
uv add ipykernel jupyter

# 5. requirements.txt 확인 및 설치
if [ -f "requirements.txt" ]; then
    print_info "requirements.txt 발견. 의존성 설치 중..."
    uv pip install -r requirements.txt
    
    # 명시적으로 lockfile 생성
    print_info "의존성 잠금 파일 생성 중..."
    uv pip compile requirements.txt --output-file requirements.lock
    print_success "의존성 잠금 파일이 생성되었습니다: requirements.lock"
    print_success "requirements.txt의 패키지들이 설치되었습니다."
else
    print_warning "requirements.txt가 없습니다. 기본 패키지만 설치되었습니다."
fi

# 6. Jupyter 커널 등록
print_info "Jupyter 커널 등록 중..."
DISPLAY_NAME="$ENV_NAME (UV)"

# 기존 커널이 있다면 제거
if jupyter kernelspec list 2>/dev/null | grep -q "$ENV_NAME"; then
    print_warning "기존 '$ENV_NAME' 커널을 제거합니다..."
    jupyter kernelspec remove -f "$ENV_NAME" || {
        print_warning "커널 제거 실패, 계속 진행합니다..."
    }
fi

# 새 커널 등록 (에러 처리 추가)
uv run python -m ipykernel install --user --name "$ENV_NAME" --display-name "$DISPLAY_NAME" || {
    print_error "Jupyter 커널 등록에 실패했습니다."
    print_info "수동으로 등록하려면: python -m ipykernel install --user --name \"$ENV_NAME\" --display-name \"$DISPLAY_NAME\""
    exit 1
}
print_success "Jupyter 커널이 '$DISPLAY_NAME'로 등록되었습니다."

# 7. 설치 확인
print_info "설치 확인 중..."
echo ""
echo "=== 설치된 Python 버전 ==="
uv run python --version

echo ""
echo "=== 설치된 패키지 목록 ==="
uv pip list

echo ""
echo "=== 등록된 Jupyter 커널 ==="
jupyter kernelspec list 2>/dev/null | grep -E "(Available|$ENV_NAME)" || echo "커널 목록을 가져올 수 없습니다."

echo ""
print_success "환경 설정이 완료되었습니다!"
echo ""
echo "다음 단계:"
echo "1. 가상 환경 활성화: uv venv" 
echo "2. Jupyter Lab 실행: jupyter lab"
echo "3. 새 노트북 생성 시 '$DISPLAY_NAME' 커널 선택"
echo "4. 스크립트 실행: uv run python your_script.py"
echo ""
print_info "기존 방식으로 환경을 활성화하려면: source $VENV_PATH/bin/activate"
print_info "UV 권장 방식으로 환경 활성화: uv venv"