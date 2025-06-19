#!/bin/bash

# UV 환경 설정 및 Jupyter 커널 등록 스크립트 (pyproject.toml 기반)
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

# 1. Python 버전 설정 (가상환경 생성 전에)
print_info "Python $PYTHON_VERSION 설정 중..."
uv python pin $PYTHON_VERSION
print_success "Python $PYTHON_VERSION이 설정되었습니다."

# 2. 프로젝트 초기화
print_info "프로젝트 초기화 중..."
if [ ! -f "pyproject.toml" ]; then
    uv init --name "$ENV_NAME"
    print_success "프로젝트가 '$ENV_NAME'으로 초기화되었습니다."
else
    print_warning "이미 pyproject.toml이 존재합니다. 기존 프로젝트를 사용합니다."
fi

# 3. 필수 패키지 추가
print_info "Jupyter 관련 필수 패키지 추가 중..."
uv add ipykernel jupyter

# 4. pyproject.toml 확인 및 의존성 설치
if [ -f "pyproject.toml" ]; then
    print_info "pyproject.toml 발견. 의존성 동기화 중..."
    
    # pyproject.toml의 의존성을 기반으로 환경 동기화
    uv sync
    
    print_success "pyproject.toml 기반으로 의존성이 설치되었습니다."
    print_info "락 파일이 자동으로 생성/업데이트되었습니다: uv.lock"
else
    print_error "pyproject.toml이 없습니다. 프로젝트 초기화에 실패했을 수 있습니다."
    exit 1
fi

# 5. Jupyter 커널 등록
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
    print_info "수동으로 등록하려면: uv run python -m ipykernel install --user --name \"$ENV_NAME\" --display-name \"$DISPLAY_NAME\""
    exit 1
}
print_success "Jupyter 커널이 '$DISPLAY_NAME'로 등록되었습니다."

# 6. 설치 확인
print_info "설치 확인 중..."
echo ""
echo "=== 설치된 Python 버전 ==="
uv run python --version

echo ""
echo "=== 설치된 패키지 목록 ==="
uv pip list

echo ""
echo "=== pyproject.toml 의존성 ==="
if command -v grep &> /dev/null && [ -f "pyproject.toml" ]; then
    grep -A 20 "dependencies = \[" pyproject.toml || echo "의존성 정보를 읽을 수 없습니다."
fi

echo ""
echo "=== 등록된 Jupyter 커널 ==="
jupyter kernelspec list 2>/dev/null | grep -E "(Available|$ENV_NAME)" || echo "커널 목록을 가져올 수 없습니다."

echo ""
print_success "환경 설정이 완료되었습니다!"
echo ""
echo "=== 사용 방법 ==="
echo "1. 패키지 추가: uv add <패키지명>"
echo "2. 패키지 제거: uv remove <패키지명>"
echo "3. 의존성 동기화: uv sync"
echo "4. 스크립트 실행: uv run python your_script.py"
echo "5. Jupyter Lab 실행: uv run jupyter lab"
echo "6. 새 노트북 생성 시 '$DISPLAY_NAME' 커널 선택"
echo ""
echo "=== 파일 정보 ==="
echo "- pyproject.toml: 프로젝트 설정 및 의존성"
echo "- uv.lock: 정확한 버전 락 파일 (버전 관리에 포함 권장)"
echo "- .venv/: 가상 환경 디렉토리 (버전 관리에서 제외)"
echo ""
print_info "전통적인 방식으로 활성화: source $VENV_PATH/bin/activate"
print_info "UV 권장 방식: uv run <명령어>"