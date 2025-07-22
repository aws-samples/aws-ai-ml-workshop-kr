#!/bin/bash

# Claude Code 설치 스크립트 (Ubuntu/macOS)
# 사용법: bash install_claude_code.sh

set -e  # 에러 발생시 스크립트 종료

echo "🚀 Claude Code 설치를 시작합니다..."

# OS 감지
OS=""
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo "📋 감지된 운영체제: Linux (Ubuntu/Debian)"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    echo "📋 감지된 운영체제: macOS"
else
    echo "❌ 지원되지 않는 운영체제입니다. Ubuntu 20.04+ 또는 macOS 10.15+가 필요합니다."
    exit 1
fi

# 시스템 요구사항 체크
echo "🔍 시스템 요구사항을 확인하는 중..."

# 메모리 체크 (4GB 이상)
if [[ "$OS" == "linux" ]]; then
    MEMORY_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    MEMORY_GB=$((MEMORY_KB / 1024 / 1024))
elif [[ "$OS" == "macos" ]]; then
    MEMORY_BYTES=$(sysctl -n hw.memsize)
    MEMORY_GB=$((MEMORY_BYTES / 1024 / 1024 / 1024))
fi

if [[ $MEMORY_GB -lt 4 ]]; then
    echo "⚠️  경고: 시스템 메모리가 4GB 미만입니다 (현재: ${MEMORY_GB}GB). 성능에 영향을 줄 수 있습니다."
fi

# Node.js 설치 확인 및 설치
echo "🔧 Node.js 설치를 확인하는 중..."

if ! command -v node &> /dev/null; then
    echo "📦 Node.js가 설치되어 있지 않습니다. 설치를 진행합니다..."
    
    if [[ "$OS" == "linux" ]]; then
        # Ubuntu/Debian용 Node.js 설치
        curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
        sudo apt-get install -y nodejs
    elif [[ "$OS" == "macos" ]]; then
        # Homebrew가 있는지 확인
        if command -v brew &> /dev/null; then
            echo "🍺 Homebrew를 사용하여 Node.js를 설치합니다..."
            brew install node
        else
            echo "❌ Homebrew가 설치되어 있지 않습니다."
            echo "   다음 중 하나를 선택해주세요:"
            echo "   1. Homebrew 설치: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            echo "   2. https://nodejs.org 에서 직접 다운로드"
            exit 1
        fi
    fi
else
    echo "✅ Node.js가 이미 설치되어 있습니다."
fi

# Node.js 버전 확인
NODE_VERSION=$(node --version | sed 's/v//')
NODE_MAJOR_VERSION=$(echo $NODE_VERSION | cut -d. -f1)

if [[ $NODE_MAJOR_VERSION -lt 18 ]]; then
    echo "❌ Node.js 18+ 버전이 필요합니다. 현재 버전: $NODE_VERSION"
    echo "   최신 버전으로 업그레이드해주세요."
    exit 1
fi

echo "✅ Node.js 버전 확인 완료: v$NODE_VERSION"

# npm 권한 설정 (Linux용)
if [[ "$OS" == "linux" ]]; then
    echo "🔐 npm 전역 설치 권한을 설정하는 중..."
    
    # npm 전역 디렉토리를 홈 디렉토리로 변경
    mkdir -p ~/.npm-global
    npm config set prefix '~/.npm-global'
    
    # PATH에 추가 (bashrc/zshrc에 추가)
    if [[ $SHELL == *"zsh"* ]]; then
        SHELL_RC="$HOME/.zshrc"
    else
        SHELL_RC="$HOME/.bashrc"
    fi
    
    if ! grep -q "/.npm-global/bin" "$SHELL_RC"; then
        echo 'export PATH=~/.npm-global/bin:$PATH' >> "$SHELL_RC"
        export PATH=~/.npm-global/bin:$PATH
        echo "✅ PATH 설정이 $SHELL_RC에 추가되었습니다."
    fi
fi

# Claude Code 설치
echo "📥 Claude Code를 설치하는 중..."

if [[ "$OS" == "linux" ]]; then
    # WSL 환경인지 확인
    if [[ -n "${WSL_DISTRO_NAME}" ]]; then
        echo "🔧 WSL 환경이 감지되었습니다. 특별한 설정을 적용합니다..."
        npm config set os linux
        npm install -g @anthropic-ai/claude-code --force --no-os-check
    else
        npm install -g @anthropic-ai/claude-code
    fi
else
    npm install -g @anthropic-ai/claude-code
fi

# 설치 확인
if command -v claude &> /dev/null; then
    echo "✅ Claude Code 설치가 완료되었습니다!"
else
    echo "❌ Claude Code 설치에 실패했습니다."
    echo "   수동으로 다시 시도해보세요: npm install -g @anthropic-ai/claude-code"
    exit 1
fi

# 터미널 최적화 안내
echo ""
echo "🎯 터미널 최적화 권장사항:"
echo ""

if [[ "$OS" == "macos" ]]; then
    echo "📱 macOS Terminal.app 사용시:"
    echo "   - Settings → Profiles → Keyboard에서 'Use Option as Meta Key' 체크"
    echo "   - 이후 Option+Enter로 줄바꿈 가능"
    echo ""
    echo "📱 iTerm2 사용시:"
    echo "   - Settings → Profiles → Keys → General에서"
    echo "   - Left/Right Option key를 'Esc+'로 설정"
    echo "   - Claude Code 실행 후 '/terminal-setup' 명령어로 Shift+Enter 설정 가능"
fi

echo "🔔 알림 설정:"
echo "   - 터미널에서 소리 알림을 위해 시스템 알림 권한 허용"
if [[ "$OS" == "macos" ]]; then
    echo "   - 시스템 설정 → 알림 → [터미널 앱]에서 알림 허용"
fi

echo ""
echo "🎉 설치가 완료되었습니다!"
echo ""
echo "📋 다음 단계:"
echo "1. 새 터미널 세션을 열거나 'source ~/.bashrc' (또는 ~/.zshrc) 실행"
echo "2. 프로젝트 디렉토리로 이동: cd your-project"
echo "3. Claude Code 시작: claude"
echo "4. 인증 과정 완료 (Anthropic Console 또는 Claude Pro/Max 계정)"
echo ""
echo "💡 팁:"
echo "   - 첫 사용시 '/config' 명령어로 설정 조정"
echo "   - 'CLAUDE.md' 파일 생성으로 프로젝트 가이드 작성"
echo "   - 줄바꿈: '\' + Enter 또는 Option+Enter (Mac)"
echo ""
echo "🆘 문제가 발생하면:"
echo "   - '/help' 명령어 사용"
echo "   - https://docs.anthropic.com/en/docs/claude-code/troubleshooting 참조"