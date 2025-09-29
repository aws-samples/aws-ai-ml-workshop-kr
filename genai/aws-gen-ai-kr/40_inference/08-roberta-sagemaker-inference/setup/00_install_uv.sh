#!/bin/bash

# uv 설치 스크립트
# KLUE RoBERTa SageMaker Inference 환경 설정

set -e

echo "=========================================="
echo "UV Package Manager 설치 중..."
echo "=========================================="

# uv 설치 (최신 버전)
curl -LsSf https://astral.sh/uv/install.sh | sh

# PATH 업데이트 (현재 세션)
export PATH="$HOME/.cargo/bin:$PATH"

# .bashrc에 PATH 추가 (영구 설정)
if ! grep -q 'export PATH="$HOME/.cargo/bin:$PATH"' ~/.bashrc; then
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    echo "✅ PATH가 ~/.bashrc에 추가되었습니다."
fi

# uv 버전 확인
echo ""
echo "=========================================="
echo "UV 설치 완료!"
echo "=========================================="
uv --version

echo ""
echo "🎉 UV 설치가 완료되었습니다!"
echo "새 터미널을 열거나 'source ~/.bashrc'를 실행하여 PATH를 업데이트하세요."