#!/bin/bash

# 스크립트 실행 시작 메시지
echo "한글 폰트 설치 및 matplotlib 설정을 시작합니다..."

# 필요한 패키지 설치
echo "나눔 폰트 설치 중..."
if command -v apt-get > /dev/null; then
    # Ubuntu/Debian 계열
    sudo apt-get update
    sudo apt-get install -y fonts-nanum
elif command -v yum > /dev/null; then
    # CentOS/RHEL 계열
    sudo yum install -y nanum-fonts-all
else
    # 패키지 관리자를 찾을 수 없는 경우 직접 폰트 다운로드
    echo "패키지 관리자를 찾을 수 없습니다. 직접 폰트를 다운로드합니다..."
    mkdir -p ~/.fonts
    cd ~/.fonts
    if command -v wget > /dev/null; then
        wget https://github.com/naver/nanumfont/raw/master/NanumGothic.ttf
    elif command -v curl > /dev/null; then
        curl -O https://github.com/naver/nanumfont/raw/master/NanumGothic.ttf
    else
        echo "wget 또는 curl이 설치되어 있지 않습니다. 폰트 다운로드를 건너뜁니다."
    fi
fi

# 폰트 캐시 갱신
echo "폰트 캐시를 갱신합니다..."
fc-cache -f -v

# 설치된 폰트 확인
echo "설치된 나눔 폰트 목록:"
fc-list | grep "Nanum"

# matplotlib 설정 파일 찾기
echo "matplotlib 설정 파일 찾는 중..."
MATPLOTLIB_DIR=$(python -c "import matplotlib; print(matplotlib.get_configdir())")
MATPLOTLIB_DATA_DIR=$(python -c "import matplotlib; print(matplotlib.get_data_path())")
MATPLOTLIBRC_PATH="${MATPLOTLIB_DATA_DIR}/matplotlibrc"

echo "matplotlib 설정 파일 경로: ${MATPLOTLIBRC_PATH}"

# matplotlibrc 파일 백업 및 수정
if [ -f "$MATPLOTLIBRC_PATH" ]; then
    echo "matplotlibrc 파일 백업 및 수정 중..."
    cp "$MATPLOTLIBRC_PATH" "${MATPLOTLIBRC_PATH}.backup"
    
    # font.family 설정 변경 또는 추가
    if grep -q "^#font.family" "$MATPLOTLIBRC_PATH"; then
        sed -i 's/^#font.family.*/font.family: sans-serif/' "$MATPLOTLIBRC_PATH"
    elif grep -q "^font.family" "$MATPLOTLIBRC_PATH"; then
        sed -i 's/^font.family.*/font.family: sans-serif/' "$MATPLOTLIBRC_PATH"
    else
        echo "font.family: sans-serif" >> "$MATPLOTLIBRC_PATH"
    fi
    
    # font.sans-serif 설정 변경 또는 추가
    if grep -q "^#font.sans-serif" "$MATPLOTLIBRC_PATH"; then
        sed -i 's/^#font.sans-serif.*/font.sans-serif: NanumGothic, DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, sans-serif/' "$MATPLOTLIBRC_PATH"
    elif grep -q "^font.sans-serif" "$MATPLOTLIBRC_PATH"; then
        sed -i 's/^font.sans-serif.*/font.sans-serif: NanumGothic, DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, sans-serif/' "$MATPLOTLIBRC_PATH"
    else
        echo "font.sans-serif: NanumGothic, DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, sans-serif" >> "$MATPLOTLIBRC_PATH"
    fi
    
    # axes.unicode_minus 설정 변경 또는 추가
    if grep -q "^#axes.unicode_minus" "$MATPLOTLIBRC_PATH"; then
        sed -i 's/^#axes.unicode_minus.*/axes.unicode_minus: False/' "$MATPLOTLIBRC_PATH"
    elif grep -q "^axes.unicode_minus" "$MATPLOTLIBRC_PATH"; then
        sed -i 's/^axes.unicode_minus.*/axes.unicode_minus: False/' "$MATPLOTLIBRC_PATH"
    else
        echo "axes.unicode_minus: False" >> "$MATPLOTLIBRC_PATH"
    fi
else
    echo "matplotlibrc 파일을 찾을 수 없습니다. 새로운 설정 파일을 생성합니다..."
    mkdir -p "${MATPLOTLIB_DIR}"
    cat > "${MATPLOTLIB_DIR}/matplotlibrc" << EOF
font.family: sans-serif
font.sans-serif: NanumGothic, DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, sans-serif
axes.unicode_minus: False
EOF
    echo "새로운 설정 파일이 생성되었습니다: ${MATPLOTLIB_DIR}/matplotlibrc"
fi

# matplotlib 폰트 캐시 삭제
echo "matplotlib 폰트 캐시 삭제 중..."
rm -rf ~/.cache/matplotlib/* 2>/dev/null || echo "캐시 파일이 없거나 삭제할 수 없습니다."

echo "설정이 완료되었습니다. 이제 matplotlib에서 한글이 올바르게 표시될 것입니다."
echo "문제가 계속된다면 다음 코드를 파이썬 스크립트에 직접 추가해보세요:"
echo "
import matplotlib.font_manager as fm
# 나눔고딕 폰트 경로 찾기
font_path = fm.findfont('NanumGothic')
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False
"

# 테스트 스크립트 생성
echo "테스트 스크립트 생성 중..."
cat > test_korean_font.py << EOF
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 그래프 생성
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='사인 함수')
plt.plot(x, -y, label='-사인 함수')
plt.title('한글 테스트: 사인 함수 그래프')
plt.xlabel('x축 라벨')
plt.ylabel('y축 라벨')
plt.legend()
plt.grid(True)
plt.savefig('korean_font_test.png')
plt.show()

print("테스트 완료! korean_font_test.png 파일을 확인하세요.")
EOF

echo "테스트 스크립트가 생성되었습니다: test_korean_font.py"
echo "다음 명령어로 테스트할 수 있습니다: python test_korean_font.py"