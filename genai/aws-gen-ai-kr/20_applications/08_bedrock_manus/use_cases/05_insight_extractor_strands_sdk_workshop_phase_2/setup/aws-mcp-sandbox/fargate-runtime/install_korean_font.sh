#!/bin/bash

# 스크립트 실행 시작 메시지
echo "한글 폰트 설치 및 matplotlib 설정을 시작합니다..."

# 필요한 패키지 설치
echo "나눔 폰트 설치 중..."
if command -v apt-get > /dev/null; then
    # Ubuntu/Debian 계열 (Docker 컨테이너에서는 sudo 없이 실행)
    apt-get update
    apt-get install -y fonts-nanum fontconfig
elif command -v yum > /dev/null; then
    # CentOS/RHEL 계열
    yum install -y nanum-fonts-all
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
fc-list | grep "Nanum" || echo "나눔 폰트를 찾을 수 없습니다."

# matplotlib 설정을 위한 Python 스크립트 실행
echo "matplotlib 한글 폰트 설정 중..."
python3 -c "
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# matplotlib 설정 디렉토리 확인
config_dir = matplotlib.get_configdir()
data_dir = matplotlib.get_data_path()

print(f'matplotlib config dir: {config_dir}')
print(f'matplotlib data dir: {data_dir}')

# 나눔 폰트 경로 확인 및 추가
font_dirs = ['/usr/share/fonts/truetype/nanum/', '/usr/share/fonts/nanum/', '~/.fonts/']
for font_dir in font_dirs:
    expanded_dir = os.path.expanduser(font_dir)
    if os.path.exists(expanded_dir):
        try:
            font_files = fm.findSystemFonts(fontpaths=expanded_dir)
            for font_file in font_files:
                fm.fontManager.addfont(font_file)
            print(f'Added fonts from {expanded_dir}')
        except Exception as e:
            print(f'Error adding fonts from {expanded_dir}: {e}')

# 폰트 캐시 갱신
fm._rebuild()
print('Font cache rebuilt')

# matplotlib 설정 파일 생성/수정
matplotlibrc_path = os.path.join(config_dir, 'matplotlibrc')
os.makedirs(config_dir, exist_ok=True)

# 설정 내용
config_content = '''
font.family: sans-serif
font.sans-serif: NanumGothic, DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, sans-serif
axes.unicode_minus: False
'''

with open(matplotlibrc_path, 'w') as f:
    f.write(config_content.strip())

print(f'Created matplotlib config file: {matplotlibrc_path}')

# 사용 가능한 나눔 폰트 확인
nanum_fonts = [f.name for f in fm.fontManager.ttflist if 'Nanum' in f.name]
if nanum_fonts:
    print('사용 가능한 나눔 폰트:', nanum_fonts)
else:
    print('나눔 폰트를 찾을 수 없습니다.')

# matplotlib 캐시 삭제
import shutil
cache_dir = matplotlib.get_cachedir()
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f'Removed matplotlib cache: {cache_dir}')
"

# 테스트 스크립트 생성
echo "테스트 스크립트 생성 중..."
cat > /tmp/test_korean_font.py << 'EOF'
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# 사용 가능한 폰트 확인
print("사용 가능한 폰트 목록:")
nanum_fonts = [f.name for f in fm.fontManager.ttflist if 'Nanum' in f.name]
if nanum_fonts:
    print("나눔 폰트:", nanum_fonts)
    plt.rcParams['font.family'] = nanum_fonts[0]
else:
    print("나눔 폰트를 찾을 수 없습니다. 기본 설정을 사용합니다.")

plt.rcParams['axes.unicode_minus'] = False

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
plt.savefig('/tmp/korean_font_test.png', dpi=100, bbox_inches='tight')

print("테스트 완료! /tmp/korean_font_test.png 파일이 생성되었습니다.")
print("현재 폰트 설정:", plt.rcParams['font.family'])
EOF

echo "설정이 완료되었습니다. 이제 matplotlib에서 한글이 올바르게 표시될 것입니다."
echo "테스트하려면: python /tmp/test_korean_font.py"