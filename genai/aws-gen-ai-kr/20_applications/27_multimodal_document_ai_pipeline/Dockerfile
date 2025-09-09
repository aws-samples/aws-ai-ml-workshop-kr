# filepath: /workspaces/python-2/llm-hsjeon/slm-fine-tuning/Dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu20.04

# 비대화형 모드 설정
ENV DEBIAN_FRONTEND=noninteractive

# -------------------------
# 1) 시스템 패키지 설치
# -------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        # NVIDIA 관련 유틸리티
        cuda-command-line-tools-12-4 \
        # OCR / PDF 처리 의존성
        tesseract-ocr \
        poppler-utils \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-dev \
        cmake && \
    rm -rf /var/lib/apt/lists/*

# -------------------------
# 2) Python 3.11 설치
# -------------------------
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-distutils && \
    rm -rf /var/lib/apt/lists/*

# -------------------------
# 3) get-pip.py로 Python 3.11 pip 설치
# -------------------------
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    # 심볼릭 링크 설정 (기본 python, pip → Python3.11)
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.11 /usr/bin/pip3 && \
    ln -sf /usr/local/bin/pip /usr/bin/pip

# -------------------------
# 4) NVIDIA 환경변수 설정
# -------------------------
ENV PATH="/usr/local/cuda-12.4/bin:${PATH}"

# -------------------------
# 5) Python 패키지 설치 (requirements.txt 사용)
# -------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -------------------------
# 6) 작업 디렉토리 설정
# -------------------------
#WORKDIR /app

# -------------------------
# 컨테이너 실행 시 기본 명령
# -------------------------
CMD ["python3", "--version"]