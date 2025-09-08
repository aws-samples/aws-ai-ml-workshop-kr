FROM python:3.12-slim

# 시스템 패키지 업데이트 및 기본 도구 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 한글 폰트 설치 (기존 프로젝트의 한글 폰트 지원) - 임시 비활성화
COPY install_korean_font.sh /tmp/
RUN chmod +x /tmp/install_korean_font.sh && /tmp/install_korean_font.sh

# Python 패키지 설치
RUN pip install --no-cache-dir \
    flask==3.0.0 \
    pandas==2.3.1 \
    numpy==1.26.4 \
    matplotlib==3.10.5 \
    seaborn==0.13.2 \
    plotly==6.3.0 \
    scikit-learn==1.7.1 \
    koreanize-matplotlib==0.1.1 \
    lovelyplots==1.0.2 \
    weasyprint==66.0

# 작업 디렉토리 설정
WORKDIR /app

# 애플리케이션 코드 복사
COPY . .

# 포트 설정
EXPOSE 8080

# 헬스체크 설정
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 애플리케이션 실행
CMD ["python", "api.py"]