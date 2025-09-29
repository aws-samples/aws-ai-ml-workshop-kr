# Setup Scripts for KLUE RoBERTa SageMaker Inference

KLUE RoBERTa SageMaker 추론 환경을 설정하기 위한 간단한 스크립트들입니다.

## 📁 파일 구성

```
setup/
├── 00_install_uv.sh        # UV 패키지 매니저 설치
├── 01_setup_environment.sh # Python 가상환경 및 패키지 설치
├── 02_test_environment.sh  # 환경 설정 테스트
├── run_all_setup.sh       # 원클릭 전체 설정
├── pyproject.toml         # 프로젝트 의존성 정의
└── README.md              # 이 파일
```

## 🚀 설치 순서

### 원클릭 설정 (추천)
```bash
cd /home/ubuntu/lab/16-robert-sagemaker-inference/setup
./run_all_setup.sh
```

### 단계별 설정
```bash
# 1단계: UV 설치
./00_install_uv.sh

# 2단계: 환경 설정
./01_setup_environment.sh

# 3단계: 환경 테스트
./02_test_environment.sh
```

## 📦 설치되는 패키지

- **PyTorch 2.0.1** (CUDA 지원)
- **Transformers** (≥4.30.0)
- **SageMaker SDK**
- **Boto3** (AWS SDK)
- **NumPy**
- **Jupyter Lab**
- **IPython Kernel**

## 🎯 특징

- **빠른 설치**: UV 패키지 매니저 사용
- **GPU 지원**: CUDA 11.8 지원 PyTorch
- **Jupyter 통합**: 자동 커널 등록

## 📋 사용법

```bash
# 가상환경 활성화
cd /home/ubuntu/lab/16-robert-sagemaker-inference
source .venv/bin/activate

# Jupyter Lab 실행
jupyter lab

# 모델 테스트
python test_local_model.py
```

## 📊 환경 정보

- **Python**: 3.10
- **PyTorch**: 2.0.1 + CUDA 11.8
- **Jupyter 커널**: `klue-roberta-inference`