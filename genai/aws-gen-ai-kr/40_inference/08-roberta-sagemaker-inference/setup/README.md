# KLUE RoBERTa SageMaker 추론 환경 설정

KLUE RoBERTa 모델을 SageMaker에서 추론하기 위한 Conda 기반 환경 설정입니다.

## 📁 파일 구성

```
setup/
├── setup.sh           # Conda 기반 환경 설정 스크립트
├── requirements.txt   # Python 패키지 목록
└── README.md         # 이 문서
```

## 🚀 빠른 시작

```bash
# 1. setup 디렉토리로 이동
cd /home/ec2-user/SageMaker/aws-ai-ml-workshop-kr/genai/aws-gen-ai-kr/40_inference/08-roberta-sagemaker-inference/setup

# 2. 설정 스크립트 실행
./setup.sh
```

## 📦 설치되는 패키지

### 핵심 패키지
- **PyTorch**: 2.5.0 (CUDA 12.1 지원)
- **Transformers**: ≥4.30.0
- **NumPy**: <2.0

### AWS 관련
- **SageMaker SDK**: ≥2.251.0
- **Boto3**: ≥1.26.0

### 기타
- **IPyKernel**: Jupyter 커널 연결용
- **Python-dotenv**: 환경 변수 관리

## 🎯 특징

- **Conda 환경**: 안정적인 패키지 관리
- **GPU 지원**: CUDA 12.1 지원 PyTorch
- **SageMaker 최적화**: 노트북 인스턴스에 최적화된 설정
- **Jupyter 통합**: 자동 커널 등록

## ✅ 설치 확인

### 터미널에서 확인
```bash
# Conda 환경 활성화
conda activate klue_roberta

# Python 버전 확인
python --version  # Python 3.11.x

# PyTorch 및 CUDA 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Jupyter 노트북에서 확인
1. 커널 선택: **"conda_klue_roberta"**
2. 다음 코드 실행:
```python
!which python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### 패키지 import 오류
```bash
# Conda 환경 확인
conda activate klue_roberta
pip list | grep torch

# 패키지 재설치
pip install -r requirements.txt
```

## 📊 환경 정보

- **Conda 환경명**: `klue_roberta`
- **Python**: 3.11
- **PyTorch**: 2.5.0 + CUDA 12.1
- **Jupyter 커널**: "KLUE RoBERTa (Python 3.11)"

## 🗑️ 환경 초기화

```bash
# Conda 환경 및 커널 제거
conda env remove -n klue_roberta -y
jupyter kernelspec uninstall klue_roberta -y

# 재설치
./setup.sh
```