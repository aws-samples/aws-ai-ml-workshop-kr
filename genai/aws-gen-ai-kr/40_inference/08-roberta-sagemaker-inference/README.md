# KLUE RoBERTa SageMaker Inference

이 워크샵은 KLUE RoBERTa 모델을 BiEncoder 방식으로 SageMaker Endpoint에 배포하고, 쿼리와 문서의 임베딩을 생성하여 의미적 유사도를 계산하는 실습입니다.

## 사전 준비

### 1. 환경 설정

프로젝트 루트에서 자동 설정 스크립트를 실행합니다:

```bash
./setup/run_all_setup.sh
```

이 스크립트는 다음을 자동으로 수행합니다:
- UV 패키지 매니저 설치
- Python 3.11 가상환경 생성
- 필요한 패키지 설치 (PyTorch, SageMaker SDK 등)
- 환경 테스트

### 2. .env 파일 작성

프로젝트 루트에 `.env` 파일을 생성하고 AWS SageMaker Role ARN을 설정합니다:

```bash
SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE
```

**필수 IAM 권한**: Role에는 다음 권한이 필요합니다:
- AmazonEC2ContainerRegistryFullAccess
- AmazonS3FullAccess
- AmazonSageMakerFullAccess

## 실습 진행

### 가상환경 활성화

```bash
source .venv/bin/activate
```

### VS Code에서 노트북 실행

VS Code에서 `.ipynb` 파일을 열고 우측 상단의 커널 선택 버튼을 클릭하여 `.venv` 가상환경의 Python 인터프리터를 선택합니다.

### 노트북 실행 순서

`notebook` 폴더에는 두 개의 노트북이 있습니다:

1. **01_sagemaker_inference_dual_encoder_local.ipynb**
   - 로컬 환경에서 SageMaker Local Mode를 사용한 추론
   - 모델 동작 검증 및 빠른 테스트
   - 로컬 Docker를 사용하여 SageMaker 환경 시뮬레이션

2. **02_sagemaker_inference_dual_encoder.ipynb**
   - 실제 SageMaker Endpoint 생성 및 배포
   - S3에 모델 업로드
   - ml.g4dn.xlarge 인스턴스를 사용한 실제 추론

**참고**: 01번 노트북을 스킵하고 02번 노트북만 단독으로 실행해도 됩니다.

## 주요 기능

- **BiEncoder 방식**: 쿼리와 문서를 독립적으로 인코딩하여 임베딩 생성
- **배치 추론**: 여러 쿼리와 문서를 동시에 처리
- **코사인 유사도 계산**: 쿼리-문서 간 의미적 유사도 측정
- **SageMaker Local Mode**: 로컬에서 빠른 테스트 및 디버깅

## 모델 구조

```
model.tar.gz/
├── config.json
├── model.safetensors
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
├── vocab.txt
└── code/
    ├── inference.py
    └── requirements.txt
```

## 기술 스택

- **모델**: KLUE RoBERTa Base
- **프레임워크**: PyTorch 2.5.0
- **배포**: AWS SageMaker
- **Python**: 3.11