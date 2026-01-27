# Fine-tuning

Amazon Nova 2 Lite 모델을 가드레일 분류 작업에 맞게 Fine-tuning하는 파이프라인

## 개요

이 모듈은 Amazon Bedrock에서 Nova 2 Lite 모델을 Fine-tuning하여 패션 이커머스 반품 요청의 Safe/Unsafe 분류를 수행하는 커스텀 모델을 생성합니다.

```
훈련 데이터 (JSONL) → S3 업로드 → Bedrock Fine-tuning → 커스텀 모델
```

---

## 사전 준비

### 1. 환경 설정

프로젝트 루트의 [README.md](../README.md)의 "사전 준비" 섹션을 참조하세요.

### 2. 데이터 준비

Fine-tuning 전에 데이터 변환이 필요합니다:

```bash
# 프로젝트 루트에서 실행
python data-preparation/convert_to_nova_format.py
```

이 스크립트는 `data/nova/training_data.jsonl` 파일을 생성합니다.

---

## 실행

### Fine-tuning 시작

```bash
python fine-tuning/run_fine_tuning.py
```

### 수행 작업

1. **S3 버킷 설정**: 버킷 생성 및 Bedrock 접근 정책 설정
2. **데이터 업로드**: 훈련 데이터를 S3에 업로드
3. **IAM 역할 확인**: Fine-tuning용 IAM 역할 확인/생성
4. **Fine-tuning 작업 생성**: Bedrock 커스터마이제이션 작업 시작
5. **작업 모니터링**: 완료까지 상태 추적

---

## 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| Base Model | amazon.nova-2-lite-v1:0:256k | Nova 2 Lite (256k 컨텍스트) |
| Epochs | 3 | 훈련 에포크 수 |
| Batch Size | 1 | 배치 크기 |
| Learning Rate | 1e-5 | 학습률 |
| Warmup Steps | 0 | 웜업 스텝 |

> **참고**: Nova 2.0은 Validation 데이터셋을 지원하지 않습니다.

---

## 출력 파일

Fine-tuning 완료 후 `evaluation/results/` 폴더에 생성되는 파일:

| 파일 | 설명 |
|------|------|
| `nova_2_lite_job_info.json` | Fine-tuning 작업 정보 (Job ARN, 모델명 등) |
| `nova_2_lite_final_status.json` | 최종 상태 및 메트릭 |

---

## 소요 시간

- **Fine-tuning**: 약 2시간 (데이터셋 크기에 따라 변동)
- **모니터링 간격**: 60초

---

## Fine-tuning 후 단계

Fine-tuning이 완료되면:

1. **평가**: `evaluation/` 폴더의 스크립트로 모델 성능 평가
2. **배포**: 온디맨드 추론을 위한 Custom Model Deployment 생성
3. **프로덕션**: Provisioned Throughput 구매 (프로덕션 사용 시)

### Custom Model Deployment 생성 (AWS Console)

1. Amazon Bedrock 콘솔 접속
2. Custom models → 생성된 모델 선택
3. "Create custom model deployment" 클릭
4. 배포 완료 후 Deployment ARN 확인

---

## 비용 고려사항

### Fine-tuning 비용
- 훈련 토큰 수에 따라 과금
- Nova 2 Lite는 비용 효율적인 선택

### 추론 비용
- **On-demand**: 사용량 기반 과금
- **Provisioned Throughput**: 시간당 고정 비용 (대량 트래픽에 적합)

---

## 문제 해결

### IAM 역할 오류

```
Error: Role cannot be assumed by bedrock.amazonaws.com
```

해결: 스크립트가 자동으로 `BedrockFineTuningRole` 역할을 생성합니다. 권한이 부족하면 수동으로 생성하세요.

### S3 접근 오류

```
Error: Access Denied
```

해결: S3 버킷 정책에 Bedrock 서비스 접근 권한이 있는지 확인하세요.

### 작업 실패

`nova_2_lite_final_status.json`의 `failure_message` 필드를 확인하세요.

---

## 관련 문서

- [Amazon Bedrock Custom Models](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html)
- [Nova 모델 Fine-tuning 가이드](https://docs.aws.amazon.com/bedrock/latest/userguide/nova-fine-tuning.html)
