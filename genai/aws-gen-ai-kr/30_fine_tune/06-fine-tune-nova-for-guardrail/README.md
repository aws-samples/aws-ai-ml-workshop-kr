# Fine-tune Nova for Guardrail

패션 이커머스 반품/환불 요청에서 부정 의도를 탐지하는 가드레일 모델을 Amazon Nova로 Fine-tuning하는 프로젝트

## 1. 산업 배경

### 1.1 패션 반품에 가드레일이 필요한 이유

패션 이커머스는 다른 산업에 비해 반품률이 매우 높으며, 이로 인한 부정 반품 비용이 급증하고 있습니다:

- **반품률**: 패션 이커머스 20-40% (일반 소매 8-10% 대비)
- **부정 반품 비용**: 2023년 연간 소매 반품 부정 행위 1,030억 달러
- **착용 후 반품**: 소비자의 69%가 반품 의도로 구매했다고 인정
- **브래킷팅**: 여러 사이즈/색상 주문 후 하나만 보유, 나머지 반품

### 1.2 비즈니스 영향

가드레일 모델 도입으로 기대되는 효과:

- **비용 절감**: 처리 전 부정 반품 차단
- **정책 집행**: 반품 규칙의 일관된 적용
- **고객 경험**: 느린 수동 검토 대신 빠른 거부
- **부정 반품 방지**: 남용 패턴 감지 (반복적 착용 후 반품, 허위 주장)

### 1.3 왜 Fine-tuning이 필요한가?

**Base 모델의 한계:**

| 문제 | 설명 |
|------|------|
| **도메인 지식 부족** | 한국 패션 이커머스의 반품 정책과 부정 패턴에 대한 이해 부족 |
| **과도한 친절함** | 대형 모델(Qwen3 32B)은 Unsafe 탐지율 10.8%로 부정 요청도 수용하려는 경향 |
| **맥락 이해 실패** | "불량이라고 하면 되나요?"와 같은 미묘한 부정 의도 구분 실패 |

**Fine-tuning의 장점:**

- **도메인 특화**: 837개의 한국어 반품 시나리오로 정책 위반 패턴 학습
- **의도 파악**: 정직한 질문 vs 속이려는 의도 구분 능력 향상 (+16.2%)
- **비용 효율**: 작은 모델(Nova 2 Lite) + Fine-tuning이 대형 모델보다 효과적
- **일관성**: 동일한 정책 기준으로 일관된 판단

---

## 2. 아키텍처

```
고객 요청 → [가드레일 모델] → Safe? → AI 에이전트 처리
                              → Unsafe? → 차단 및 거부
```

고객과 AI 에이전트 사이에 위치하여 부정한 요청은 차단하고 정당한 요청만 통과시킵니다.

---

## 3. 성능 결과

### 3.1 모델별 정확도 비교

| 모델 | 정확도 | Safe | Unsafe |
|------|--------|------|--------|
| **Nova 2 Lite (Fine-tuned)** | **90.5%** | 91.5% | **89.2%** |
| Nova 2 Lite (Base) | 78.6% | 83.0% | 73.0% |
| Qwen3 32B (Base) | 59.5% | 97.9% | 10.8% |

### 3.2 Fine-tuning 향상폭

- 전체 정확도: **+11.9%** (78.6% → 90.5%)
- Unsafe 탐지: **+16.2%** (73.0% → 89.2%)

### 3.3 100% 탐지 카테고리

Fine-tuning 후 완벽하게 탐지:
- Wardrobing Strategy, Tag Manipulation, False Damage Claim, Policy Abuse, Fraud Bypass

---

## 4. 빠른 시작

### 4.1 사전 준비

**AWS 자격 증명 설정:**
```bash
aws configure
```

**필요 권한:**
- **S3**: 버킷 생성, 읽기/쓰기 (`s3:CreateBucket`, `s3:PutObject`, `s3:GetObject`)
- **Bedrock**: 모델 커스터마이징 (`bedrock:CreateModelCustomizationJob`, `bedrock:GetModelCustomizationJob`)
- **IAM**: 역할 생성 (선택, 스크립트가 자동 생성)

**의존성 설치:**
```bash
pip install boto3 tqdm python-dotenv
```

**환경 변수 설정 (선택):**

`.env` 파일 생성:
```env
AWS_REGION=us-east-1
S3_BUCKET_NAME=guard-rail-fine-tuning-data
```

### 4.2 데이터 검증

```bash
python data-preparation/validate_dataset.py
```

### 4.3 Nova 형식 변환

```bash
python data-preparation/convert_to_nova_format.py
```

### 4.4 Fine-tuning 실행

```bash
python fine-tuning/run_fine_tuning.py
```

### 4.5 모델 평가

```bash
# Base 모델 평가
python evaluation/evaluate_nova.py

# Fine-tuned 모델 평가
python evaluation/evaluate_nova.py --finetuned --deployment-arn <ARN>
```

---

## 5. 프로젝트 구조

```
06-fine-tune-nova-for-guardrail/
├── README.md                    # 프로젝트 개요 (이 파일)
├── CLAUDE.md                    # Claude Code 가이드
│
├── data-preparation/            # 1. 데이터 준비
│   ├── README.md               # 데이터셋 상세 문서
│   ├── config.py               # 설정 파일
│   ├── validate_dataset.py     # 데이터셋 검증
│   ├── generate_additional_samples.py  # 샘플 생성
│   └── convert_to_nova_format.py       # Nova 형식 변환
│
├── fine-tuning/                 # 2. 모델 Fine-tuning
│   ├── README.md               # Fine-tuning 가이드
│   └── run_fine_tuning.py      # Fine-tuning 실행
│
├── evaluation/                  # 3. 모델 평가
│   ├── README.md               # 평가 가이드
│   ├── evaluate_nova.py        # Nova 모델 평가
│   ├── evaluate_qwen.py        # Qwen 모델 평가
│   └── results/                # 평가 결과 JSON 파일
│
├── data/                        # 통합 데이터 폴더
│   ├── raw/                    # 원본 데이터셋 (837 샘플)
│   └── nova/                   # Nova 형식 변환 데이터
│
└── docs/                        # 상세 문서
    ├── DETAILS.md              # 카테고리 분포, 확장 이력
    └── LABELING_GUIDE.md       # 레이블링 가이드
```

---

## 6. 데이터셋 개요

**총 837 샘플** | 13개 카테고리 | 48/52 Unsafe/Safe 비율

| 레이블 | 카테고리 수 | 샘플 수 | 예시 |
|--------|------------|---------|------|
| **Unsafe** | 8가지 | 402 | 착용 후 반품, 허위 불량 주장, 정책 우회 요청 |
| **Safe** | 5가지 | 435 | 정당한 불량 신고, 정책 문의, 절차 질문 |

> 자세한 데이터셋 정보, 카테고리별 분포, 레이블링 기준은 [data-preparation/README.md](data-preparation/README.md) 참조

---

## 7. 문서

### 7.1 모듈별 가이드

| 모듈 | 설명 | 문서 |
|------|------|------|
| **Data Preparation** | 데이터셋 구조, 검증, 변환 | [README](data-preparation/README.md) |
| **Fine-tuning** | AWS Bedrock Fine-tuning 파이프라인 | [README](fine-tuning/README.md) |
| **Evaluation** | 모델 성능 평가 | [README](evaluation/README.md) |

### 7.2 참고 문서

| 문서 | 설명 |
|------|------|
| [docs/LABELING_GUIDE.md](docs/LABELING_GUIDE.md) | Safe/Unsafe 판단 상세 가이드, 경계 사례 |
| [docs/DETAILS.md](docs/DETAILS.md) | 카테고리 분포, 확장 이력 |
| [evaluation/results.md](evaluation/results.md) | 상세 성능 평가 결과 |

---

## 8. 결론

Nova 2 Lite Fine-tuning을 통해:
- **Unsafe 탐지 89.2%** 달성
- 핵심 사기 패턴 완벽 탐지
- 비용 효율적인 가드레일 모델 구축

**프로덕션 권장**: Nova 2 Lite Fine-tuned 모델

---

**최종 업데이트**: 2026년 1월
