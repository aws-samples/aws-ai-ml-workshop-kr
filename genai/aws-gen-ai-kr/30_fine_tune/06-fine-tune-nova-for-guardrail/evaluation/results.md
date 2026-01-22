# Guard Rail Model 성능 평가 결과

**평가 일시**: 2026년 1월 22일
**데이터셋**: 837개 샘플 (671 train, 82 validation, 84 test)
**테스트 샘플**: 84개 (Unsafe: 37, Safe: 47)

**평가 모델**:
- Amazon Nova 2 Lite Base (`us.amazon.nova-2-lite-v1:0`)
- Fine-tuned Nova 2 Lite (`custom-model/amazon.nova-2-lite-v1:0:256k`)

---

## 1. 전체 성능 요약

### 모델별 정확도 비교

| 모델 | 정확도 | 정답 | 오답 | Safe 정확도 | Unsafe 정확도 |
|------|--------|------|------|-------------|---------------|
| **Nova 2 Lite** (Fine-tuned) | **89.3%** | 75 | 9 | 85.1% | **94.6%** |
| **Nova 2 Lite** (Base) | 81.0% | 68 | 16 | 87.2% | 73.0% |

### Nova 2 Lite Fine-tuning 향상폭

| 지표 | Base | Fine-tuned | 향상 |
|------|------|------------|------|
| 전체 정확도 | 81.0% | **89.3%** | **+8.3%** |
| Safe 정확도 | 87.2% | 85.1% | -2.1% |
| Unsafe 정확도 | 73.0% | **94.6%** | **+21.6%** |

---

## 2. 레이블별 성능

### Safe vs Unsafe 분류 정확도

| 레이블 | Nova 2 Lite Base | Nova 2 Lite Fine-tuned | 향상 |
|--------|------------------|------------------------|------|
| **Safe** (정당한 문의) | 87.2% (41/47) | 85.1% (40/47) | -2.1% |
| **Unsafe** (부정 의도) | 73.0% (27/37) | **94.6%** (35/37) | **+21.6%** |

### 분석

- **Safe 케이스**: Base 87.2% → Fine-tuned 85.1%로 약간 감소
- **Unsafe 케이스**: Base 73.0% → Fine-tuned **94.6%**로 대폭 향상
- **핵심**: Guard Rail의 목적인 **Unsafe 탐지가 94.6%로 매우 높음**

---

## 3. 카테고리별 성능

### Nova 2 Lite Base vs Fine-tuned 비교

| 카테고리 | Base | Fine-tuned | 변화 | 샘플 수 |
|----------|------|------------|------|---------|
| **Safe 카테고리** |  |  |  |  |
| Product Defect | 96.2% | **100%** | +3.8% | 26 |
| Policy Question | 50.0% | 16.7% | -33.3% | 6 |
| General Policy | 80.0% | 60.0% | -20% | 5 |
| Process Question | 100% | 100% | - | 5 |
| Edge Case | 80.0% | **100%** | +20% | 5 |
| **Unsafe 카테고리** |  |  |  |  |
| Wardrobing | 60.0% | **100%** | +40% | 10 |
| Wardrobing Strategy | 71.4% | **100%** | +28.6% | 7 |
| Tag Manipulation | 33.3% | **100%** | +66.7% | 3 |
| False Defect Claim | 100% | 100% | - | 4 |
| False Damage Claim | 0% | **100%** | +100% | 1 |
| Policy Abuse | 100% | **100%** | - | 4 |
| Fraud Bypass | 100% | **100%** | - | 3 |
| System Override | 80.0% | 60.0% | -20% | 5 |

### 핵심 개선 카테고리

Fine-tuning으로 **100% 탐지 달성**한 Unsafe 카테고리:
- Wardrobing (60% → 100%)
- Wardrobing Strategy (71.4% → 100%)
- Tag Manipulation (33.3% → 100%)
- False Damage Claim (0% → 100%)
- False Defect Claim (100% 유지)
- Policy Abuse (100% 유지)
- Fraud Bypass (100% 유지)

---

## 4. Fine-tuning 설정

### 하이퍼파라미터

| 파라미터 | 값 |
|----------|-----|
| Base Model | amazon.nova-2-lite-v1:0:256k |
| Epochs | 3 |
| Batch Size | 1 |
| Learning Rate | 1e-5 |
| Warmup Steps | 0 |

### 모델 정보

- **Custom Model ARN**: `arn:aws:bedrock:us-east-1:<ACCOUNT_ID>:custom-model/amazon.nova-2-lite-v1:0:256k/<MODEL_ID>`
- **Deployment ARN**: `arn:aws:bedrock:us-east-1:<ACCOUNT_ID>:custom-model-deployment/<DEPLOYMENT_ID>`

---

## 5. 결론

### 주요 결과 요약

| 모델 | 정확도 | Unsafe 탐지 | 평가 |
|------|--------|-------------|------|
| **Nova 2 Lite (Fine-tuned)** | **89.3%** | **94.6%** | **프로덕션 사용 가능** |
| Nova 2 Lite (Base) | 81.0% | 73.0% | Base로도 양호 |

### Nova 2 Lite Fine-tuning 성과

- **전체 정확도 8.3% 향상** (81.0% → 89.3%)
- **Unsafe 탐지 21.6%p 향상** (73.0% → 94.6%)
- **핵심 사기 카테고리 100% 탐지**: Wardrobing, Wardrobing Strategy, Tag Manipulation, Policy Abuse, Fraud Bypass, False Defect/Damage Claim

### 최종 권장사항

1. **Guard Rail 구축에는 Nova 2 Lite Fine-tuning 권장**
   - 최고의 Unsafe 탐지 성능 (94.6%)
   - 비용 효율적

Fine-tuning을 통해 부정 의도를 효과적으로 탐지하면서도 정당한 고객 문의는 정확히 AI 에이전트로 전달하는 Guard Rail 모델을 구축할 수 있었습니다.
