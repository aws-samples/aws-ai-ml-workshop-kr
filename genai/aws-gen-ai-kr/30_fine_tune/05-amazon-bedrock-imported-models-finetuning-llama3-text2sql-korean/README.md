# llama3-text2sql-korean

## 🔭 Overview

- 이 repo에는 Amazon Sagemaker를 이용해 **llama3-8b 모델**을 **한국어 호환 text to sql** 용도로 파인 튜닝하는 예제를 작성했습니다.

- Amazon Bedrock Imported Models에 파인 튜닝을 완료한 모델을 업로드하고 모델의 성능을 평가하는 방식도 보실 수 있습니다.

### 개발 환경

- Amazon Sagemaker Notebook instance (ml.t3.medium)
- `conda_pytorch_p310` 이용
- Amazon Bedrock Imported Models 이용 추론

### 사용 모델 및 데이터 셋

- 사용 모델: [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- 사용 데이터 셋: [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context)

#### 데이터 셋 예시

```
Answer: SELECT COUNT(*) FROM head WHERE age > 56
Question: How many heads of the departments are older than 56 ?
Context: CREATE TABLE head (age INTEGER)
```

- Amazon Translate 이용하여 user question을 한국어로 번역한 후 사용

#### 번역 및 전처리 예시

```
{"messages":[{"content":"You are a powerful text-to-SQL model. Your job is to answer questions about a database.You can use the following table schema for context: CREATE TABLE table_name_52 (bronze VARCHAR, rank VARCHAR, total VARCHAR, nation VARCHAR)","role":"system"},{"content":"Return the SQL query that answers the following question: 총 합계가 3보다 작고 폴란드 국가이고 순위가 4보다 큰 브론즈의 총계는 얼마입니까?","role":"user"},{"content":"SELECT COUNT(bronze) FROM table_name_52 WHERE total < 3 AND nation = \"poland\" AND rank > 4","role":"assistant"}]}

```

## 💻 실행 안내

> 이 레포지토리는 실습용, 프로덕션용 에셋을 전부 담고 있습니다. use case에 맞춰 아래 옵션 중 하나를 선택하여 실행하세요.

### Option 1 - 실습용 3200개 샘플 데이터셋 이용

- 실행 소요 시간: 약 1시간 30분 (데이터 전처리 5분, 파인 튜닝 35분, Amazon Bedrock Imported Model 셋업 15분, evaluation 실행)
- 실행 순서:
  1. [0_setup.ipynb](./notebook/0_setup.ipynb) 노트북을 실행하여 환경을 설정합니다.
  2. [1_data_preprocessing.ipynb](./notebook/1_data_preprocessing.ipynb) 노트북을 실행하여 3200개의 샘플 데이터를 전처리합니다.
  3. [2_fine_tuning.ipynb](./notebook/2_fine_tuning.ipynb) 노트북을 실행하여 모델을 파인 튜닝합니다.
  4. [3_deploy.ipynb](./notebook/3_deploy.ipynb) 노트북을 실행하여 모델을 Amazon Bedrock에 배포합니다.
  5. [4_evaluation.ipynb](./notebook/4_evaluation.ipynb) 노트북을 실행하여 모델 성능을 평가합니다.

### Option 2 - 프로덕션용 78577개 전체 데이터셋 이용

- 실행 소요 시간: 약 12시간 (데이터 전처리 3시간, 파인 튜닝 8시간, Amazon Bedrock Imported Model 셋업 15분, evaluation 실행)
- 실행 순서:
  1. [0_setup.ipynb](./notebook/0_setup.ipynb) 노트북을 실행하여 환경을 설정합니다.
  2. [1-1_full_data_preprocessing.ipynb](./notebook/1-1_full_data_preprocessing.ipynb) 노트북을 실행하여 78577개의 전체 데이터셋을 전처리합니다.
  3. [2_fine_tuning.ipynb](./notebook/2_fine_tuning.ipynb) 노트북을 실행하여 모델을 파인 튜닝합니다.
  4. [3_deploy.ipynb](./notebook/3_deploy.ipynb) 노트북을 실행하여 모델을 Amazon Bedrock에 배포합니다.
  5. [4_evaluation.ipynb](./notebook/4_evaluation.ipynb) 노트북을 실행하여 모델 성능을 평가합니다.

### 실습 비용

> ⚠️ 주의: 실제 비용은 **사용 시간**과 **리전**에 따라 달라질 수 있습니다. 불필요한 비용 발생을 방지하기 위해 사용하지 않는 리소스는 반드시 정지 또는 삭제해주세요.

- Amazon SageMaker Notebook Instance (ml.t3.medium): 시간 당 약 $0.05 (us-east-1, 2024/10 기준)
- Amazon SageMaker Training Job (ml.g5.2xlarge): 시간 당 약 $1.515 (us-east-1, 2024/10 기준)
- Amazon Bedrock Imported Model: 요금은 모델 크기 및 사용량에 따라 다릅니다. 자세한 내용은 [Amazon Bedrock 요금 페이지](https://aws.amazon.com/ko/bedrock/pricing/)를 참조하세요.
- Amazon Translate (Batch job) : 1백만 자당 $15.00 (us-east-1, 2024/10 기준)

#### 비용 참고

- **Option 1** (3,200개 샘플 데이터셋): $2.871 + Amazon Bedrock Claude Sonnet/Imported Model 호출 비용
- **Option 2** (78,577개 전체 데이터셋): $59.72 + Amazon Bedrock Claude Sonnet/Imported Model 호출 비용

## 🗂️ 디렉토리 구조

```text

.
├── README.md
├── datasets        # 전처리가 완료된 전체 데이터셋 (78577개)
│   ├── ko_test_dataset.json
│   ├── ko_train_dataset.json
│   └── ko_validation_dataset.json
├── images          # 실습 가이드용 이미지 셋
│
├── notebook        # 실습 수행용 파이썬 노트북
│   ├── 0_setup.ipynb
│   ├── 1-1_full_data_preprocessing.ipynb # 프로덕션 용 전체 데이터 셋 전처리 (78577개)
│   ├── 1_data_preprocessing.ipynb        # 실습용 샘플 데이터 셋 전처리 (3200개)
│   ├── 2_fine_tuning.ipynb
│   ├── 3_deploy.ipynb
│   └── 4_evaluation.ipynb
└── scripts         # 파인 튜닝에 이용하는 스크립트
    ├── requirements.txt
    └── run_fsdp_qlora_llama3.py

```

프로젝트의 주요 구성 요소는 다음과 같습니다:

- [0_setup.ipynb](./notebook/0_setup.ipynb): 초기 환경 설정 과정
- [1_data_preprocessing.ipynb](./notebook/1_data_preprocessing.ipynb): 데이터 전처리 과정 (실습용 샘플 데이터 이용, 프로덕션 용 전체 데이터 셋은 [1-1_full_data_preprocessing.ipynb](./notebook/1-1_full_data_preprocessing.ipynb) 노트북 이용)
- [2_fine_tuning.ipynb](./notebook/2_fine_tuning.ipynb): 모델 파인 튜닝 과정
- [3_deploy.ipynb](./notebook/3_deploy.ipynb): Amazon Bedrock에 모델 배포 과정
- [4_evaluation.ipynb](./notebook/4_evaluation.ipynb): 모델 평가 과정

## 📝 References

- [SageMaker 에서 Llama3.1 8B 파인 튜닝, 모델 배포 및 추론 하기](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/30_fine_tune/03-fine-tune-llama3/llama3-1)
- [Amazon Bedrock Samples](https://github.com/aws-samples/amazon-bedrock-samples/blob/main/custom_models/import_models/llama-3/customized-text-to-sql-model.ipynb)
- [Import a fine-tuned Meta Llama 3 model for SQL query generation on Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/import-a-fine-tuned-meta-llama-3-model-for-sql-query-generation-on-amazon-bedrock/)
