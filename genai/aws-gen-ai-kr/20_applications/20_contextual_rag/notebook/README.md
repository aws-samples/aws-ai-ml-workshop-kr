# Contextual RAG System

이 프로젝트는 Amazon Bedrock을 활용한 Contextual RAG(Retrieval Augmented Generation) 시스템입니다.

## 주요 기능

- Amazon Bedrock Foundation Model을 활용한 텍스트 생성
- 임베딩 기반 유사도 검색(KNN) 및 키워드 기반 검색(BM25) 하이브리드 검색
- Rank Fusion과 Cross-encoder Reranking을 통한 검색 결과 최적화

## 시스템 구성

- `BedrockService`: AWS Bedrock 서비스와의 상호작용 담당
- `RerankerService`: 검색 결과 재순위화 처리
- `Config`: 환경 설정 관리

## 설치 및 설정

1. 필요한 환경 변수 설정 (.env 파일)
```
# AWS Configuration
AWS_REGION=
AWS_PROFILE=

# Bedrock Configuration  
BEDROCK_MODEL_ID=
BEDROCK_RETRIES=
EMBED_MODEL_ID=

# OpenSearch Configuration
OPENSEARCH_PREFIX=
OPENSEARCH_DOMAIN_NAME=
OPENSEARCH_DOCUMENT_NAME=
OPENSEARCH_USER=
OPENSEARCH_PASSWORD=

# Reranker Configuration
RERANKER_AWS_REGION=
RERANKER_AWS_PROFILE=
RERANKER_MODEL_ID=

# Rank Fusion Configuration
RERANK_TOP_K=
HYBRID_SCORE_FILTER=
FINAL_RERANKED_RESULTS=
KNN_WEIGHT=

# Application Configuration
RATE_LIMIT_DELAY: API 요청 간 지연 시간(초) (기본값: 60)

```
2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

