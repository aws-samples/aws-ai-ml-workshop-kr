# Amazon Bedrock with Amazon OpenSearch Service for RAG

이 섹션은 '한화생명'에서 Amazon OpenSearch Service 서비스를 Vector DB로 Amazon Bedrock에서 Claude 모델을 활용하여 Retrieval Augmented Generation (RAG)를 수행한 사례를 보여주고 있습니다. 

## 주요 프로세스 처리 단계
 1) 고객사의 PDF 문서를 데이터로 처리하여 메모리에 로딩한 다음 이를 OpenSearch index에 추가합니다.
 2) Langchain을 이용하여 Lexical Search와 Semantic Search를 각각 수행합니다.
 3) 2개의 Search를 통합해서 Hybrid Search를 수행합니다.

## 노트북 소개
Dataset: 한화생명 보험약관 PDF

- `00_setup.ipynb`: OpenSearch 클러스터 생성 및 노리(nori) 한글 형태소 분석기 연동
- `01_bedrock_fundamental.ipynb`: Bedrock 활용과 API를 호출하는 방식
- `02_rag_opensearch_kr.ipynb`: Amazon Bedrock과 OpenSearch를 이용한 기본적인 RAG 구현

## 참고 페이지
원본 워크샵에 더 많은 실습 자료가 있으며, 이를 workshop 용으로 변경하였습니다.
- https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/10_advanced_question_answering