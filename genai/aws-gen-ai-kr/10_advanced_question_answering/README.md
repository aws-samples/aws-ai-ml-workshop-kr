# Advanced Question & Answering

이 섹션에서는 기본적인 Question & Answering이 아닌 프로덕션 적용에 유용한 고급 내용을 소개합니다.

## 노트북 소개
Dataset: 신한은행 FAQ 데이터 세트

- `00_setup.ipynb`: OpenSearch 클러스터 생성 및 노리(nori) 한글 형태소 분석기 연동
- `01_rag_faiss_kr.ipynb`: FAISS 인메모리 벡터DB로 시멘틱 검색
- `02_1_rag_opensearch_lexical_kr.ipynb`: OpenSearch 키워드 검색
- `02_2_rag_opensearch_semantic_kr.ipynb`: OpenSearch 시멘틱 검색 (벡터 검색)
- `03_1_rag_opensearch_hybrid_kr.ipynb`: OpenSearch 하이브리드 검색
- `03_2_rag_opensearch_hybrid_ensemble_retriever_kr.ipynb`: Ensemble Retriever
- `04_rag_kendra_kr.ipynb`: Amazon Kendra 기반 RAG 
- `05_rag_qa_chatbot_hybrid_search.ipynb`: Conversational ChatBot
- `06_rag_faiss_with_images.ipynb`: FAISS 인메모리 벡터DB로 시멘틱 검색 후 답변에 연관된 이미지도 표시
