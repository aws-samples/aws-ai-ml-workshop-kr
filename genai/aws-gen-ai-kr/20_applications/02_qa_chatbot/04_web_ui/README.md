# How to run this application

## Structure

1. `bedrock.py`

- Amazon Bedrock 및 Reranker, Hybrid search, parent-document 등의 RAG 기술 구현 파일

2. `streamlit.py`

- 애플리케이션의 front-end 파일, 실행 시 `bedrock.py`을 import해서 사용

## Start

1. Python 종속 라이브러리 설치

```
pip install -r requirements.txt
```

2. Streamlit 애플리케이션 작동

```
streamlit run streamlit.py --server.port<MY_PORTNUMBER>
```

3. 접속하기

- Streamlit 작동 시 표시되는 External link로 접속
