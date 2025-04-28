# Agentic Text-to-SQL Workflow with AWS-hosted Langfuse and Amazon Bedrock


## 1. 선수 사항
### 1.1. 리소스 설정을 위한 CloudFormation 템플릿 실행
- TBD
### 1.2. 오픈 서치에 노리 플러그인 설치
- TBD
### 1.3. Langfuse 구성


## 2. 실습
### 2.1. 실습: Text-to-SQL 스키마 준비
이 실습은 Text-to-SQL 애플리케이션을 위한 스키마 문서 준비를 포함합니다.
[schema_prep](./images/text2sql/schema-prep-1.png)

#### 파일
- `lab1_text2sql_schema_preparation/`
  - `1.sample_queries.ipynb`: 샘플 쿼리 문서를 준비하기 위한 주피터 노트북.
  - `2.detailed_schema.ipynb`: 상세 스키마 문서를 준비하기 위한 주피터 노트북.

### 2.2. 실습: LangGraph를 사용한 워크플로 구성
이 실습에서는 LangGraph를 사용하여 순환 워크플로를 구축합니다.
[langgraph](./images/text2sql/langgraph.png)

#### 파일
- `lab2_text2sql_langgraph/`
  - `1.langfuse_text2sql_langgraph.ipynb`: LangGraph를 사용하여 Text-to-SQL 워크플로를 개발하기 위한 주피터 노트북.


## A. 참조
- 이 리포는 [여기](https://github.com/kevmyung/text-to-sql-bedrock)의 코드를 참조하여 작성했습니다.
