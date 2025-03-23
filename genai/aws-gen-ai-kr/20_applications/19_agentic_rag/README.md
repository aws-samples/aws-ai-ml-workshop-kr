# AWS Agentic AI Workshop: 실전 응용을 위한 Agentic RAG 배우기

<p align="left">
    <a href="https://github.com/aws-samples">
            <img alt="Build" src="https://img.shields.io/badge/Contribution-Welcome-blue">
    </a>
    <a href="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Faws-samples%2Faws-ai-ml-workshop-kr%2Ftree%2Fmaster%2Fgenai%2Faws-gen-ai-kr%2F20_applications%2F19_agentic_rag&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
</p>

Last Updated: Mar 23, 2025


이 워크샵은 AWS의 생성형 AI 서비스와 기술을 활용한 엔드투엔드 개발 과정을 안내합니다. 필요한 개념 및 코드를 단계별 학습 경로로 제공하여, 기본 RAG부터 에이전틱 AI 시스템까지 구축하는 방법을 제공합니다.

시작은 기본 환경 설정으로 시작하여, 
- 프롬프트 엔지니어링의 Chain-of-Thought(CoT) 및 Chain-of-Density(CoD) 기법을 통해 LLM의 추론 능력을 향상시키는 방법을 배웁니다. 
- 검색 기능의 핵심인 OpenSearch 기본 사항 및 여러가지 테크닉을 배웁니다. 
- 맥락적 데이터 전처리 기법을 통해 입력 데이터의 품질을 개선하는 방법을 익힙니다.
- LangGraph, Bedrock 및 Langfuse를 활용한 에이전틱 워크플로우 개발의 시작을 위한 기본 패턴을 배웁니다. 
- 고급 RAG 기법을 통해 LLM의 정확성과 관련성을 향상시키는 방법을 배웁니다.
- RAG 평가 프레임워크를 통해 시스템의 성능을 객관적으로 측정하는 것을 배웁니다. 
- Self-RAG와 같은 에이전틱 검색 기술을 적용하여 LLM이 스스로 검색 과정을 계획하고 최적화하는 방법을 학습합니다.

<br>
이 워크샵은 AWS 생성형 AI 서비스를 활용하여 단순한 정보 검색을 넘어, 목표 지향적이고 자율적인 에이전틱 RAG 시스템을 설계, 구현, 평가하는 전체 과정을 제공합니다.

---

## 1. 워크샵 환경 준비 
이 워크샵은 [SageMaker AI Studio Jupyterlab](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/studio-updated-jl.html) 및 [SageMaker AI Studio Code Editor](https://docs.aws.amazon.com/sagemaker/latest/dg/code-editor.html)에서 테스트 완료 되었습니다.
아래 링크를 클릭해서 워크샵 환경 준비를 해주세요.
- [워크샵 가상 환경 준비 ](01_setup/README.md)
- Langfuse 설치 혹은 사용
    - 관지자 용
        - [Langfuse 설치 ](01_setup/LAGNFUSE_ADMIN_README.md)
    - 일반 사용자 용 : <-- 위에 관리자가 Langfuse 설치 되었을시에 가능합니다.   
        - [Langfuse 사용 ](01_setup/LAGNFUSE_END_USER_README.md)

## 2. 워크샵 실행
### 2.1 LLM 프롬프팅 기법의 진화: Chain-of-Thought에서 Chain-of-Draft까지 
- [02_prompt_engineering_cot_cod](02_prompt_engineering_cot_cod/README.md)
### 2.2 OpenSearch의 기본과 활용
- [03_opensearch_basic](03_opensearch_basic/README.md)
### 2.3 RAG 성능 향상을 위한 데이터 전처리와 Contextual Retrieval 실습
- [04_preprocessing_contextual_data](04_preprocessing_contextual_data/README.md)
### 2.4 LangGraph와 Amazon Bedrock 최적의 통합: Agentic AI 워크플로우 패턴 실습과 Langfuse 모니터링 시작하기
- [05_start_agentic_workflow_langgraph_bedrock_langfuse](05_start_agentic_workflow_langgraph_bedrock_langfuse/README.md)
### 2.5 Advanced RAG로 환각 줄여보기
- [06_advanced_rag](06_advanced_rag/README.md)
### 2.6 RAG 시스템을 체계적으로 평가하는 방법: Amazon Bedrock과 RAGAS 메트릭스를 활용한 검증 프레임워크
- [07_rag_evaluation_framework](07_rag_evaluation_framework/README.md)
### 2.7 생성형 AI 새로운 패러다임: Agent AI 
- [08_self_rag](08_self_rag/README.md)


