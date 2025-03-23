# LangGraph 와 Amazon Bedrock 으로  Agentic Workflows and Agents 시작하기

이 워크샵에서는 LangGraph를 활용한 에이전틱 워크플로우와 에이전트 개발에 대해 다양한 측면을 배울 수 있습니다:

* **LangChain 기초와 Amazon Bedrock 연동** - LangChain 프레임워크의 기본 사용법과 Amazon Bedrock의 강력한 언어 모델을 연동하는 방법을 학습합니다.

* **Langfuse를 통한 모니터링** - LLM 애플리케이션의 성능과 행동을 모니터링하고 추적하는 방법을 배웁니다. <u>특히 보안을 강화 하기 위해서, AWS Fargate 에 자체 LangFuse 서버를 구축해서 사용 합니다.</u> 

* **프롬프트 체이닝** - 여러 프롬프트를 순차적으로 연결하여 복잡한 작업을 단계별로 처리하는 방법을 익힙니다.

* **병렬 처리 워크플로우** - 여러 작업을 동시에 실행하고 그 결과를 결합하는 효율적인 워크플로우 설계 방법을 학습합니다.

* **조건부 라우팅** - LLM의 판단에 따라 워크플로우의 경로를 동적으로 결정하는 라우팅 메커니즘을 구현합니다.

* **오케스트레이터-워커 패턴** - 복잡한 작업을 계획하고 실행하는 역할을 분리하여 효율적인 워크플로우를 구축하는 방법을 배웁니다.

* **평가자-최적화자 패턴** - Generator-Evaluater 패턴으로서 LLM 출력물의 품질을 평가하고 자동으로 개선하는 피드백 루프를 설계합니다.

* **도구 활용 에이전트 개발** - 단일 및 다중 도구와 연동하여 복잡한 작업을 수행할 수 있는 자율적인 에이전트 시스템을 구축합니다.

<br>
이 워크샵은 Amazon Bedrock 서비스를 기반으로 LangGraph 프레임워크를 활용해 복잡한 LLM 워크플로우를 설계하고 구현하는 실용적인 기술을 제공합니다.

---

## 1. 실습 가이드

#### 1.1 워크샵 환경 준비 
이 워크샵은 [SageMaker AI Studio Jupyterlab](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/studio-updated-jl.html) 및 [SageMaker AI Studio Code Editor](https://docs.aws.amazon.com/sagemaker/latest/dg/code-editor.html)에서 테스트 완료 되었습니다.
아래 링크를 클릭해서 워크샵 환경 준비를 해주세요.
- [워크샵 환경 준비 ](../01_setup/README.md)

## 2.워크샵 노트북
### 2.1. LangChain
- [01-get-started-langchain-bedrock.ipynb](01_langchain/01-get-started-langchain-bedrock.ipynb)

### 2.2. Langfuse
- LangChain 을 사용한 Langfuse 사용 방법
    - [01-get-started-langfuse-bedrock.ipynb](02_langfuse/01-get-started-langfuse-bedrock.ipynb)
- AWS boto3 converse API 을 사용한 Langfuse 사용 방법
    - [02-get-started-langfuse-boto3-bedrock.ipynb](02_langfuse/02-get-started-langfuse-boto3-bedrock.ipynb)

### 2.3. LangGraph
- [01-prompt-chaining-langgraph-bedrock.ipynb](03_langgraph/01-prompt-chaining-langgraph-bedrock.ipynb)
- [02-parallelization-langgraph-bedrock.ipynb](03_langgraph/02-parallelization-langgraph-bedrock.ipynb)
- [03-routing-langgraph-bedrock.ipynb](03_langgraph/03-routing-langgraph-bedrock.ipynb)
- [04-orchestrator-worker-langgraph-bedrock.ipynb](03_langgraph/04-orchestrator-worker-langgraph-bedrock.ipynb)
- [05-evaluator-optimizer-langgraph-bedroc.ipynb](03_langgraph/05-evaluator-optimizer-langgraph-bedroc.ipynb)
- [06-agent-single-tools-langgraph-bedroc.ipynb](03_langgraph/06-agent-single-tools-langgraph-bedroc.ipynb)
- [07-agent-multi-tools-langgraph-bedroc.ipynb](03_langgraph/07-agent-multi-tools-langgraph-bedroc.ipynb)


