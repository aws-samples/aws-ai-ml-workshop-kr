# Amazon Bedrock Agent Workshop: Advanced Stock Analysis & AICC

> **안내**: 이 저장소는 Amazon Bedrock 워크샵의 실습 코드만을 포함하고 있습니다. 
> 실제 워크샵 수행과 상세한 가이드를 위해서는 아래 링크의 전체 워크샵 페이지를 참조해 주세요.
> 
> [Amazon Bedrock Workshop - Advanced](https://catalog.us-east-1.prod.workshops.aws/workshops/86f59566-0ae7-44be-80ab-9044b83c88f2)

### 소개
이 워크샵은 Amazon Bedrock의 기본 기능과 패턴을 실습하고, 프롬프트 엔지니어링을 통해 실제 사례(AICC)를 학습하며, Bedrock Agent를 활용한 어플리케이션 구축을 목표로 합니다.
이번 워크샵을 통해 기초부터 고급까지 Amazon Bedrock을 활용한 다양한 기술들을 습득하게 될 것입니다.

### 대상
이 워크샵은 생성형 AI 및 기계 학습에 관심이 있는 엔지니어, 데이터 과학자, 개발자들을 대상으로 합니다.
기초 지식이 있는 분들에게 적합하며, 고급 사용자를 위한 심화 내용도 포함되어 있습니다.

### 목표

![Architecture](/dataset/images/workshop_overview.ko.png)

1. **[Bedrock 기능 및 패턴](bedrock_basic_workshop/README.md)** : Bedrock의 기본 기능을 이해하고 간단한 애플리케이션을 실습합니다.
2. **[프롬프트 엔지니어링](aicc_prompting_workshop/README.md)** : 자동차 보험 센터(AICC)를 기준으로 프롬프트 엔지니어링 방법을 배우고 실습합니다.
3. **[Bedrock Agent 어플리케이션](stock_agent_workshop/README.md)** : Bedrock Agent를 활용하여 주식 분석 어플리케이션을 구축합니다.

### 수행 시간

워크샵을 수행하는데 총 6시간이 소요됩니다.

- 환경 설정 : 45 min
- Bedrock 기능 및 패턴 : 1 hr ~ 1hr 30 min
- 프롬프트 엔지니어링 : 1 hr 30 min
- Bedrock Agent 어플리케이션 : 2 hr
- 리소스 정리 : 15 min

### 주요 기술
- [Amazon Bedrock](https://aws.amazon.com/bedrock/) : 파운데이션 모델을 사용하기 위한 완전 관리형 서비스입니다. 이를 통해 텍스트 생성 및 이미지 생성을 위한 단일 API 세트를 사용하여 Amazon 및 타사의 모델에 액세스할 수 있습니다.
- [Streamlit](https://streamlit.io/) : 프론트엔드 개발 기술 없이도 파이썬 코드의 웹 프론트엔드를 빠르게 만들 수 있습니다. Streamlit은 기술자와 비기술자 모두에게 보여줄 수 있는 개념 증명(PoC)을 만드는 데 유용합니다.

### 사용 리전
- us-west-2

---

# Amazon Bedrock Workshop - Advanced

> **Note**: This repository contains only the practice code for the Amazon Bedrock workshop.
> For the actual workshop execution and detailed guide, please refer to the full workshop page at the link below.
> 
> [Amazon Bedrock Workshop - Advanced](https://catalog.us-east-1.prod.workshops.aws/workshops/86f59566-0ae7-44be-80ab-9044b83c88f2)

### Introduction
This workshop aims to practice the basic functions and patterns of Amazon Bedrock, learn real-world cases (AICC) through prompt engineering, and build applications using Bedrock Agent.
Through this workshop, you will acquire various skills utilizing Amazon Bedrock, from basics to advanced levels.

### Target Audience
This workshop is designed for engineers, data scientists, and developers interested in generative AI and machine learning.
It is suitable for those with basic knowledge and includes in-depth content for advanced users.

### Objectives

![Architecture](/dataset/images/workshop_overview.en.png)

1. **[Bedrock Features and Patterns](bedrock_basic_workshop/README.md)**: Understand the basic functions of Bedrock and practice simple applications.
2. **[Prompt Engineering](aicc_prompting_workshop/README.md)**: Learn and practice prompt engineering methods based on the Automotive Insurance Contact Center (AICC).
3. **[Bedrock Agent Application](stock_agent_workshop/README.md)**: Build a stock analysis application using Bedrock Agent.

### Duration

The workshop takes a total of 6 hours to complete.

- Environment Setup: 45 min
- Bedrock Features and Patterns: 1 hr ~ 1hr 30 min
- Prompt Engineering: 1 hr 30 min
- Bedrock Agent Application: 2 hr
- Resource Cleanup: 15 min

### Key Technologies
- [Amazon Bedrock](https://aws.amazon.com/bedrock/): A fully managed service for using foundation models. It allows access to models from Amazon and third parties using a single API set for text generation and image generation.
- [Streamlit](https://streamlit.io/): Quickly create web frontends for Python code without frontend development skills. Streamlit is useful for creating proofs of concept (PoC) that can be shown to both technical and non-technical audiences.

### Region Used
- us-west-2
