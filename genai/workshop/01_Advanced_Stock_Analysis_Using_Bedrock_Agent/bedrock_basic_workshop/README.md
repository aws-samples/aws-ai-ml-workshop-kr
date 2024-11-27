# Amazon Bedrock 워크샵 - 기본 기능 및 패턴 실습

이 워크샵에서는 Amazon Bedrock의 주요 기능과 일반적인 사용 패턴을 실습합니다. 총 5가지 실습을 통해 Bedrock의 다양한 기능을 경험하고 실제 애플리케이션에 적용하는 방법을 학습합니다.

> **중요**: 이 저장소는 실습 코드만을 포함하고 있습니다. 실제 워크샵 수행과 상세한 가이드를 위해서는 아래 링크의 전체 워크샵 페이지를 참조해 주세요.
> 
> [Bedrock 기능 및 패턴](https://catalog.us-east-1.prod.workshops.aws/workshops/86f59566-0ae7-44be-80ab-9044b83c88f2/ko-KR/basic)

## 폴더 구조

- `practice`: 실습을 직접 수행할 수 있는 코드가 포함된 폴더입니다. 참가자들은 이 폴더의 파일들을 수정하며 실습을 진행합니다.
- `completed`: 각 실습의 완성된 코드가 포함된 폴더입니다. 참고용으로 사용할 수 있습니다.

## 실습 내용

1. **Converse API**: Bedrock의 기본적인 대화 기능을 사용하여 질문에 대한 응답을 생성하고, 스트리밍 응답 처리 방법을 학습합니다.

2. **Tool use**: Bedrock 모델이 외부 도구를 활용하여 정보를 검색하고 응답하는 방법을 실습합니다. 주식 가격 조회 기능을 예로 사용합니다.

3. **임베딩**: Titan Embeddings 모델을 사용하여 텍스트의 의미를 벡터로 변환하고, 텍스트 간 유사도를 계산하는 방법을 학습합니다.

4. **Chatbot with Tool use**: Streamlit을 사용하여 간단한 챗봇 인터페이스를 구현하고, Tool use 기능을 통합하여 대화형 주식 정보 조회 기능을 구현합니다.

5. **Text with RAG (Retrieval-Augmented Generation)**: PDF 문서에서 정보를 추출하고 벡터화하여 저장한 후, 사용자 질문에 대해 관련 정보를 검색하고 응답을 생성하는 RAG 시스템을 구현합니다.

이 실습들을 통해 참가자들은 Amazon Bedrock의 다양한 기능을 직접 경험하고, 실제 애플리케이션 개발에 적용할 수 있는 실용적인 지식을 얻게 됩니다.

각 실습의 상세한 단계와 설명은 위에 제공된 워크샵 링크에서 확인할 수 있습니다. 실습을 진행하실 때는 `practice` 폴더의 파일을 사용하시고, 필요한 경우 `completed` 폴더의 완성된 코드를 참고하실 수 있습니다.

---

# Amazon Bedrock Workshop - Basic Features and Patterns Practice

This workshop provides hands-on experience with the key features and common usage patterns of Amazon Bedrock. Through five practical exercises, participants will explore various functionalities of Bedrock and learn how to apply them in real-world applications.

> **Important**: This repository contains only the practice code. For the actual workshop execution and detailed guide, please refer to the full workshop page at the link below.
> 
> [Bedrock Features and Patterns](https://catalog.us-east-1.prod.workshops.aws/workshops/86f59566-0ae7-44be-80ab-9044b83c88f2/en-US/basic)

## Folder Structure

- `practice`: This folder contains the code for hands-on exercises. Participants will modify files in this folder during the workshop.
- `completed`: This folder contains the completed code for each exercise. It can be used for reference.

## Workshop Contents

1. **Converse API**: Learn to use Bedrock's basic conversation functionality to generate responses to questions and handle streaming responses.

2. **Tool use**: Practice how Bedrock models can utilize external tools to search for information and respond. We use stock price lookup as an example.

3. **Embeddings**: Learn how to use the Titan Embeddings model to convert text meanings into vectors and calculate similarity between texts.

4. **Chatbot with Tool use**: Implement a simple chatbot interface using Streamlit and integrate the Tool use functionality to create an interactive stock information query system.

5. **Text with RAG (Retrieval-Augmented Generation)**: Implement a RAG system that extracts information from PDF documents, vectorizes and stores it, then retrieves relevant information and generates responses to user questions.

Through these exercises, participants will gain hands-on experience with various features of Amazon Bedrock and acquire practical knowledge that can be applied to real application development.

Detailed steps and explanations for each exercise can be found in the workshop link provided above. When conducting the exercises, use the files in the `practice` folder, and if necessary, refer to the completed code in the `completed` folder.
