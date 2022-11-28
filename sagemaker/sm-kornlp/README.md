# Korean NLP downstream tasks Hands-on Labs on Amazon SageMaker

## Introduction

### Hugging Face on Amazon SageMaker
텍스트 분석, 번역, 문장 요약, 엔티티 분류 등 자연어 처리의 AI/ML 적용 사례들이 매우 많지만, 여전히 많은 고객들이 기술적인 진입 장벽으로 도입에 어려움을 겪고 있습니다. 일부 분들은 이미 Hugging Face 트랜스포머 라이브러리와 SageMaker의 많은 예제 코드들로 AIML 적용을 가속화하고 있지만, 2021년 초까지는 SageMaker가 Hugging Face 전용 컨테이너를 네이티브하게 지원하지 않아 커스텀 스크립트와 커스텀 컨테이너를 따로 작성해야 하는 어려움이 있었습니다.

하지만, 최근 AWS는 Amazon SageMaker에서 Hugging Face의 트랜스포머 모델을 더 쉽게 훈련하고 배포할 수 있는 Hugging Face 딥러닝 훈련 컨테이너 및 추론 컨테이너를 도입했습니다. 따라서, 인프라 설정에 대한 고민 없이 몇 줄의 코드만으로(Low-code) 빠르게 자연어 처리 모델의 훈련 및 프로덕션 배포가 가능하게 되었습니다.

### Korean Language Support
기술 성숙도가 낮거나 자연어 처리 모델을 검토하기 시작하는 분들은 한국어 핸즈온을 많이 찾습니다. 하지만, 한국어에 대한 핸즈온이 체계적으로 정리되어 있지 않아 도입에 많은 어려움을 겪는 것을 보아 왔고, 이에 따라 초심자들도 쉽게 시작할 수 있는 한국어 핸즈온을 개발하였습니다.

## Hands-on Labs 
각 핸즈온은 독립적으로 실행 가능합니다. 워크샵 진행 시 유즈케이스에 따라 자유롭게 핸즈온을 수행하시면 됩니다.

### [Multiclass Classification](multiclass-classification)

### [Named Entity Recognition (NER)](named-entity-recognition)

### [Question Answering](question-answering)

### [Chatbot and Semantic Search using Sentence-BERT (SBERT)](sentence-bert-finetuning)

### [Natural Language Inference (NLI)](natural-language-inference)

### [Summarization](summarization)

### [Translation](translation)

### [TrOCR](trocr)

<br>

## References
더 자세한 내용은 [REFERENCES](REFERENCES.md)를 참조하세요.