# Amazon SageMaker 실습 코드

본 폴더는 SageMaker를 다양한 기능을 실습할 수 있는 예제를 포함하고 잇습니다.

---
## 1. SageMaker 를 이용한 ML/DL 모델 개발과 추론

#### 1-1. 빌트인 알고리즘 활용하기

- [XGBoost 시작하기](xgboost/Readme.md) - SageMaker Built-in XGBoost 알고리즘을 마케팅응답을 예측하는 이진분류 문제에 적용해봅니다. [바로가기](xgboost/Readme.md)

#### 1-2. BYOS (Bring Your Own Script)

- [Tensorflow script mode 사용하기](byos-tensorflow/Readme.md) - SageMaker에서 제공하는 Tensorflow 컨테이너를 이용하여 보스톤지역의 집값을 예측하는 회귀모델을 만들고 활용해 봅니다. [바로가기](byos-tensorflow/Readme.md) 

#### 1-3. BYOC (Bring Your Own Container)

- [BYOC Scikit-learn](byoc/scikit_bring_your_own/scikit_bring_your_own.ipynb) - SageMaker 커스텀 컨테이너로 생성하는 방법을 이해할 수 있습니다. 예제코드는 Scikit-learn을 이용한 붓꽃 품종을 분류하는 간단한 모델을 이용합니다.[바로가기](byoc/scikit_bring_your_own/scikit_bring_your_own.ipynb)

#### 1-4. BYOM (Bring Your Own Model)

- [Tensorflow deployment](tf-deploy/README.md) - Tensorflow Serving 실습 [바로가기](tf-deploy/README.md)

#### 1-5. SageMaker Ground Truth

- [Hello GroundTruth](hello-gt/README.md) - SageMaker GroundTruth 시작하기 [바로가기](hello-gt/README.md)

#### 1-6. SageMaker Data Wrangler

---

## 2. SageMaker 고급 기능 활용하기

#### 2-1. SageMaker Debugger

#### 2-2. SageMaker Distributed Training
- [Amazon SageMaker Distributed Training (Image Classification for Oxford-IIIT Pet Dataset)](https://github.com/aws-samples/sagemaker-distributed-training-pytorch-kr) 
- [End-to-end ML Image Classification (Bengali.AI Handwritten Grapheme Classification)](https://github.com/daekeun-ml/end-to-end-pytorch-on-sagemaker)
- [Amazon SageMaker Distributed Training Hands-on Lab - TensorFlow 2.x](https://github.com/daekeun-ml/sagemaker-distributed-training-tf2)

#### 2-3. SageMaker Clarify

#### 2-4. SageMaker Feature Store

---

## 3. SageMaker MLOps 적용하기

#### 3-1. SageMaker Pipeline

#### 3-2. SageMaker Project
- [SageMaker Pipeline](sm-pipeline/README.md) - SageMaker Pipeline & Project 실습 [바로가기](sm-pipeline/README.md)

#### 3-3. SageMaker Model monitor

- [SageMaker Model Monitor](model-monitor/SageMaker-ModelMonitoring.ipynb) SageMaker Model Monitor 기능 체험 [바로가기](model-monitor/SageMaker-ModelMonitoring.ipynb)

---
## 4. SageMaker 보안 & 거버넌스

#### 4-1. SageMaker ABAC

#### 4-2. Sagemaker Multi account deployment

---
## 5. SageMaker를 이용한 머신러닝/딥러닝 문제 해결

#### 5-1. SageMaker Canvas (No code 머신러닝)
- [SageMaker Canvas 공식 실습가이드(영문)](https://catalog.us-east-1.prod.workshops.aws/workshops/80ba0ea5-7cf9-4b8c-9d3f-1cd988b6c071/en-US/)
- [AWS Glue DataBrew와 SageMaker Canvas를 이용한 No code 머신러닝 모델 개발/적용](canvas-and-glue-databrew/Readme.md)

#### 5-2. AutoML
- [AutoGluon Hello World!](autogluon/autogluon_helloworld.ipynb) - 오픈소스 AutoGluon의 Getting Started 예제입니다. [바로가기](autogluon/autogluon_helloworld.ipynb)
- [Code Free Auto Gluon](autogluon/README.md) - 람다와 SageMaker 커스텀 컨테이너를 이용하여 AutoGluon 실행하기 [바로가기](autogluon/README.md)
- [AutoGluon on AWS](https://github.com/aws-samples/autogluon-on-aws) - 정형 데이터 외에 이미지, 텍스트, 멀티모달, 코드프리 등의 다양한 심화 예제들을 제공하고 있습니다.

#### 5-3. Computer Vision

#### 5-4. NLP

- [Korean NLP Hands-on labs)](https://github.com/aws-samples/sm-kornlp) - Amazon SageMaker 기반 한국어 자연어 처리 샘플 (Multiclass Classification, Named Entity Recognition, Question Answering, Chatbot and Semantic Search using Sentence-BERT, Natural Language Inference, Summarization, Translation, TrOCR 등)
    - [Multiclass Classification](https://github.com/aws-samples/sm-kornlp/tree/main/multiclass-classification)
    - [Named Entity Recognition (NER)](https://github.com/aws-samples/sm-kornlp/tree/main/named-entity-recognition)
    - [Question Answering](https://github.com/aws-samples/sm-kornlp/tree/main/question-answering)
    - [Chatbot and Semantic Search using Sentence-BERT (SBERT)](https://github.com/aws-samples/sm-kornlp/tree/main/sentence-bert-finetuning)
    - [Natural Language Inference (NLI)](https://github.com/aws-samples/sm-kornlp/tree/main/natural-language-inference)
    - [Summarization](https://github.com/aws-samples/sm-kornlp/tree/main/summarization)
    - [Translation](https://github.com/aws-samples/sm-kornlp/tree/main/translation)
    - [TrOCR](https://github.com/aws-samples/sm-kornlp/tree/main/trocr)    

#### 5-5. Time-series
- [Time series on AWS Hands-on Lab](https://github.com/daekeun-ml/time-series-on-aws-hol)

#### 5-6. AIoT 
- [End-to-end AIoT w/ SageMaker and Greengrass 2.0 on NVIDIA Jetson Nano](https://github.com/aws-samples/aiot-e2e-sagemaker-greengrass-v2-nvidia-jetson)
- [AWS IoT Greengrass V2 for beginners (Korean)](https://catalog.us-east-1.prod.workshops.aws/workshops/0b21ceb7-2108-4a82-9e76-4c56d4b52db5)

#### 5-6. Business case별 문제해결



