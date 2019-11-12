# BlazingText 한국어 학습 Hands-on-Lab

## Introduction

Amazon SageMaker BlazingText 알고리즘은 Word2vec 기반 임베딩과 FastText 기반 텍스트 분류 알고리즘을 최적화하여 구현하였습니다. BlazingText는 멀티 코어 CPU 또는 GPU를 사용하여 몇 분 내에 10억 개 이상의 단어를 훈련할 수 있으며, 문자 단위(character) n-gram에 대한 벡터 표현을 학습하여 미등록 단어에 대한 벡터(OOV; out-of-vocabulary)의 표현이 가능합니다.

이 워크샵을 통해 여러분은 BlazingText를 활용하여 한국어 위키피디아의 단어 임베딩(Word Embedding), 네이버 영화 리뷰의 긍정/부정 분류를 수행하는 방법, 호스팅된 SageMaker Endpoint를 웹서비스에 활용하는 방법, 그리고 사전 학습된 모델을 SageMaker Endpoint로 호스팅하는 방법을 배울 수 있습니다.

워크샵은 아래 5개의 모듈로 이루어져 있습니다. 모듈 1은 필수로 수행하셔야 하며,
모듈 2-5는 순서대로 하실 필요는 없지만 이해를 돕기 위해 순서대로 하는 것을 권장드립니다. 
(단, 모듈 4를 진행하기 위해서는 모듈 3을 먼저 진행해야 합니다.)

1. [사전 준비 (필수)](get_started.md)
2. [BlazingText를 활용한 한국어 위키피디아 Word2Vec 임베딩](blazingtext_word2vec_korean.ipynb)
3. [BlazingText를 활용한 네이버 영화 리뷰 감성(Sentiment) 이진 분류](blazingtext_text_classification_korean.ipynb)
4. [AWS Lambda와 AWS API Gateway로 영화 리뷰 긍정/부정 분류 웹서비스 생성하기](blazingtext_endpoint_api_gateway.md)
5. [사전 학습된 FastText를 BlazingText에 호스팅하기](blazingtext_hosting_pretrained_fasttext_korean.ipynb)


## License Summary

이 샘플 코드는 MIT-0 라이센스에 따라 제공됩니다. LICENSE 파일을 참조하십시오.