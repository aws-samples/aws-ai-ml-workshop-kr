+++
title = "실습 가이드"
date = 2019-10-23T15:44:53+09:00
weight = 311
+++

### Lab 개요 

Amazon SageMaker는 데이터 사이언티스트와 개발자들이 쉽고 빠르게 구성,
학습하고 어떤 규모로든 기계 학습된 모델을 배포할 수 있도록 해주는 관리형
서비스 입니다. 이 워크샵을 통해 SageMaker notebook instance를 생성하고
샘플 Jupyter notebook을 실습하면서 SageMaker의 일부 기능을 알아보도록
합니다.

### 목표

-   SageMaker에 내장된 학습 기능을 사용하여 모델 훈련 Job을 생성 합니다.

-   SageMaker의 endpoint 기능을 사용하여 생성된 모델이 예측에 사용될 수
    있도록 endpoint를 생성합니다.

-   머신 러닝이 정형 데이터(e.g. CSV 파일)와 비정형 데이터(e.g.
    이미지)에 모두 적용 될수 있음을 확인 합니다.

### 준비 조건 

-   AWS 계정: AWS IAM, S3, SageMaker 자원을 생성할 수 있는 권한이
    필요합니다.

-   AWS Region: SageMaker는 지원되는 region은
    <https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/>
    에서 확인하실 수 있습니다. 이번 실습은Seoul (ap-northeast-2)
    region에서 실행 합니다.

-   Browser: 최신 버전의 Chrome, Firefox를 사용하세요.

**※ 주의 사항:** Notebook 안의 Cell에서 코드 실행후 결과 값이 나오는
데는 수 초가 걸립니다. 훈련 Job을 실행하는 경우 수 분이 걸릴 수도
있습니다. 실습 완료 후에는 아래 가이드에 따라 생성된 자원을 꼭
종료/삭제해 주세요.