# SageMaker 에서 DeepSpeed 사용하기 

SageMaker 에서 DeepSpeed 를 통한 분산 훈련을 할 수 있습니다. 
이 워크샵은 버트 모델을 네이버 영화 리뷰 데이터 세트로 학습을 통해서, SageMaker 에서 DeepSpeed 의 사용 방법을 배울 수 있습니다. 

이 워크샵은 
아래와 같은 단계로 실제로 실습 할 수 있습니다.

## Step 1:
아래 설치 하기를 클릭하시고, 가이드를 따라가 주세요.
- [setup/README.md](setup/README.md)

## Step 2: 
notebook/01-sm-deepspeed-training.ipynb 을 열고 실행 하시면 됩니다.
- 아래와 같이 3가지로 테스트를 할 수 있습니다.
    - 현재의 노트북 인스턴스에서 모델 학습하기
    - SageMaker 의 로컬 모드로 학습하기 ( 3. SageMaker Training 준비에서 USE_LOCAL_MODE = True 로 수정)
    - SageMaker 의 클라우드 모드로 학습하기 ( 3. SageMaker Training 준비에서 USE_LOCAL_MODE = False 로 수정)

