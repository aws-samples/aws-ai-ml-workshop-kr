# Lab6: 6_sm_eventbridge
- EventBridge를 활용하여 CodeBuild config 파일을 레포지토리에 push 하면 code pipeline 을 통해 SageMaker pipeline 실행 (Training).
- EventBridge를 활용하여 Model Approval 하면 code pipeline 을 통해 SageMaker pipeline 실행 (Serving).

# 1. 실습 파일 

- 1.0.Prepare-New-Dataset.ipynb
    - SageMaker 훈련 Pipeline 실행을 위한 신규 훈련 데이터 준비
- 1.1.Create_eventbridge_for_training_codepipeline.ipynb
    - Repository Push 이벤트를 트리거하는 EventBridge Rule 및 Target 생성
- 2.1.Create_eventbridge_for_serving_codepipeline.ipynb
    - Repository Push 이벤트를 트리거하는 EventBridge Rule 및 Target 생성
