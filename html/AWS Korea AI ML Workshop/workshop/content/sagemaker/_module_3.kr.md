+++
title = "Module 3: Linear Learner를 사용해 시계열 예측하기"
menuTitle = "Linear Learner를 사용해 시계열 예측하기"
date = 2019-10-15T15:17:48+09:00
weight = 203
+++

SageMaker의 Jupyter 노트북 페이지 상단의 탭메뉴에서 “SageMaker Examples”를 클릭 후 샘플 노트북 목록에서 `Introduction to Applying Machine Learning` 을 선택합니다.

![select_linear_time_series_forecast](/images/sagemaker/module_3/select_linear_time_series_forecast.png?classes=border)

샘플 목록중 `linear_time_series_forecast.ipynb` 를 찾아 우측의 `Use` 버튼을 클릭합니다. 다음과 같은 팝업창에서 `Create copy` 버튼을 클릭하여 관련 파일들을 사용자의 홈디렉토리로 복사를 진행합니다.

새로운 브라우저 탭에서 노트북이 오픈되면 준비가 완료됩니다.

![time_series_forecasting_with_linear_learner](/images/sagemaker/module_3/time_series_forecasting_with_linear_learner.png?classes=border)

모듈 실행중 아래 코드를 만나면 `<your_s3_bucket_name_here>` 부분에 모듈 1에서 생성한 s3 버킷의 이름(예: sagemaker-xxxxx)을 넣고 실행합니다. 부등호 부호(‘<’, ’>’)는 넣지 않습니다.

![s3_bucket_name](/images/sagemaker/module_3/s3_bucket_name.png?classes=border)

이 노트북의 소스 파일은 {{% button href="https://github.com/awslabs/amazon-sagemaker-examples/tree/master/introduction_to_applying_machine_learning/linear_time_series_forecast" icon="fab fa-github" %}}Github{{% /button %}} 에 공개되어 있습니다.

※ 이 모델을 훈련하는데는 약 10분에서 15 분이 소요됩니다. 