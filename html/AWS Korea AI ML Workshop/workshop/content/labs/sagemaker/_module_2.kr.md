+++
title = "Module 2: Linear Learner MNIST"
menuTitle = "Linear Learner MNIST"
date = 2019-10-15T15:17:43+09:00
weight = 202
+++

SageMaker의 Jupyter 노트북 페이지 상단의 탭메뉴에서 “SageMaker Examples”를 클릭 후 샘플 노트북 목록에서 `Introduction to Amazon Algorithms` 을 선택합니다.

![sagemaker_select_amazon_algorithms](/images/sagemaker/module_2/sagemaker_select_amazon_algorithms.png?classes=border)

샘플 목록중 `linear_learner_mnist.ipynb` 를 찾아 우측의 `Use`  버튼을 클릭합니다. 다음과 같은 팝업창에서 `Create copy` 버튼을 클릭하여 관련 파일들을 사용자의 홈디렉토리로 복사를 진행합니다.

![create_a_copy_in_your_home_directory](/images/sagemaker/module_2/create_a_copy_in_your_home_directory.png?classes=border)

새로운 브라우저 탭에서 노트북이 오픈되면 준비가 완료됩니다.

![linear_learner_mnist](/images/sagemaker/module_2/linear_learner_mnist.png?classes=border)

모듈 실행중 아래 코드를 만나면 `<your_s3_bucket_name_here>` 부분에 모듈 1에서 생성한 s3 버킷의 이름(예: sagemaker-xxxxx)을 넣고 실행합니다. 부등호 부호(‘<’, ’>’)는 넣지 않습니다.

![s3_bucket_name](/images/sagemaker/module_2/s3_bucket_name.png?classes=border)

이 노트북의 소스 파일은 {{% button href="https://github.com/awslabs/amazon-sagemaker-examples/tree/master/introduction_to_amazon_algorithms/linear_learner_mnist" icon="fab fa-github" %}}Github{{% /button %}} 에 공개되어 있습니다.

※ 이 모델을 훈련하는데는 약 10분에서 15 분이 소요됩니다. 