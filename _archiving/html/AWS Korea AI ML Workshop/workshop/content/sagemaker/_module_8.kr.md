+++
title = "Module 8: Bring-your-own-container 기능 실습하기"
menuTitle = "Bring-your-own-container 기능 실습하기"
date = 2019-10-15T15:18:03+09:00
weight = 208
+++

Amazon SageMaker는 머신 러닝 모델을 훈련하고 배포하기 위해 Docker container를 사용합니다. SageMaker에서 현재 지원하고 있지 않는 알고리즘이나 머신 러닝 프레임워크, 또는 여러분이 로컬 환경에서 개발한 모델이라도 Docker container로 만들어 SageMaker에서 훈련하고 배포할 수 있습니다.

이번 모듈에서는 Scikit-Learn으로 작성된 모델을 컨테이너로 패키징 해봅니다. 이 내용은 AWS 블로그 [Train and host Scikit-Learn models in Amazon SageMaker by building a Scikit Docker container](https://aws.amazon.com/blogs/machine-learning/train-and-host-scikit-learn-models-in-amazon-sagemaker-by-building-a-scikit-docker-container/) 에 잘 설명되어 있습니다.

참고로 현재 SageMaker는 [pre-built scikit-Learn 컨테이너](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/scikit_learn_iris/Scikit-learn%20Estimator%20Example%20With%20Batch%20Transform.ipynb)를 제공하고 있기 때문에, Scikit-Learn 모델을 사용하기 위해 매번 이 모듈의 내용대로 새로운 컨테이너를 만들 필요는 없습니다. Scikit-Learn을 사용하는 예제에 대해서는 [본 Github 예제](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk/scikit_learn_iris) 를 참고하시기 바랍니다.

SageMaker의 Jupyter 노트북 페이지 상단의 탭메뉴에서 `SageMaker Examples` 를 클릭합니다.

![select_sagemaker_examples](/images/sagemaker/module_5/select_sagemaker_examples.png?classes=border)

샘플 노트북 목록에서 `Advanced Functionality` 을 선택합니다.

![bring_your_own_container](/images/sagemaker/module_8/bring_your_own_container.png?classes=border)

샘플 목록중 `scikit_bring_your_own.ipynb` 를 찾아 우측의 `Use` 버튼을 클릭합니다. 다음과 같은 팝업창에서 `Create copy` 버튼을 클릭하여 관련 파일들을 사용자의 홈디렉토리로 복사를 진행합니다.

![create_a_copy_in_your_home_directory](/images/sagemaker/module_8/create_a_copy_in_your_home_directory.png?classes=border)

새로운 브라우저 탭에서 노트북이 오픈되면 준비가 완료됩니다.

![building_your_own_algorithm_container](/images/sagemaker/module_8/building_your_own_algorithm_container.png?classes=border)

모듈 실행시 사용할 S3 bucket을 default로 생성한 것이 아닌 미리 생성된 버켓을 사용하실때에는 소스 코드 중 `sess.default_bucket()` 부분을 (현재 3군데에 사용이 되고 있습니다). 모두 본인의 S3 bucket으로 치환하시면 됩니다.

![put_s3_bucket_name](/images/sagemaker/module_3/default_s3_bucket.png?classes=border)

이 노트북의 소스 파일은 {{% button href="https://github.com/awslabs/amazon-sagemaker-examples/tree/master/advanced_functionality/scikit_bring_your_own" icon="fab fa-github" %}}Github{{% /button %}} 에 공개되어 있습니다.


※ 이 모듈의 실습에는 약 30분이 소요됩니다.