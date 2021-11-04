+++
title = "Module 7: PyTorch MNIST"
menuTitle = "PyTorch MNIST"
date = 2019-10-15T15:18:00+09:00
weight = 207
+++

이 모듈에서는 앞에서 실행한 하이퍼파라미터 튜닝 작업의 결과를 해석하는 과정을 실습합니다. BokehJS와 pandas 라이브러리를 사용해 튜닝 결과를 Jupyter 노트북에서 테이블과 그래프 형태로 시각화해볼 수 있습니다.

SageMaker의 Jupyter 노트북 페이지 상단의 탭메뉴에서 `SageMaker Examples` 를 클릭합니다.

![select_sagemaker_examples](/images/sagemaker/module_5/select_sagemaker_examples.png?classes=border)

샘플 노트북 목록에서 `Sagemaker Python Sdk` 를 선택합니다.

![select_python_sdk](/images/sagemaker/module_7/select_python_sdk.png?classes=border)

샘플 목록중 `pytorch_mnist.ipynb` 를 찾아 우측의 `Copy` 버튼을 클릭합니다. 다음과 같은 팝업창에서 `Create copy` 버튼을 클릭하여 관련 파일들을 사용자의 홈디렉토리로 복사를 진행합니다.

![create_a_copy_in_your_home_directory](/images/sagemaker/module_7/create_a_copy_in_your_home_directory.png?classes=border)

새로운 브라우저 탭에서 노트북이 오픈되면 준비가 완료됩니다.

![mnist_training_using_pytorch](/images/sagemaker/module_7/mnist_training_using_pytorch.png?classes=border)

모듈 실행중 아래 코드를 만나면 `bucket = sagemaker.Session().default_bucket()` 라인을 `bucket = ‘<모듈 1에서 생성한 s3 버킷의 이름(예: sagemaker-xxxxx)>’` 으로 수정합니다. 부등호 부호(‘<’, ’>’)는 넣지 않습니다.

![s3_bucket_name](/images/sagemaker/module_7/s3_bucket_name.png?classes=border)

이 노트북의 소스 파일은 {{% button href="https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk/pytorch_mnist" icon="fab fa-github" %}}Github{{% /button %}} 에 공개되어 있습니다.

※ 이 모델을 훈련하는데는 약 10분에서 15 분이 소요됩니다.