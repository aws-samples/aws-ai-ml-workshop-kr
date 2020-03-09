+++
title = "Module 5: TensorFlow MNIST로 자동 모델 튜닝하기"
menuTitle = "TensorFlow MNIST로 자동 모델 튜닝하기"
date = 2019-10-15T15:17:54+09:00
weight = 205
+++

이 모듈에서는 TensorFlow MNIST 이미지 분류 예제를 기반으로 SageMaker의 자동 모델 튜닝 기능을 실습합니다. 이 기능은 기계 학습 알고리즘의 최적의 하이퍼파라미터 (Hyperparameter) 값을 베이지안 최적화 기법을 통해 찾아줍니다.

SageMaker의 Jupyter 노트북 페이지 상단의 탭메뉴에서 `SageMaker Examples` 를 클릭합니다.

![select_sagemaker_examples](/images/sagemaker/module_5/select_sagemaker_examples.png?classes=border)

샘플 노트북 목록에서 `Hyperparameter Tuning` 을 선택합니다.

![select_hyperparamter_tuning](/images/sagemaker/module_5/select_hyperparamter_tuning.png?classes=border)

샘플 목록중 `hpo_tensorflow_mnist.ipynb` 를 찾아 우측의 `Copy` 버튼을 클릭합니다. 다음과 같은 팝업창에서 `Create copy` 버튼을 클릭하여 관련 파일들을 사용자의 홈디렉토리로 복사를 진행합니다.

![create_a_copy_in_your_home_directory](/images/sagemaker/module_5/create_a_copy_in_your_home_directory.png?classes=border)

새로운 브라우저 탭에서 노트북이 오픈되면 준비가 완료됩니다.

![hyperparameter_tuning_using_sagemaker_tensorflow_container](/images/sagemaker/module_5/hyperparameter_tuning_using_sagemaker_tensorflow_container.png?classes=border)

모듈 실행중 아래 코드를 만나면 `bucket = sagemaker.Session().default_bucket()` 라인을 `bucket = ‘<모듈 1에서 생성한 s3 버킷의 이름(예: sagemaker-xxxxx)>’`으로 수정합니다. 부등호 부호(‘<’, ’>’)는 넣지 않습니다.

![insert_s3_bucket_name](/images/sagemaker/module_5/insert_s3_bucket_name.png?classes=border)

이 모듈에서는 MNIST 이미지 분류 예제의 하이퍼파라미터 중에서 learning rate 값을 자동으로 튜닝하며, 효과적인 탐색을 위해 최대값과 최소값을 아래 그림과 같이 설정합니다.

![hpo_range](/images/sagemaker/module_5/hpo_range.png?classes=border)

베이지안 최적화 기법은 하이퍼파라미터를 변경하면서 미리 지정된 숫자만큼 실험을 반복하는 특징이 있습니다. 이번 모듈에서는 병렬로 3개의 학습을 3번, 즉 총 9번의 실험을 시도하도록 아래와 같이 설정합니다.

![run_hpo](/images/sagemaker/module_5/run_hpo.png?classes=border)

하이퍼파라미터 튜닝 작업은 아래와 같은 코드로 실행하며, 실행하면 각 하이퍼파라미터 값에 대한 개별 학습이 백그라운드에서 시작됩니다.

![describe_hpo_job](/images/sagemaker/module_5/describe_hpo_job.png?classes=border)

이 때, SageMaker의 콘솔에서 새로운 `하이퍼파라미터 튜닝 작업 (Hyperparameter tuning jobs)`이 생성된 것을 확인할 수 있습니다. 다음 모듈을 위해 이 작업의 이름을 메모해 놓습니다.

![hpo_jobs](/images/sagemaker/module_5/hpo_jobs.png?classes=border)

실험이 모두 끝나면 하이퍼파라미터 튜닝 작업의 이름을 클릭해 튜닝 결과를 확인합니다. 아래 그림에서는 `learning_rate`가 `0.004928838215245632`가 최적의 값이며 이때의 loss 값은 `0.0642523318529129`인 것을 확인할 수 있습니다.

![hpo_result](/images/sagemaker/module_5/hpo_result.png?classes=border)

이 노트북의 소스 파일은 {{% button href="https://github.com/awslabs/amazon-sagemaker-examples/tree/master/hyperparameter_tuning/tensorflow_mnist" icon="fab fa-github" %}}Github{{% /button %}} 에 공개되어 있습니다.

※ 이 모델을 훈련하는데는 약 20분에서 25 분이 소요됩니다. 