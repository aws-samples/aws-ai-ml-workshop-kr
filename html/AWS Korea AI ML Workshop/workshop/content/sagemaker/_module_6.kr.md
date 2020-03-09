+++
title = "Module 6: 자동 모델 튜닝 결과 분석하기"
menuTitle = "자동 모델 튜닝 결과 분석하기"
date = 2019-10-15T15:17:57+09:00
weight = 206
+++

이 모듈에서는 앞에서 실행한 하이퍼파라미터 튜닝 작업의 결과를 해석하는 과정을 실습합니다. BokehJS와 pandas 라이브러리를 사용해 튜닝 결과를 Jupyter 노트북에서 테이블과 그래프 형태로 시각화해볼 수 있습니다.

SageMaker의 Jupyter 노트북 페이지 상단의 탭메뉴에서 `SageMaker Examples` 를 클릭합니다.

![select_sagemaker_examples](/images/sagemaker/module_5/select_sagemaker_examples.png?classes=border)

샘플 노트북 목록에서 `Hyperparameter Tuning` 을 선택합니다.

![select_hyperparamter_tuning](/images/sagemaker/module_5/select_hyperparamter_tuning.png?classes=border)

샘플 목록중 `HPO_Analyze_TuningJob_Results.ipynb` 를 찾아 우측의 `Use` 버튼을 클릭합니다. 다음과 같은 팝업창에서 `Create copy` 버튼을 클릭하여 관련 파일들을 사용자의 홈디렉토리로 복사를 진행합니다.

![create_a_copy_in_your_home_directory](/images/sagemaker/module_6/create_a_copy_in_your_home_directory.png?classes=border)

새로운 브라우저 탭에서 노트북이 오픈되면 준비가 완료됩니다.

![analyze_results_of_a_hyperparameter_tuning_job](/images/sagemaker/module_6/analyze_results_of_a_hyperparameter_tuning_job.png?classes=border)

모듈의 첫 부분에서 아래의 코드를 만나면 앞 모듈에서 실행된 `하이퍼파라미터 튜닝 작업 (Hyperparameter tuning jobs)`의 이름을 따옴표 안에 넣습니다.

![set_tuning_job](/images/sagemaker/module_6/set_tuning_job.png?classes=border)

실행 결과로 나오는 두 개의 그래프에서, 탐색된 하이퍼파라미터 값의 변화에 따른 loss 함수 값의 변화를 해석해 보시기 바랍니다. (그래프가 안나오면 해당 셀을 다시 한번 실행해 보세요.)

이 노트북의 소스 파일은 {{% button href="https://github.com/awslabs/amazon-sagemaker-examples/tree/master/hyperparameter_tuning/analyze_results" icon="fab fa-github" %}}Github{{% /button %}} 에 공개되어 있습니다.

※ 이 모듈의 실습에는 약 5분이 소요됩니다. 