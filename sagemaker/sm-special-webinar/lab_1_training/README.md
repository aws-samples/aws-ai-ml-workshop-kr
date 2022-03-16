# Amazon SageMaker 모델 학습 실습

본 실습은 SageMaker 모델 학습을 수행하는 방식 중 script mode로, 고객 분들이 가지고 있는 모델 학습 코드를 이용하여 별도 학습 클러스터에서 container 기반으로 학습을 수행하는 방법입니다. 
- 1.1.SageMaker-Training.ipynb는 학습을 위한 가장 기본적인 코드로만 구성되어 있습니다.
- 1.2.SageMaker-Training+Experiments.ipynb는 1.1에 추가적으로 SageMaker Experiments 서비스를 추가하여 실험 관리/비교를 제공합니다.
- 1.3.SageMaker-Training+Experiments+Processing.ipynb는 1.2에 추가적으로 SageMaker Processing 서비스를 추가하여 Evaluation하는 방법을 실습합니다.

각 기능은 Serverless로 별도의 학습 및 처리 클러스터를 이용하여 동작하기에, 필요한 경우 MLOps의 workflow에 따라 모델 학습과 처리 단계에서 바로 사용할 수 있습니다.