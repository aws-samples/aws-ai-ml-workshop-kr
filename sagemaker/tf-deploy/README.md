# Tensorflow Serving with SageMaker 

본 코드는 SageMaker 공식예제의 Tensorflow deployment 예제를 한글로 번역하고 일부 내용을 보완한 것입니다.
원본 소스는 아래 링크를 참조하십시오.
- https://github.com/aws/amazon-sagemaker-examples/tree/master/frameworks/tensorflow

본 코드의 실행은 다음 1,2,3번 노트북 파이을 차례로 실행하시면 됩니다.
- [1.mnist_train.ipynb](1.mnist_train.ipynb) - SageMaker Tensorflow Script mode로 MNIST 학습 모델을 생성합니다.
- [2.mnist_deploy_man.ipynb](2.mnist_deploy_man.ipynb) - Tensorflow Serving을 이용하여 학습된 모델로 추론을 실행합니다.
- [3.mnist_deploy_sm.ipynb](3.mnist_deploy_sm.ipynb) - SageMaker Endpoint기능을 이용하여 Tensorflow Serving을 실행합니다.




