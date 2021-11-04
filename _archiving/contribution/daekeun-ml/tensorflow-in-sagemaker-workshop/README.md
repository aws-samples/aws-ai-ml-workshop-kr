# SageMaker Workshop: Tensorflow-Keras 모델을 Amazon SageMaker에서 학습하기
### 본 문서는 [Running your TensorFlow Models in SageMaker Workshop](https://github.com/aws-samples/TensorFlow-in-SageMaker-workshop) 의 한국어 버전이며, 원본과 다른 점들은 아래와 같습니다.
- Minor한 오타 수정
- 보충 설명 대폭 추가
- 솔루션 코드 포함 (원본 버전은 코드 솔루션이 제공되지 않습니다.)
- TensorFlow 1.14 대응 (원본은 TensorFlow 1.12 대응)


## Introduction

TensorFlow™를 통해 개발자는 클라우드에서 딥러닝을 쉽고 빠르게 시작할 수 있습니다.
이 프레임워크는 다양한 산업 분아에서 사용되고 있으며 특히 컴퓨터 비전, 자연어 이해 및 음성 번역과 같은 영역에서 딥러닝 연구 및 응용 프로그램 개발에 널리 사용됩니다.
머신 러닝 모델을 대규모로 구축, 학습 및 배포 할 수있는 플랫폼인 Amazon SageMaker를 통해 완전히 관리되는(fully-managed) TensorFlow 환경에서 AWS를 시작할 수 있습니다.

## Use Machine Learning Frameworks with Amazon SageMaker

Amazon SageMaker Python SDK는 다양한 머신러닝 및 딥러닝 프레임워크(framework)를 사용하여 Amazon SageMaker에서 모델을 쉽게 학습하고 배포할 수 있는 오픈 소스 API 및 컨테이너(containers)를 제공합니다. Amazon SageMaker Python SDK에 대한 일반적인 정보는 https://sagemaker.readthedocs.io/ 를 참조하세요.

Amazon SageMaker를 사용하여 사용자 지정 TensorFlow 코드를 사용하여 모델을 학습하고 배포할 수 있습니다. Amazon SageMaker Python SDK TensorFlow Estimator 및 model과 Amazon SageMaker 오픈 소스 TensorFlow 컨테이너를 사용하면 TensorFlow 스크립트를 작성하고 Amazon SageMaker에서 쉽게 실행할 수 있습니다.

이 워크샵에서는 TensorFlow 샘플 코드를 Amazon SageMaker에서 실행하는 방법을 소개합니다. 
SageMaker Python SDK에서 TensorFlow를 사용하기 위한 자세한 정보는 API references를 참조해 주세요.

워크샵은 아래 5개의 모듈로 이루어져 있습니다.

1. [Porting a TensorFlow script to run in SageMaker using SageMaker script mode.](0_Running_TensorFlow_In_SageMaker.ipynb)
2. [Monitoring your training job using TensorBoard and Amazon CloudWatch metrics.](1_Monitoring_your_TensorFlow_scripts.ipynb)
3. [Optimizing your training job using SageMaker pipemode input.](2_Using_Pipemode_input_for_big_datasets.ipynb)
4. [Running a distributed training job.](3_Distributed_training_with_Horovod.ipynb)
5. [Deploying your trained model on Amazon SageMaker.](4_Deploying_your_TensorFlow_model.ipynb)


## License Summary

이 샘플 코드는 MIT-0 라이센스에 따라 제공됩니다. LICENSE 파일을 참조하십시오.
