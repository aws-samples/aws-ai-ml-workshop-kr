# Containers for Amazon SageMaker 

## Overview

단일 모델을 소규모 서비스로 배포 시에는 여러 모듈을 구성할 필요 없이 하나의 모듈 안에서 필요한 로직을 구성해도 무방합니다. 여러 종류의 모델들을 프로덕션 환경에서 배포 시,추론 환경을 안정적으로 빌드해야 함은 물론이고 각 모델의 프레임워크 종류, 프레임워크 버전 및 종속성을 고려해야 합니다. 또한, 동일한 시스템에서 실행되는 여러 모델들이 한정된 리소스를 두고 경쟁할 수 있으며, 특정 모델에서 오류 발생 시 여러 호스팅 모델들의 성능을 저하시킬 수 있습니다.

마이크로서비스 구조는 각 모듈을 독립된 형태로 구성하기 때문에 각 모듈의 관리가 쉽고 다양한 형태의 모델에 빠르게 대응할 수 있다는 장점이 있습니다. 도커(Docker)로 대표되는 컨테이너화 기술은 가상 머신과 달리 공통 운영 제체를 공유하면서 여러 모듈들에게 독립된 환경을 제공함으로써 유지 보수가 용이합니다.

Amazon SageMaker는 완전 관리형 머신 러닝 플랫폼으로 피쳐 전처리, 모델 훈련 및 배포의 머신 러닝 일련의 과정에 도커 컨테이너를 활용합니다. 컨테이너 내에 런타임, 라이브러리, 코드 등 필요한 모든 것이 패키징되기에, 로컬 환경에서 프로덕션 환경까지 일관성을 가지고 동일한 환경에서 모델을 훈련하고 배포할 수 있습니다.
AWS에서는 이미 딥러닝 프레임워크별로 각 태스크에 적합한(전처리, 훈련, 추론, 엘라스틱 추론 등) 전용 컨테이너를 AWS의 Docker 레지스트리 서비스인 Amazon Elastic Container Registry (이하 ECR) 에서 관리하고 있기 때문에 여러분은 컨테이너 빌드에 대한 고민을 하실 필요가 없습니다. 물론, 도커 파일들은 모두 오픈 소스로 공개되어 있기 때문에 도커 파일을 기반으로 여러분만의 컨테이너를 빌드해서 ECR로 등록할 수도 있습니다.

도커 컨테이너 개념과 ECR을 처음 접하시는 분들은 먼저 아래 링크를 통해 주요 개념을 이해하는 것을 권장 드립니다.
- https://docs.docker.com/get-started/
- https://aws.amazon.com/ecr

## Built-in algorithm Containers
SageMaker에서 제공하고 있는 17가지의 빌트인 알고리즘은 훈련 및 배포에 필요한 코드가 사전 패키징되어 있기에 별도의 코드를 작성할 필요가 없습니다.

빌트인 알고리즘 컨테이너 목록은 https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html 에서 찾을 수 있습니다. 예를 들어 서울 리전(ap-northeast-2)의 Linear Learner 알고리즘에 대한 컨테이너 이름은 `835164637446.dkr.ecr.ap-northeast-2.amazonaws.com/linear-learner:latest` 입니다. 빌트인 컨테이너는 SageMaker 관리형 인스턴스(훈련 인스턴스, 서빙 인스턴스)로만 가져올 수 있으므로 로컬 환경에서 실행할 수 없습니다.

다만, 예외적으로 XGBoost와 BlazingText는 오픈소스 라이브러리(BlazingText는 FastText)와 호환되므로 온프렘에서 훈련한 모델을 `model.tar.gz`로 아카이빙하여 S3에 업로드하는 방식이 가능합니다.

자세한 내용은 https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers-prebuilt.html 를 참조해 주세요.

## Managed Framework Containers 

SageMaker는 가장 널리 사용되고 있는 주요 머신 러닝 프레임워크와 각 프레임워크에 적합한 의존성 패키지를 제공하고 있습니다. 각 프레임워크에 대한 전처리, 훈련 및 추론 컨테이너는 AWS에서 최신 버전으로 정기적으로 업데이트되며, 딥러닝 프레임워크에는 CPU 및 GPU 인스턴스에 대한 별도의 컨테이너가 있습니다. 이러한 모든 컨테이너를 통칭하여 딥러닝 컨테이너(https://aws.amazon.com/machine-learning/containers)라고 합니다.

따라서, 여러분은 커스텀 컨테이너를 빌드하고 유지 관리할 필요 없이 알고리즘을 구현하는 파이썬 스크립트 코드 개발에만 집중할 수 있습니다. 여러분의 스크립트 코드는 프레임워크 SDK의 엔트리포인트로 전달하면 나머지 작업은 SageMaker가 자동으로 수행해 줍니다.

각 프레임워크의 컨테이너를 빌드하기 위한 Dockerfile과 소스 코드 또한 GitHub을 통해 제공하고 있습니다.

- Scikit-learn: https://github.com/aws/sagemaker-scikit-learn-container
- XGBoost: https://github.com/aws/sagemaker-xgboost-container
- PyTorch: https://github.com/aws/sagemaker-pytorch-container
- TensorFlow: https://github.com/aws/sagemaker-tensorflow-container
- MXNet: https://github.com/aws/sagemaker-mxnet-container
- Spark: https://github.com/aws/sagemaker-spark-container
- Hugging Face: https://github.com/aws/sagemaker-huggingface-inference-toolkit

자세한 내용은 https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers-prebuilt.html 를 참조해 주세요.

## Bring Your Own Container (BYOC)

아래와 같이 커스텀 컨테이너를 직접 빌드하는 것이 보다 효과적인 경우들이 있습니다.

- 프레임워크의 특정 버전이 지원되지 않는 경우
- 여러 프레임워크를 필요로 하는 경우(예: TensorFlow, PyTorch 동시 사용)
- 환경에 의존하는 라이브러리들이 매우 많을 경우
- 기본 환경에서 제공되지 않는 전처리/훈련/배포 솔루션을 사용하는 경우

이 때, 커스텀 컨테이너를 이용하면 SageMaker에서 사전 제공하지 않는 환경일 경우에도 SageMaker 기반으로 동작하도록 할 수 있습니다. 

자세한 내용은 https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers-create.html 를 참조해 주세요.