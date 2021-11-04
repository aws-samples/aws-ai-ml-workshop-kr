+++
title = "Module 8: Internet-facing 앱 개발"
menuTitle = "Internet-facing 앱 개발"
date = 2019-10-15T17:25:48+09:00
weight = 209
draft=true
+++

Amazon SageMaker는 데이터 사이언티스트와 개발자들이 쉽고 빠르게 구성, 학습하고 어떤 규모 로든 기계 학습된 모델을 배포할 수 있도록 해주는 관리형 서비스 입니다. 이 워크샵을 통해 Sagemaker notebook instance를 생성하고 샘플 Jupyter notebook을 실습하면서 SageMaker의 일부 기능을 알아보도록 합니다.  

이 모듈에서는 영어를 독일어로 변환하는 SageMaker의 Sequence-to-Sequence 알고리즘을 이용한 언어번역기를 학습해보고 이 서비스를 인터넷을 통해 활용할 수 있는 방법에 대해 실습해 보겠습니다.

본 Hands-on에서는 SageMaker에서 생성한 Endpoint inference service를 웹 상에서 호출하기 위해 AWS Lambda와 AWS API Gateway를 Figure 4과 같은 데모를 만들어 보겠습니다. 

![sagemaker_internet_facing_app_data_flow](/images/sagemaker/module_9/sagemaker_internet_facing_app_data_flow.png?classes=border)

위 그림은 기능 데모를 위해 가장 간략한 구조를 채택하고 있습니다. 예를 들어 [Amazon S3의 Static Website에 다른 도메인 이름을 지정하기 위한 Route 53 서비스](https://docs.aws.amazon.com/AmazonS3/latest/dev/website-hosting-custom-domain-walkthrough.html)나  [캐슁 서비스를 위한 CloudFront](https://docs.aws.amazon.com/AmazonS3/latest/dev/website-hosting-cloudfront-walkthrough.html) 등의 서비스는 실제 비즈니스 적용 시에는 고려 되어야할 서비스입니다.

전제 Lab 시간은 일반 사용자의 경우 한시간에서 한시간 30분정도 소요 예상 됩니다. 


### 1: 영어-독어 번역 ML 모델 학습

#### Sequence-to-Sequence 알고리즘 노트북 열기

SageMaker가 지원하는 Seq2Seq 알고리즘은 MXNet 기반으로 개발된 Sockeye 알고리즘을 기반으로 개발된 최신의 Encoder-decoder 구조를 구현한 것으로 문서자동요약이나 언어 번역 서비스에 적용할 수 있습니다.

실습을 위해서 현재 설치되어 있는 SageMaker의 Jupyter 노트북의 예제들 중 아래의 디렉토리에 위한 Jupyter 노트북을 내려받으신 후 실행하시면 됩니다.

https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/src/release/2018-10/module8-SageMaker-Seq2Seq-Translation-English-German-InternetFacingApp.ipynb

![internet_facing_app](/images/sagemaker/module_9/internet_facing_app.png?classes=border)

#### 노트북에 대한 설명

본 노트북은 아래에 위치한 예제 노트북의 수정된 버전으로 미리 학습된 머신 러닝 모델을 사용하도록 바뀌었습니다.

`/sample-notebooks/introduction_to_amazon_algorithms/seq2seq_translation_en-de/SageMaker-Seq2Seq-Translation-English-German.ipynb`

상기 노트북은 빠른 학습 시간을 위해 Figure 7와 같이 전체 데이터 중 첫번째 10000개의 데이터의 대해서만 학습을 해서 `Seq2Seq` 알고리즘의 사용방법을 소개하고 있습니다.

![select_sample_data](/images/sagemaker/module_9/select_sample_data.png?classes=border)

아래의 이미지는 다운받은 corpus의 실제 데이터 내용으로 영어 및 독일어 데이터가 어떻게 문장 대 문장으로 매핑 되고 있는지를 보여주고 있습니다.

![comparision_english_german_dataset](/images/sagemaker/module_9/comparision_english_german_dataset.png?classes=border)

실제로는10000개의 샘플 문장으로 훈련한 번역기는 좋은 결과를 보여줄 수 없습니다. 그렇지만 전체 데이터 학습을 위해서는 선택하시는 SageMaker의 서버 Instance Type에 따라 다르지만 수시간에서 수일의 장시간이 소요될 수 있습니다. 따라서 이 노트북의 개발자들은 좀더 나은 품질의 번역 결과 체험을 원하시는 사용자들 위해 전체 데이터에 이미 훈련이 된 모델을 공유하고 있습니다. 

이 Pre-trained model을 사용하기 위해서는 노트북의 코드 중 `Endpoint Configuration` 직전의 코드를 아래와 같이 수정해서 이미 훈련된 모델을 다운로드 한 다음 본인의 S3 버켓으로 업로드 하시면 됩니다. 이때 Jupyter 노트북 마지막 줄의 `sage.delete_endpoint` 는 데모를 계속 진행하기 위해 실행하지 않습니다. 이를 위해 이번에는 가장 마지막 줄에 있는 코드를 주석 처리하겠습니다.

![delete_end_point_comment](/images/sagemaker/module_9/delete_end_point_comment.png?classes=border)

#### Pre-trained 모델을 사용 하기 위한 노트북 수정

노트북에서 하단의 S3 bucket 이름에 상기 생성한 S3 이름을 입력하시고 우측의 예와 비슷한 형식으로 prefix를 입력하시면 됩니다 .

![s3_bucket_name](/images/sagemaker/module_9/s3_bucket_name.png?classes=border)

#### 노트북 실행 방법

이제 노트북 전체를 실행할 준비가 되었습니다. Jupyter 노트북을 실행하는 방법은 코드가 있는 셀을 클릭으로 선택하신 후 Shift-enter 키를 누르시거나 또는 Jupyter 노트북 상단의 툴바에서 `Run cell, select below` 버튼을 클릭하셔도 됩니다.

전체 실행 과정은 약 12분에서 15분 정도 소요 됩니다. 각각의 셀을 실행시키면서 셀 하단에 표시되는 처리결과들을 확인해 보시기 바랍니다. 

노트북 코드 중 `Create endpoint configuration` 셀에서 현재 InstanceType이 `ml.m4.large` 로 되어 있습니다. Seq2Seq 알고리즘은 Neural network 기반이기 때문에 `ml.p2.xlarge` (GPU) instance를 사용하실 수 있지만 본 실습에서는 Free tier가 지원되는 `ml.m4.xlarge` 를 사용하고 있습니다. `ml.t2.*` instance는 time-out 문제가 발생할 수 있으므로 본 실습에서는 사용하지 않습니다.

![endpoint_configuration](/images/sagemaker/module_9/endpoint_configuration.png?classes=border)

노트북 코드 중 `Create endpoint` 셀은 새로운 서버를 설치하고 실행 코드를 설치하는 과정이므로 본 노트북에서는 가장 많은 시간 (약 10~11여분)이 소요 되는데 아래와 같은 메세지를 확인하시면 다음 모듈로 진행하시면 됩니다.

`Endpoint creation ended with EndpointStatus = InService`

![endpoint_creation_result](/images/sagemaker/module_9/endpoint_creation_result.png?classes=border)

노트북 가장 하단의 `delete_endpoint` 는 주석 처리 되어 있어야 endpoint 서버가 다음 실습을 위해 계속 운용될 수 있습니다. 만약에 실행 전에 수정하셨다면 `Create endpoint` 부분의 코드를 다시 실행하시기 바랍니다.

### 2: SageMaker Endpoint 호출 Lambda 함수 개발하기

본 모듈에서는 방금 생성한 SageMaker의 Inference service를 호출하는 Lambda 함수를 개발해 보겠습니다. 

#### Lambda 함수 생성하기 
