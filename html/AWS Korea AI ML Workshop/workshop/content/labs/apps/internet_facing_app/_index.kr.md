+++
title = "Internet-facing 앱 개발"
menuTitle = "Internet-facing 앱 개발"
date = 2019-10-15T15:16:07+09:00
weight = 310
+++

Amazon SageMaker는 데이터 사이언티스트와 개발자들이 쉽고 빠르게 구성,
학습하고 어떤 규모 로든 기계 학습된 모델을 배포할 수 있도록 해주는
관리형 서비스 입니다. 이 워크샵을 통해 Sagemaker notebook instance를
생성하고 샘플 Jupyter notebook을 실습하면서 SageMaker의 일부 기능을
알아보도록 합니다.

이 모듈에서는 영어를 독일어로 변환하는 SageMaker의 Sequence-to-Sequence
알고리즘을 이용한 언어번역기를 학습해보고 이 서비스를 인터넷을 통해
활용할 수 있는 방법에 대해 실습해 보겠습니다.

본 Hands-on에서는 SageMaker에서 생성한 Endpoint inference service를 웹
상에서 호출하기 위해 AWS Lambda와 AWS API Gateway를 Figure 4과 같은
데모를 만들어 보겠습니다.

![](/images/apps/internet_facing_app/image14.png?width="6.728391294838145in"
height="3.4464293525809273in)

<center>**Figure 4. SageMaker Internet-facing App Data Flow.**</center>

Figure 4에서는 SageMaker의 기능 데모를 위해 가장 간략한 구조를 채택하고
있습니다. 예를 들어 [Amazon S3의 Static Website에 다른 도메인 이름을
지정하기 위한 Route 53
서비스](https://docs.aws.amazon.com/AmazonS3/latest/dev/website-hosting-custom-domain-walkthrough.html)나
[캐슁 서비스를 위한
CloudFront](https://docs.aws.amazon.com/AmazonS3/latest/dev/website-hosting-cloudfront-walkthrough.html)
등의 서비스는 실제 비즈니스 적용 시에는 고려 되어야할 서비스입니다.

전제 Lab 시간은 일반 사용자의 경우 한시간에서 한시간 30분정도 소요 예상
됩니다.

{{% children  %}}



