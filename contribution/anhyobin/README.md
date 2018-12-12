# 게임 데이터 분석을 위한 Data Lake 구축과 Machine Learning 을 활용한 분석

## 실습 소개
이번 실습에서는 AWS 상에서 임의로 생성한 게임 데이터를 분석해봅니다. 이를 통해 AWS 상에서 데이터를 실시간으로 수집하고, 가공한 뒤 실제로 분석을 간편하게 수행해볼 수 있습니다. 더 나아가 수집한 데이터를 학습하고 이를 이용해 이상 행동 패턴을 판단해보게 됩니다.

## 실습 아키텍처
![alt text](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/image/architecture.png)

## 실습 흐름
DynamoDB 의 User Profile 데이터와 EC2 의 Play Log 데이터를 각각의 Kinesis Data Firehose 를 이용하여 실시간으로 S3 에 수집합니다. 이후 Glue Crawler 를 활용하여 Data Catalog 를 만들고 실제 분석을 위한 가공은 Glue Job 을 활용한 ETL 작업을 하게 됩니다.
이렇게 가공한 데이터는 이후 Athena 를 활용하여 쿼리를 통한 분석 해보고, 또한 SageMaker 를 이용하여 Machine Learning 을 통해 보다 정교한 분석 및 예측을 수행합니다.
