# Identifying abnormal player behavior with Machine Learning

## Introduction
In this lab, you will analyze game data which randomly generated on AWS. Through this lab, you will collect data in real-time, process it, and then perform analyze it on AWS. Further, you will identify abnormal player behavior through machine learning.

## Architecture
![alt text](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/image/architecture.png)

## 실습 흐름
DynamoDB 의 User Profile 데이터와 EC2 의 Play Log 데이터를 각각의 Kinesis Data Firehose 를 이용하여 실시간으로 S3 에 수집합니다. 이후 Glue Crawler 를 활용하여 Data Catalog 를 만들고 실제 분석을 위한 가공은 Glue Job 을 활용한 ETL 작업을 하게 됩니다.
이렇게 가공한 데이터는 이후 Athena 를 활용하여 쿼리를 통한 분석 해보고, 또한 SageMaker 를 이용하여 Machine Learning 을 통해 보다 정교한 분석 및 예측을 수행합니다.
