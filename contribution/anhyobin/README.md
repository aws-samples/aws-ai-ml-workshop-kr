# Identifying abnormal player behavior with Machine Learning

## Introduction
In this lab, you will analyze game data which randomly generated on AWS. Through this lab, you will collect data in real-time, process it, and then perform analyze it on AWS. Further, you will identify abnormal player behavior through machine learning.

## Architecture
![alt text](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/architecture.png)

## Lab Flow
User Profile data in Amazon DynamoDB and Play Log data in Amazon EC2 will be collected in Amazon S3 in near real-time using Amazon Kinesis Data Firehose. After that, AWS Glue is used to create the Data Catalog and data preparation is done with Glue ETL Job.
This processed data is then analyzed using Amazon Athena to query and Amazon SageMaker for more sophisticated analysis and prediction through Machine Learning.

## About Data
There are 2 types of data are collected. Before get starts, let's take a look at the data used in this lab.

1. User Profile data in stored in Amazon DynamoDB and contains information about the level and class of users.

| pidx  | uclass | ulevel | utimestamp |
| :---- | :----- | :----- | :--------- |
| 8672 | healer | 9 | 2018-10-12 05:53:59.318075 |
| 13233 | warrior | 11 | 2018-10-12 05:48:44.748598 |

2. Play Log data contains information about the user's current coordinates, next coordinates, actions, and so on.

| posnewx | posnewy | posnewz | posoldx | posoldy | posoldz | pidx | idx | action | createdate |
| :------ | :------ | :------ | :------ | :------ | :------ | :--- | :-- | :----- | :--------- |
| 542 | 824 | 0 | 541 | 828 | 0 | 8672 | 30725885 | 0 | 2018-10-12 05:53:59.318075 |
| 668 | 245 | 0 | 666 | 240 | 0 | 13233 | 30721726 | 0 | 2018-10-12 05:48:44.748598 |

It contains about 40 million play records from a total 20043 users. The virtual map where users played game is as follows. The <span style="color:red">**Red Zone**</span> is the area where normal users can not go into.
