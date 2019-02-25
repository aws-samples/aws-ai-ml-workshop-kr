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


| First Header  | Second Header | Third Header         |
| :------------ | :-----------: | -------------------: |
| First row     | Data          | Very long data entry |
| Second row    | **Cell**      | *Cell*               |
| Third row     | Cell that spans across two columns  ||
[Table caption, works as a reference][section-mmd-tables-table1] 
