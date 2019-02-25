# Identifying abnormal player behavior with Machine Learning

## Introduction
In this lab, you will analyze game data which randomly generated on AWS. Through this lab, you will collect data in real-time, process it, and then perform analyze it on AWS. Further, you will identify abnormal player behavior through machine learning.

### Architecture
<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/architecture.png"</img> 
</div>

### Lab Flow
User Profile data in Amazon DynamoDB and Play Log data in Amazon EC2 will be collected in Amazon S3 in near real-time using Amazon Kinesis Data Firehose. After that, AWS Glue is used to create the Data Catalog and data preparation is done with Glue ETL Job.
This processed data is then analyzed using Amazon Athena to query and Amazon SageMaker for more sophisticated analysis and prediction through Machine Learning.

### About Data
There are 2 types of data are collected. Let's take a look at the data used in this lab first.

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

It contains about 40 million play records from a total 20043 users. The virtual map where users played game is as follows. The **Red Zone** is the area where normal users can not go into.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/map.png"</img> 
</div>

## Lab
Select **us-east-1 (N.Virginia)** region on AWS Management Console before get started.

### Create Amazon EC2 Key Pairs
A key is required to make SSH connection to Amazon EC2 instance to be created later. If you already have a key in us-east-1 region, you make skip this step.
1. In the AWS Management Console, select **EC2** service.
2. On the left menu, click the **[Key Pairs]** menu, and then click **[Create Key Pair]** button.
3. Enter the **[Key pair name]** and click **[Create]** button to finish.
4. Verify .pem file is downloaded successfully.

### Create Amazon S3 Bucket
You need the Amazon S3 Bucket to store all the necessary data. In this lab, let's create a raw bucket to store raw data and analytic bucket to store processed data for analysis.
1. In the AWS Management Console, select **S3** service.
2. Click **[+ Create bucket]** button to create a bucket.
3. In **[Bucket name]**, enter a your own unique name, such as **gaming-raw**, and click **[Create]** button.
4. Create second bucket with name, such as **gaming-analytics**. Specify the name of the bucket so that it can be distinguished.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/Picture1.png"</img> 
</div>

### Create AWS CloudFormation Stack
The Amazon EC2, Amazon DynamoDB, AWS Lambda, and AWS IAM Roles used in the lab are created through the AWS CloudFormation stack. In addition to simply provisioning the resources, AWS CloudFormation stack also execute the logic to initialize Amazon DynamoDB through invoke Lambda function.
1. In the AWS Management Console, select **CloudFormation** service.
2. Click **[Create new stack]** button. Select **[Specify an Amazon S3 template URL]** option and enter this url(https://s3.amazonaws.com/anhyobin-gaming/cloudformation.yaml). Click **[Next]** button.
