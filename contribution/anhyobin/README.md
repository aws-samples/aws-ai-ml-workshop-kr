# Identifying abnormal player behavior with Machine Learning

## Table of Contents

## Introduction
In this lab, you will analyze game data which randomly generated on AWS. Through this lab, you will collect data in real-time, process it, and then perform analyze it on AWS. Further, you will identify abnormal player behavior through machine learning.

### Architecture
<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/architecture.png"></img> 
</div>

### Lab Flow
User Profile data in DynamoDB and Play Log data in EC2 will be collected in S3 in near real-time using Kinesis Data Firehose. After that, Glue is used to create the Data Catalog and data preparation is done with Glue ETL Job.
This processed data is then analyzed using Athena to query and SageMaker for more sophisticated analysis and prediction through Machine Learning.

### About Data
There are 2 types of data are collected. Let's take a look at the data used in this lab first.
1. User Profile data in stored in DynamoDB and contains information about the level and class of users.

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
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/map.png"></img> 
</div>

## Lab
Select **us-east-1 (N.Virginia)** region on AWS Management Console before get started.

### Create Amazon EC2 Key Pairs
A key is required to make SSH connection to EC2 instance to be created later. If you already have a key in us-east-1 region, you make skip this step.
1. In the AWS Management Console, select **EC2** service.
2. On the left menu, click the **[Key Pairs]** menu, and then click **[Create Key Pair]** button.
3. Enter the **[Key pair name]** and click **[Create]** button to finish.
4. Verify .pem file is downloaded successfully.

### Create Amazon S3 Bucket
You need the S3 Bucket to store all the necessary data. In this lab, let's create a raw bucket to store raw data and analytic bucket to store processed data for analysis.
1. In the AWS Management Console, select **S3** service.
2. Click **[+ Create bucket]** button to create a bucket.
3. In **[Bucket name]**, enter a your own unique name, such as **gaming-raw**, and click **[Create]** button.
4. Create second bucket with name, such as **gaming-analytics**. Specify the name of the bucket so that it can be distinguished.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/1.png"></img> 
</div>

### Create AWS CloudFormation Stack
The EC2, DynamoDB, Lambda, and IAM Roles used in the lab are created through the CloudFormation stack. In addition to simply provisioning the resources, CloudFormation stack also execute the logic to initialize DynamoDB through invoke Lambda function.
1. In the AWS Management Console, select **CloudFormation** service.
2. Click **[Create new stack]** button. Select **[Specify an Amazon S3 template URL]** option and enter this URL https://s3.amazonaws.com/anhyobin-gaming/cloudformation.yaml. Click **[Next]** button.
<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/2.png"></img> 
</div>

3. Enter name on **[Stack name]** field and select EC2 Key Pairs which created before on **[KeyName]**. Click **[Next]** to proceed.
4. Click **[Next]** on option page and check **[I acknowledged that AWS CloudFormation might create IAM resources with custom names.]** button then click **[Create]** to create stack.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/3.png"></img> 
</div>

5. This may takes about 10 minutes.
6. Read the following description of the AWS CloudFormation that is created while the stack is being created.

AWS CloudFormation template used in this lab automatically invokes the Lambda function **DDBInitialize**. This is possible through **custom resources** provided by AWS CloudFormation.
https://docs.aws.amazon.com/ko_kr/AWSCloudFormation/latest/UserGuide/template-custom-resources.html

```yaml
DDBInitLambdaInvoke:
  Type: Custom::DDBInitLambdaInvoke
  Properties:
    ServiceToken: !GetAtt DDBInitLambda.Arn
```

AWS CloudFormation basically creates the resources defined in the template in parallel at the same time, but it is also possible to control the logic with the **DependsOn** property in between.

```yaml
DDBInitLambda:
  Type: AWS::Lambda::Function
  DependsOn: DDBTable
```

Note that you need to send a resource creation complete response to CloudFormation stack to proceed. Therefor, the **DDBInitialize** function includeds:

```python
def send_response(event, context, response_status, response_data):
    response_body = json.dumps({
        "Status": response_status,
        "Reason": "See the details in CloudWatch Log Stream: " + context.log_stream_name,
        "PhysicalResourceId": context.log_stream_name,
        "StackId": event['StackId'],
        "RequestId": event['RequestId'],
        "LogicalResourceId": event['LogicalResourceId'],
        "Data": response_data
    })
    
    headers = {
        "Content-Type": "",
        "Content-Length": str(len(response_body))
    }
    
    response = requests.put(event["ResponseURL"], headers = headers, data = response_body)
```

Also, custom resources are executed in Create, Update, Delete situations for CloudFormation. So if you want to execute on specific condition, you should add logic to Lambda function like below:

```python
if event['RequestType'] == 'Delete':
  print 'Send response to CFN.'
  send_response(event, context, "SUCCESS", {"Message": "CFN deleted!"})
```

7. On the **[Resources]** tab, confirm that the all resource creation is completed. You can find connection information of EC2 instance in **[Outputs]** tab.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/4.png"></img> 
</div>

8. Let's confirm DynamoDB create and initialize properly via the Lambda fucntion. In AWS Management Console, select **DynamoDB** service.
9. Select the **[Tables]** menu on the left to see that the **UserProfile** table has been created. Select it and click **[Items]** on the right menu to check that the data in the table has been written.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/5.png"></img> 
</div>

### Create Amazon Kinesis Data Firehose
Data generated from DynamoDB and EC2 instnace are collected through Kinesis Data Firehose. Kinesis Data Firehose is a fully managed service for deliver streaming data to a specific target.
1. In AWS Management Console, select **Kinesis** service.
2. Select **[Get started]** button, then click **[Create delivery stream]** button on Deliver streaming data with Kinesis Firehose delivery streams.
3. Enter **stream-playlog** on **[Delivery stream name]**. Select **[Direct PUT or other sources]** option for Source. Click **[Next]** to proceed.
4. Kinesis Data Firehose supports data pre-processing with AWS Lambda. But at this time, we will not use this feature. Click **[Next]**.
5. Select **[Amazon S3]** as a destionation, and select **raw** data bucket which created before for **[S3 bucket]**. Enter **playlog/** on **[Prefix]**. Click **[Next]** if the setting are as follows:

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/6.png"></img> 
</div>

6. Set **1MB** on **[Buffer size]** and set **60seconds** on **[Buffer interval]**.
7. Click the **[Create new or choose]** button under IAM role. The IAM page opens and automatically configures the IAM role. Click the **[Allow]** button. When you return to the Kinesis Data Firehose creation page, select **[Next]** and confirm your settings (Destination, S3 buffer conditions, etc.). Click **[Create delivery stream]** to complete.
8. If the status changes to Activce after a while, the Kinesis Data Firehose creation is complete.
9. In this lab, 2 Kinesis Data Firehose are required. Create the second Kinesis Data Firehose in the same way as above.
10 In **[Delivery stream name]**, enter **stream-userprofile**.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/7.png"></img> 
</div>

11. Select **[Amazon S3]** as a destionation, and select **raw** data bucket which created before for **[S3 bucket]**. Enter **userlog/** on **[Prefix]**. Click **[Next]** if the setting are as follows:

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/8.png"></img> 
</div>

12. Set **1MB** on **[Buffer size]** and set **60seconds** on **[Buffer interval]**.
13. If you have created 2 Kinesis Data Firehose, proceed to the next step.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/9.png"></img> 
</div>

### Configure Amazon DynamoDB
DynamoDB has UserProfile data. In this step, you configure level-up history data for each user to store to S3. This is done using the Kinesis and Lambda function.
1. In AWS Management Console, select **DynamoDB** service.
2. Select the **[Tables]** menu on the left and select **UserProfile** table.
3. Click **[Manage Stream]** button on the **[Overview]** tab. This allow you to capture changes in DynamoDB table.
4. Select **[New and old images]** option and click the **[Enable]** button. You can see that the stream feature is activated as follows:

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/10.png"></img> 
</div>

5. In AWS Management Console, select **Lambda** service. Add a DynamoDB Stream to a Lambda function as an event trigger, and if any changes are made on DynamoDB, the data will be collected to Kinesis.
6. Select the pre-created **StreamUserLog** function and click **[DynamoDB]** on the left.
7. For **[DynamoDB table]**, select **UserProfile**, enter **100** on **[Batch size]**, and select **[Trim horizon]** on **[Starting position]**. Make sure that the **[Enable trigger]** option is checked below, then click the **[Add]** button below.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/11.png"></img> 
</div>

8. Click the **[Save]** button in the top right corner to save changes. Make sure that you have applied the following screen capture:

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/12.png"></img> 
</div>

A quick look at the Lambda function reveals that changes are capture and collect to **stream-userprofile** Kinesis Data Firehose that you created earlier. Lambda can also store data directly into S3, but with this configuration, Kinesis acts as a buffer to prevent excessive S3 PUT requests.

```python
response = client.put_record(
  DeliveryStreamName = 'stream-userprofile',
  Record = {
    'Data' : data
  }
)
```

### Amazon EC2 instance setup and data collect through Amazon Kinesis Agent
In this lab, EC2 instance is used to continuously generate PlayLog and update UserProfile stored in DynamoDB. PlayLog is collected via Kinesis Agent installed on EC2 instance to Kinesis Data Firehose. In case of UserProfile, collection is done through DynamoDB streams when there is update on table. As a result, all data is collected in the S3 bucket, which is the target destination of each Kinesis Data Firehose. In this way, the raw data will be collected in S3 as Data Lake, which will be easily utilized for later analysis.

1. In AWS Management Console, select **EC2** service.
2. Select the **[Instances]** menu on the left and select the **PlayLogGenerator** instance created via CloudFormation.
3. If you look at the **[Description]** tab for that instance, required permissions are associated with an IAM role called **Ec2GeneratorRole**. This is a best practice for authorizing applications running on EC2 instances.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/13.png"></img> 
</div>

4. After checking the public IP of the instance, SSH connect to it using SSH client.
5. First, check whether the files exists as follows:

```bash
[ec2-user@ip-172-31-84-120 ~]$ ls
playlog_gen.py  StreamLog  UserList

[ec2-user@ip-172-31-84-120 ~]$ ls -l /tmp/archived_playlog/2018/10/09/01/
total 6493876
-rw-rw-r-- 1 ec2-user ec2-user 169079641 Oct 17 08:32 run-1538992116187-part-r-00000
-rw-rw-r-- 1 ec2-user ec2-user 169128956 Oct 17 08:32 run-1538992116187-part-r-00001
...
...
...
```

6. Because it is hard to upload 40 million data sets at this time, upload the archived log data from **/tmp/archived_playlog/** path to the S3 bucket using following AWS CLI command:

```bash
[ec2-user@ip-172-31-84-120 ~]$ aws s3 cp /tmp/archived_playlog/ s3://<YOUR RAW BUCKET>/playlog/ --recursive
upload: ../../tmp/archived_playlog/2018/10/09/01/run-1538992116187-part-r-00000 to s3://gaming-raw/playlog/2018/10/09/01/run-1538992116187-part-r-00000
...
...
...
```

7. In AWS Management Console, select **S3** service.
8. In the raw bucket, you can see that the 40 copies have been copied. Under the bucket, prefix **YYYY/MM/DD/HH** is for data partitioning. In order to use the same partition as the data you collect with Kinesis Data Firehose, you must make sure that it is stored with the following structure:

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/14.png"></img> 
</div>

9. Return to the EC2 instance and check the Kinesis Agent settings via the following command:

```bash
[ec2-user@ip-172-31-84-120 ~]$ sudo service aws-kinesis-agent status
aws-kinesis-agent is stopped

[ec2-user@ip-172-31-84-120 ~]$ cat /etc/aws-kinesis/agent.json 
{
  "cloudwatch.emitMetrics": true,
  "firehose.endpoint": "firehose.us-east-1.amazonaws.com",
  
  "flows": [
    {
      "filePattern": "/tmp/playlog/*.json",
      "deliveryStream": "stream-playlog"
    }
  ]
}
```
