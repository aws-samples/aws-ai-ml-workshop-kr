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

In agent.json, you can see that all logs corresponding to **"filePattern": "/tmp/playlog/&#42;.json"** are configured to be collected by **"deliveryStream": "stream-playlog"**, the Kinesis Data Firehose.

10. Run following command to implement Kinesis Agent:

```bash
[ec2-user@ip-172-31-84-120 ~]$ sudo service aws-kinesis-agent start
aws-kinesis-agent startup                                  [  OK  ]
```

11. Run following command to generate logs. Make sure include & to run on the backend.

```bash
[ec2-user@ip-172-31-84-120 ~]$ python playlog_gen.py &
[1] 2296
```

12. Python script use multiprocessing to generate PlayLogs and update UserProfile on DynamoDB continuously.

```bash
[ec2-user@ip-172-31-84-120 ~]$ ps -ef | grep python
ec2-user  2296  2138  1 06:44 pts/0    00:00:00 python playlog_gen.py
ec2-user  2298  2296  0 06:44 pts/0    00:00:00 python playlog_gen.py
ec2-user  2299  2296 19 06:44 pts/0    00:00:01 python playlog_gen.py
ec2-user  2303  2138  0 06:44 pts/0    00:00:00 grep --color=auto python
```

```python
proc1 = Process(target = playlog)
proc2 = Process(target = dynamodb)
```

```python
def playlog():
  filename = '/tmp/playlog/' + str(flag) + '_playlog.json'
  with open(filename, 'a') as logFile:
    json.dump(raw_data, logFile)
    # Kinesis Agent parsed from each file based on \n
    logFile.write('\n')
    os.chmod(filename, 0o777)
```

```python
def dynamodb():
  if(ulevel < 100):
    response = table.update_item(
      Key = {'pidx': selectUser},
      UpdateExpression = "SET ulevel = :ul, utimestamp = :ut",
      ExpressionAttributeValues = {
        ':ul' : ulevel + 1,
        ':ut' : currentTime
      },
      ReturnValues = "UPDATED_NEW"
    )
```

13. You can see log are generated in the path **/tmp/playlog/**.

```bash
[ec2-user@ip-172-31-84-120 ~]$ ls -l /tmp/playlog
total 456
-rwxrwxrwx 1 ec2-user ec2-user 183067 Oct 21 06:44 0_playlog.json
-rwxrwxrwx 1 ec2-user ec2-user 185000 Oct 21 06:45 1_playlog.json
-rwxrwxrwx 1 ec2-user ec2-user  92415 Oct 21 06:45 2_playlog.json
```

14. You can check the Kinesis Agent's log through **/var/log/aws-kinesis-agent/aws-kinesis-agent.log**. This allows you to se the data you are collecting with the Kinesis Data Firehose after parsing by Kinesis Agent. Run the following command to check the log:

```bash
[ec2-user@ip-172-31-84-120 ~]$ tail -f /var/log/aws-kinesis-agent/aws-kinesis-agent.log
2018-10-21 06:47:14.324+0000 ip-172-31-84-120 (Agent.MetricsEmitter RUNNING) com.amazon.kinesis.streaming.agent.Agent [INFO] Agent: Progress: 6223 records parsed (1142565 bytes), and 6000 records sent successfully to destinations. Uptime: 210022ms
...
...
...
```

15. In AWS Management Console, select **S3** service. Make sure that Kinesis Data Firehose is collecting the data to the target bucket.
16. In **raw** data bucket, you can see the **playlog** and **userlog** folder are created. Under **userlog**, updated UserProfile data from DynamoDB is stored.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/15.png"></img> 
</div>

17. Under **playlog**, you can see that the data is being partitioned into **YYYY/MM/DD/HH** structure and collected in near real-time as show below.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/16.png"></img> 
</div>

18. Go to the **DynamoDB** service in the AWS Management Console and check the **UserProfile** table. You can see the table has been updated.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/18.png"></img> 
</div>

19. You have completed the data collection phrase.

### Create AWS Glue Data Catalog
In the previous process, you collected the logs generated by the server and the user information updated in DynamoDB to S3 in near real-time. Now let's start data analysis.
The first thing you do is create a Glue Data Catalog. This is a central metastore repository that is compatible with Apache Hive Metatore. It contatins the table definition and location of data set, and uses the Glue Crawler to fill it. The Glue Crawler associate to the data store, extracts the data schema, and populates the Glue Data Catalog with metadata.

1. In AWS Management Console, select **Glue** service.
2. Select the **[Crawlers]** menu on the left side and click the **[Add crawler]** button.
3. In **[Crawler name]**, enter the desired name, such as **gamelog-raw**, and click the **[Next]** button.
4. Select **[S3]** for **[Choose a data store]** and **[Specified path in my account]** for **[Crawl data in]**. In **[Include path]**, click on the folder button on the right and select the bucket from which you collected the raw data. Click **[Next]** to proceed.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/19.png"></img> 
</div>

5. Select **[No]** for **[Add another data store]** and click **[Next]**. For the IAM role, select the IAM role generated by CloudFormation. Select the **[Choose an existing IAM role]** option and the **[IAM role]** to select **[GlueETLRole]**. Click **[Next]**.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/20.png"></img> 
</div>

6. **[Frequency]** selects the **[Hourly]** option. This is to keep this information up to date because the data is stored in S3 in **YYYY/MM/DD/HH** structure.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/21.png"></img> 
</div>

7. You need a database that stores metadata. Click the **[Add database]** button, enter **gamelogdb** for **[Database name]**, and click the **[Create]** button. Click **[Next]**, review the content, and click the **[Finish]** button to finish.
8. The generated Glue Crawler will run through the hourly schedule, but first you need the initial data for the lab. Select the created crawler and click the **[Run crawler]** button.
9. When the crawling is completed, you can find 2 tables added as below:

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/22.png"></img> 
</div>

10. Let's check the data catalog. Select the **[Tables]** menu on the left, 2 tables have been added. Click each of them to view the corresponding table information. Here you can see that the data store contains the table information and that the partitions are automatically recognized.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/23.png"></img> 
</div>

11. In addition, table information of DynamoDB can be managed through Glue Data Catalog. Create the Glue Crawler in a similar way as before.
12. In **[Crawler name]**, enter the same name as **userprofile**, and in **[Choose a data store]**, select **[DynamoDB]**. Select **UserProfile** for **[Table name]** and **[GlueETLRole]** for **[IAM role]**. For DynamoDB tables, **[Frequency]** is select **[Run on demand]** because there is no schema change.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/24.png"></img> 
</div>

13. Data from DynamoDB can also be stored in the same Data Catalog database. Select the previously created **[gamelogdb]** to complete the Glue Crawler creation.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/25.png"></img> 
</div>

14. After you have finished creating the crawler, click the **[Run crawler]** button to run it.
15. Once the crawler is running, 1 table added. Go to the **[Tables]** menu on the left. You can see a total of 3 tables as shown below:

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/26.png"></img> 
</div>

### Run AWS Glue ETL Job
In this step, let's perform the ETL operation to the Glue Data Catalog created earlier. In the lab, you use Athena and SageMaker to do a data analysis, each with different data. With Athena, you query to analyze the entire set of data from the lab, and SageMaker will have to create separate sets of data for training the machine learning model. These ETL operations can be performed through the Glue Job.
In Glue you can run ETL scripts written in two languages: Python and Scala. You can also create, test, and run scripts automatically.

1. In the AWS Management Console, select **Glue** service.
2. Click the **[Jobs]** button under the ETL on the left and select the **[Add job]** button to start creating the Glue Job.
3. Enter **gamelog-etl** for **[Name]** and **[GlueETLRole]** for **[IAM role]**. The **[A new script to be authored by you]** for **[This job runs]** to creates own Python ETL script. Expand the **[Advanced properties]** menu at the bottom and set **[Job metrics]** to **[Enable]**. This allows monitoring through CloudWatch when ETL jobs are performed. Click **[Next]**.
4. Connections will be used when using data stored in RDS, Redshift, etc. Click **[Next]** and click the **[Save job and edit script]** button to create the job.
5. On the Scripting page, delete all content, then copy and paste the Python script from the below. Change the name of bucket with your analytics bucket in **s3Bucket = "s3://&#60;YOUR ANALYTICS BUCKET&#62;"** section, click the **[Save]** button at the top, and click the **[Run job]** button.

```python
import sys
import datetime
from awsglue.transforms import *
from awsglue.dynamicframe import DynamicFrame
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

s3Bucket = "s3://YOUR ANALYTICS BUCKET"
s3Folder ="/gamelog/"

# Set source data with playlog in S3, userprofile in DynamoDB
datasource0 = glueContext.create_dynamic_frame.from_catalog(database = "gamelogdb", table_name = "playlog")
datasource1 = glueContext.create_dynamic_frame.from_catalog(database = "gamelogdb", table_name = "userprofile")

df1 = datasource0.toDF()
df1.createOrReplaceTempView("playlogView")
df2 = datasource1.toDF()
df2.createOrReplaceTempView("userprofileView")

# Query to join playlog and userprofile
sql_select_athena = 'SELECT playlogView.partition_0, playlogView.partition_1, playlogView.partition_2, playlogView.partition_3, playlogView.posnewz, playlogView.posnewy, playlogView.posnewx, playlogView.posoldz, playlogView.posoldy, playlogView.posoldx, playlogView.action, playlogView.idx, playlogView.pidx, playlogView.createdate, userprofileView.pidx, userprofileView.uclass, userprofileView.ulevel FROM playlogView, userprofileView WHERE playlogView.pidx = userprofileView.pidx ORDER BY playlogView.createdate'
sql_select_ml = 'SELECT playlogView.posnewx, playlogView.posnewy FROM (SELECT * FROM playlogView ORDER BY playlogView.pidx, playlogView.createdate)'

exec_sql_athena = spark.sql(sql_select_athena)
exec_sql_dyf_athena = DynamicFrame.fromDF(exec_sql_athena, glueContext, "exec_sql_dyf_athena")

exec_sql_ml = spark.sql(sql_select_ml)
exec_sql_dyf_ml = DynamicFrame.fromDF(exec_sql_ml, glueContext, "exec_sql_dyf_ml")

# Set target as S3 into two types, json and csv
datasink1 = glueContext.write_dynamic_frame.from_options(frame = exec_sql_dyf_athena, connection_type = "s3", connection_options = {"path": s3Bucket + s3Folder + "gamelog_athena", "partitionKeys" : ["partition_0", "partition_1", "partition_2", "partition_3"]}, format = "json", transformation_ctx = "datasink1")
datasink2 = glueContext.write_dynamic_frame.from_options(frame = exec_sql_dyf_ml, connection_type = "s3", connection_options = {"path": s3Bucket + s3Folder + "gamelog_sagemaker"}, format = "csv", transformation_ctx = "datasink2")

job.commit()
```

Let's take a look at ETL script. ou can see that both **datasource0** and **datasource1** utilize 2 data sources, all of which are data from the Glue Data Catalog:

```python
datasource0 = glueContext.create_dynamic_frame.from_catalog(database = "gamelogdb", table_name = "playlog")
datasource1 = glueContext.create_dynamic_frame.from_catalog(database = "gamelogdb", table_name = "userprofile")
```

Query through Athena requires all the data sets, so perform data conversion task with statement **sql_select_athena**. This statement joins PlayLog and UserProfile data from S3 and DynamoDB:

```sql
WHERE playlogView.pidx = userprofileView.pidx
```

SageMaker requires 2-dimension data of x and y coordinates for model training. However, you should sort by pidx to show the behavior pattern for each user. Considering this, **sql_select_ml** looks like below:

```sql
SELECT playlogView.posnewx, playlogView.posnewy FROM (SELECT * FROM playlogView ORDER BY playlogView.pidx, playlogView.createdate)
```

Data for Athena is saved in json format and data for SageMaker is saved in csv format. In addition, it supports column based data types such as Parquet, ORC. The format is simply set via **format = ""**.

```python
datasink2 = glueContext.write_dynamic_frame.from_options(format = "csv" ...)
```

6. Click the **[Run job]** button to run. And it takes some time for the actual operation to start with log like **Oct 21, 2018, 7:24:42 PM Pending execution**. When Glue executes a job in a Serverless environment, the resources to execute the ETL script are first provisioned internally, and then the script is executed. So there is a little delay.
7. When the job has finished running, you can see a log similar to **file: s3://gaming-analytics/gamelog/gamelog_sagemaker/run-1540118641016-part-r-00135 End of LogType:stdout**.
8. In the AWS Management Console, select **S3** Service.
9. Go to the analytics bucket that you specified in the ETL script and verify that the ETL operation successfully runs and the converted data is saved. Within the bucket, 2 folders are created under the **gamelog** folder, **gamelog_athena** and **gamelog_sagemaker**, and you can see that the converted data is saved as below:


<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/27.png"></img> 
</div>

10. You have collected the raw data and have completed the conversion for analysis. But it is not the end. Finally, you need to add the newly added data to the Glue Data Catalog.
11. In the AWS Management Console, select **Glue** service.
12. Select the **[Crawlers]** menu on the left and create a Glue Crawler in a familiar way. **[Crawler name]** enter **gamelog-analytics**. For **[Choose a data store]**, select **[S3]** and **[Include path]** selects the bucket for which you have collected new analytics data. **[Frequency]** is **[Run on demand]**, **[Database]** select **[gamelogdb]** as before. After creation, click **[Run crawler]** button to execute.
13. When the execution is completed, 2 tables are added. From the **[Tables]** on the left, you can see that there are 5 tables in total.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/28.png"></img> 
</div>

14. In this way it is possible to easily build a Data Lake regardless of the data type or capacity.

### Data Analysis with Amazon Athena
Through the previous step, you proceeded to process and store the data required for analysis. Once you have built the Data Lake, you can start analyzing quickly using your preferred analytics services. Athena, as well as Redshift, EMR, etc., can analyze data using Glue Data Catalog and data stored in S3.
In this lab, you use Athena to analyze data stored in S3 using standard SQL. Because Athena is immediately integrated with the Glue Data Catalog, data can be analyzed with interactive queries directly without requiring a separate schema definition.
1. In the AWS Management Console, select **Athena** service.
2. Click the **[Get Started]** button. If the tutorial appears, close it.
3. On the left **[Database]**, select **[gamelogdb]** stored in the previously created Glue Data Catalog. 4 tables stored in S3 except the DynamoDB table appear.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/29.png"></img> 
</div>

4. The data that can be confirmed by each table is as follows:

| Table Name | Description |
| :--- | :--- |
| gamelog_athena | A dataset that contains all the information that joins playlog and userlog |
| gamelog_sagemaker | Machine learning Data set containing only x, y coordinates for model learning |
| playlog | Data set containing only the play history of users created in the EC2 instance |
| userlog | Data set containing the history of user profiles stored in DynamoDB S|

5. Now try to analyze the data through the SQL query below. You can use the Presto function with Athena. Try the sample SQL query below:

Data in the gamelog_athena table which Glue ETL perform join operation.
```sql
SELECT * FROM gamelog_athena WHERE CAST(partition_2 AS BIGINT) = 9 limit 100;
```

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/30.png"></img> 
</div>

Try to find out which class player chose the most.
```sql
SELECT COUNT(DISTINCT pidx) AS users, uclass FROM gamelog_athena GROUP BY uclass;
```
<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/31.png"></img> 
</div>

Highest level users through the data in the userlog table.
```sql
SELECT * FROM userlog WHERE ulevel IN (SELECT MAX(ulevel) FROM userlog);
```

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/32.png"></img> 
</div>

View the number of players played in a specific area.
```sql
SELECT COUNT(DISTINCT pidx) AS hotzone FROM gamelog_athena WHERE posnewx BETWEEN 300 AND 500 AND posnewy BETWEEN 400 AND 700;
```

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/33.png"></img> 
</div>

6. **[QUIZ]** Now use the query to find the player who is behaving abnormaly. The hint is on the map where the user played the game. If you use coordinate information like posnewx, posnewy, you can find the users you want.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/map.png"></img> 
</div>

### Machine learning model training and abnormal behavior identification through Amazon SageMaker
This step uses SageMaker's built-in algorithm, Random Cut Forest (RCF), to detect data set anomalies. Let's apply the RCF algorithm for anomaly detection to data collected through lab. In the past, you have already collected data sets for learning and have completed data preparations.
SageMaker is a fully managed platform that enables developers and data analysts to quickly and easily build, train and deploy machine learning models.
The RCF algorithm is a non-gradient learning algorithm that detects the outliers contained in the data set. For the RCF algorithm supported by SageMaker, anomaly score is given to each data. If the outlier score is low, the data is likely to be normal, while a high score indicates a high likelihood of an abnormality.
1. In the AWS Management Console, select **SageMaker** service.
2. Go to the **[Notebook instances]** menu on the left and click the **[Create notebook instance]** button.
3. Enter **gamelog-ml-notebook** for **[Notebook instance name]** and **[ml.m4.2xlarge]** for **[Notebook instance type]**. **[IAM role]** selects **[Create a new role]**. On the create IAM role screen, **[Specific S3 buckets]** under **[S3 buckets you specify]**, enter the analytics bucket where **gamelog_sagemaker** is stored or select **[Any S3 bucket]**.


<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/34.png"></img> 
</div>

4. Review setting, and then click the **[Create notebook instance]** button to create the notebook instance.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/35.png"></img> 
</div>

5. After a few moments, the notebook instance you created will change to **InService** status. Click the **[Open]** button to connect to the Jupyter notebook.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/36.png"></img> 
</div>

6. You can import the notebook you have already created and proceed with the exercises, but at this time, enter and execute the code yourself to see step by step. Click the **[New]** button on the right and select **[conda_python3]**.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/37.png"></img> 
</div>

7. Click **[Untitles]** at the top, enter **GameLog-RCF**, etc., and click the **[Rename]** button.
8. The first cell reads data from gamelog_sagemaker. From the code page, copy and paste the corresponding part below. In **bucket = 'YOUR ANALYTICS BUCKET'**, enter the analytics bucket name where the data is stored. Click the **[▶Run]** button.

```python
import boto3
import pandas

s3 = boto3.client('s3')
bucket = 'YOUR ANALYTICS BUCKET'
prefix = 'gamelog/gamelog_sagemaker/'
response = s3.list_objects_v2(Bucket = bucket, Prefix = prefix)
objs = []

for obj in response['Contents']:
    objs.append(obj['Key'])

game_data = pandas.concat([pandas.read_csv('s3://' + bucket + '/' + obj, delimiter = ',') for obj in objs]) 
```

9. In the second cell, graph the entire data set. At the start of the lab, it takes some time, because it uses about 40 million data sets, which is the sum of the data uploaded in S3 and the data collected by Kinesis Data Firehose. Copy and paste the part below from the code page and click the **[▶Run]** button. Only output such as **<matplotlib.axes.&#95;subplots.AxesSubplot at 0x7f23bb72c358>** will appear, and if the graph is not drawn, skip or restart it.

```python
import matplotlib

# Set graph parameters for 40 million data set
matplotlib.rcParams['agg.path.chunksize'] = 100000
matplotlib.rcParams['figure.figsize'] = [20, 10]
matplotlib.rcParams['figure.dpi'] = 100

game_data.plot.scatter(
    x = 'posnewx',
    y = 'posnewy'
)
```

If you look at the graph, you can find points that are suspicious. The x coordinate is from 1000 to 1200 and the y coordinate is from 0 to 400. Already you have found a user in Athena who has a move in that area.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/38.png"></img> 
</div>

10. In the next cell, you now encode the data into the RecordIO prodobuf format for model training. Like other algorithms in SageMaker, model learning shows the best performance in that format. This step simply converts the original data in CSV format and stores the result in the S3 bucket. From the code page, paste the code below and modify the **bucket = 'YOUR ANALYTICS BUCKET'** to your own analytics bucket and click **[▶Run]** to run it.

```python
def convert_and_upload_training_data(
    ndarray, bucket, prefix, filename='gamelog.pbr'):
    import boto3
    import os
    from sagemaker.amazon.common import numpy_to_record_serializer

    # Convert Numpy array to Protobuf RecordIO format
    serializer = numpy_to_record_serializer()
    buffer = serializer(ndarray)

    # Upload to S3
    s3_object = os.path.join(prefix, 'train', filename)
    boto3.Session().resource('s3').Bucket(bucket).Object(s3_object).upload_fileobj(buffer)
    s3_path = 's3://{}/{}'.format(bucket, s3_object)
    return s3_path

bucket = 'YOUR ANALYTICS BUCKET'
prefix = 'sagemaker/randomcutforest'
s3_train_data = convert_and_upload_training_data(
    game_data.as_matrix().reshape(-1,2),
    bucket,
    prefix)
```

The data used in the lab is 2-dimension data with x and y coordinates. Therefore, redefine in 2 dimensions as follows:
```python
game_data.as_matrix().reshape(-1,2),
```

11. Training SageMaker's RCF model for the transformed data set. From the code page, copy and paste the part below and click **[▶Run]** to execute it.

```python
import boto3
import sagemaker

containers = {
    'us-west-2': '174872318107.dkr.ecr.us-west-2.amazonaws.com/randomcutforest:latest',
    'us-east-1': '382416733822.dkr.ecr.us-east-1.amazonaws.com/randomcutforest:latest',
    'us-east-2': '404615174143.dkr.ecr.us-east-2.amazonaws.com/randomcutforest:latest',
    'eu-west-1': '438346466558.dkr.ecr.eu-west-1.amazonaws.com/randomcutforest:latest'}
region_name = boto3.Session().region_name
container = containers[region_name]

session = sagemaker.Session()

# Set training job parameter
rcf = sagemaker.estimator.Estimator(
    container,
    sagemaker.get_execution_role(),
    output_path='s3://{}/{}/output'.format(bucket, prefix),
    train_instance_count=1,
    train_instance_type='ml.c5.xlarge',
    sagemaker_session=session)

# Set RCF Hyperparameter
rcf.set_hyperparameters(
    num_samples_per_tree=1000,
    num_trees=200,
    feature_dim=2)

s3_train_input = sagemaker.session.s3_input(
    s3_train_data,
    distribution='ShardedByS3Key',
    content_type='application/x-recordio-protobuf')

rcf.fit({'train': s3_train_input})
```

Let's take a moment to look at the parameters specified in the above code. First, you have specified a docker container for the RCF algorithm as follows:
```python
containers = {
    'us-west-2': '174872318107.dkr.ecr.us-west-2.amazonaws.com/randomcutforest:latest',
    'us-east-1': '382416733822.dkr.ecr.us-east-1.amazonaws.com/randomcutforest:latest',
    'us-east-2': '404615174143.dkr.ecr.us-east-2.amazonaws.com/randomcutforest:latest',
    'eu-west-1': '438346466558.dkr.ecr.eu-west-1.amazonaws.com/randomcutforest:latest'}
```

The instance type and number to execute the algorithm are as follows. If you want to learn the model more quickly, you can change the value of this part.
```python
rcf = sagemaker.estimator.Estimator(
    container,
    sagemaker.get_execution_role(),
    output_path='s3://{}/{}/output'.format(bucket, prefix),
    train_instance_count=1,
    train_instance_type='ml.c5.xlarge',
    sagemaker_session=session)
```

You see the part of the RCF algorithm that specifies the hyper parameters needed. This allows you to assign and learn each 1000 subsamples to 200 trees. A description of each parameter can be found at the following link:
https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/rcf_hyperparameters.html
```python
rcf.set_hyperparameters(
    num_samples_per_tree=1000,
    num_trees=200,
    feature_dim=2)
```

Once the pasted code has finished running, you can check the output for the completion of learning as follows. Model training is complete.
<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/39.png"></img> 
</div>

12. Now, there is really only a process to inference through a score of abnormal behavior. From the code page, copy and paste the part below and click **[▶Run]** to execute it.
```python
from sagemaker.predictor import csv_serializer, json_deserializer

rcf_inference = rcf.deploy(
    initial_instance_count=2,
    instance_type='ml.c5.2xlarge',
)

rcf_inference.content_type = 'text/csv'
rcf_inference.serializer = csv_serializer
rcf_inference.deserializer = json_deserializer
```

You can specify the type and number of instances of the inference endpoint. At this time, use 2 ml.c5.2xlarge.
```python
rcf_inference = rcf.deploy(
    initial_instance_count=2,
    instance_type='ml.c5.2xlarge',
)
```

If you are inferring a larger dataset in production environment, you can allocate a larger instance or take advantage of SageMaker Batch Transform.
https://docs.aws.amazon.com/en_us/sagemaker/latest/dg/how-it-works-batch.html

The following output confirms that the endpoint has been created.
<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/40.png"></img> 
</div>

13. Let's infer the actual data set. It would be greate to deduce the entire dataset, but this lab has a restriction that only 2 ml.c5.2xlarge instances are used for inference. Therefore, you work on inference based on about 180,000 randomly extracted data. Or you can use the data in the gamelog_sagemaker folder in the analytics bucket. Let's first look at the data set for inference. From the code page, copy and paste the part below and click **[▶Run]** to execute it.

```python
import pandas
import urllib.request

predict_file = 'predict.csv'
predict_source = 'https://s3.amazonaws.com/anhyobin-gaming/predict.csv'

urllib.request.urlretrieve(predict_source, predict_file)
predict_data = pandas.read_csv(predict_file, delimiter = ',')

predict_data.plot.scatter(
    x = 'posnewx',
    y = 'posnewy'
)
```

The graph appears as an output. As you already know, the abnormal users are included.
<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/41.png"></img> 
</div>

14. Finally, let's do the actual inference. In this lab, you regard the outliers as abnormal values for all outlier scores over the 1.5 standard deviation range from the mean value. From the code page, copy and paste the part below and click **[▶Run]** to execute it.

```python
results = rcf_inference.predict(predict_data.as_matrix().reshape(-1,2))
scores = [datum['score'] for datum in results['scores']]
predict_data['score'] = pandas.Series(scores, index=predict_data.index)

score_mean = predict_data.score.mean()
score_std = predict_data.score.std()
score_cutoff = score_mean + 1.5 * score_std

anomalies = predict_data[predict_data['score'] > score_cutoff]

anomalies.plot.scatter(
    x = 'posnewx',
    y = 'posnewy'
)
```

The output show a moving pattern showing abnormal behavior based on the model. Data is not 100% accurate because it is randomly generated data for this lab, not actual data. But it was able to apply Machine Learning and get some results.

<div align="center">
    <img src="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/contribution/anhyobin/images/42.png"></img> 
</div>

# Conclusion
You have collected and stored the data generated in AWS through the 'Identifying abnormal player behavior with Machine Learning' lab. The actual data analysis required a lot of time and money to prepare the data, but Glue could be used to create the Data Catalog and quickly process the data through ETL operations to start the actual analysis.
So if you store your data efficiently in S3, then you can easily analyze it with services such as Athena as you did in the lab.
Finally, SageMaker was able to perform complex machine learning tasks that would be difficult to build, train, and deploy model to the actual production. Since SageMaker is already pre-installed with the most common algorithms, you can apply a variety of algorithms to your own data sets similar to the Random Cut Forest algorithm in this lab.

I am deeply grateful for your continued practice.
