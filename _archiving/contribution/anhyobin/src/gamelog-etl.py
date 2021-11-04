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