import os
import sys
import time
import boto3
import logging
from botocore.exceptions import ClientError

class kinesis_handler():
    
    def __init__(self, region_name=None):
        
        self.kinesis_client = boto3.client('kinesis')
                
        print (f"This is a Kinesis handler.")
        
    def create_streams(self, data_streams):
        
        try:
            
            for stream in data_streams:
                     
                if stream["stream_mode"] == "PROVISIONED":
                    self.kinesis_client.create_stream(
                        StreamName=stream["name"],
                        ShardCount= stream["shard_count"],
                        StreamModeDetails={
                            'StreamMode': stream["stream_mode"] #'PROVISIONED'|'ON_DEMAND'
                        }
                    )     
                else:
                    self.kinesis_client.create_stream(
                        StreamName=stream["name"],
                        StreamModeDetails={
                            'StreamMode': stream["stream_mode"] #'PROVISIONED'|'ON_DEMAND'
                        }
                    )
                
        except ClientError as e:
            
            logging.error(e)
            
            if e.response['Error']['Code'] == 'ResourceInUseException':
                print(e.response['message'])
            else:
                print(e.response['Error']['Code'])
            
            return False

        # Wait until all streams are created
        create_stream_result = {}
        waiter = self.kinesis_client.get_waiter('stream_exists')
        for stream in data_streams:
            waiter.wait(StreamName= stream["name"])
            response = self.kinesis_client.describe_stream(StreamName=stream["name"])
            create_stream_result[stream["name"]] = response["StreamDescription"]["StreamARN"]
            
        return create_stream_result
    
    def describe_stream(self, stream_name):
        
        try:
            desc_stream = self.kinesis_client.describe_stream(
                StreamName=stream_name
            )
            
        except ClientError as e:
            logging.error(e)
            
            if e.response['Error']['Code'] == 'ResourceInUseException':
                print(e.response['message'])
            else:
                print(e.response['Error']['Code'])
            
            return False
        
        return desc_stream
    
    def increase_stream_retention_period(self, stream_name, retention_period, stream_arn):
        
        try:
            response = self.kinesis_client.increase_stream_retention_period(
                StreamName=stream_name,
                RetentionPeriodHours=int(retention_period),
                StreamARN=stream_arn
            )
            
        except ClientError as e:
            
            logging.error(e)
            
            if e.response['Error']['Code'] == 'ResourceInUseException':
                print(e.response['message'])
            else:
                print(e.response['Error']['Code'])
            
            return False
        
        print (f'Stream Retention Period was increased to {retention_period} hours.')
        return True
    
    def decrease_stream_retention_period(self, stream_name, retention_period, stream_arn):
        
        try:
            response = self.kinesis_client.decrease_stream_retention_period(
                StreamName=stream_name,
                RetentionPeriodHours=int(retention_period),
                StreamARN=stream_arn
            )

        except ClientError as e:
            
            logging.error(e)
            
            if e.response['Error']['Code'] == 'ResourceInUseException':
                print(e.response['message'])
            else:
                print(e.response['Error']['Code'])
            
            return False
        
        print (f'Stream Retention Period was decreased to {retention_period} hours.')
        return True
    
    def delete_stream(self, stream_name, consumer_deletion, stream_arn):
        
        try:
            response = self.kinesis_client.delete_stream(
                StreamName=stream_name,
                EnforceConsumerDeletion=consumer_deletion,
                StreamARN=stream_arn
            )
        
        except ClientError as e:
            logging.error(e)
            
            if e.response['Error']['Code'] == 'ResourceInUseException':
                print(e.response['message'])
            else:
                print(e.response['Error']['Code'])
            
            return False
        
        print (f'Stream "{stream_name}" was deleted successfully!.')
        return True
    
    def put_record(self, stream_name, data, partition_key):
        
        try:
            self.kinesis_client.put_record(
                StreamName=stream_name,
                Data=data,
                PartitionKey=partition_key
            )
        except ClientError as e:
            logging.error(e)
            
            if e.response['Error']['Code'] == 'ResourceInUseException':
                print(e.response['message'])
            else:
                print(e.response['Error']['Code'])
            
    def get_shard_iterator(self, stream_name):
        
        output_stream_info = self.describe_stream(
            stream_name=stream_name
        )
        shard_id = output_stream_info["StreamDescription"]["Shards"][0]["ShardId"]
        
        try:
            shard_response = self.kinesis_client.get_shard_iterator(
                StreamName=stream_name,
                ShardId=shard_id,
                ShardIteratorType="LATEST"
            )
            shardIterator = shard_response["ShardIterator"]
        
        except ClientError as e:
            logging.error(e)
            
            if e.response['Error']['Code'] == 'ResourceInUseException':
                print(e.response['message'])
            else:
                print(e.response['Error']['Code'])
            
            return False
        
        return shardIterator
    
    def get_records(self, shard_iterator):
        
        try:
            response = self.kinesis_client.get_records(
                ShardIterator=shard_iterator,
                Limit=10000, # default: 10,000
            )
        
        except ClientError as e:
            logging.error(e)
            
            if e.response['Error']['Code'] == 'ResourceInUseException':
                print(e.response['message'])
            else:
                print(e.response['Error']['Code'])
            
            return False
        
        return response
    


class kinesis_analytics_handler():
    
    def __init__(self, region_name=None):
        
        self.kinesis_analytics = boto3.client('kinesisanalytics')
                
        print (f"This is a Kinesis Analytics handler.")
        
    def create_application(self, application_name, application_code, inputs, outputs):
        
        try:
            response = self.kinesis_analytics.create_application(
                ApplicationName=application_name,
                ApplicationCode=application_code,
                Inputs=inputs,
                Outputs=outputs,
            )
        
        except ClientError as e:
            
            logging.error(e)
            
            if e.response['Error']['Code'] == 'ResourceInUseException':
                print(e.response['message'])
            else:
                print(e.response['Error']['Code'])
            
            return False
        
        print (f'kinesis application "{application_name}" was created successfully!.')
        return True
    
    def describe_application(self, application_name):
        
        try:
            response = self.kinesis_analytics.describe_application(
                ApplicationName=application_name
            )
            
        except ClientError as e:
            logging.error(e)
            
            if e.response['Error']['Code'] == 'ResourceInUseException':
                print(e.response['message'])
            else:
                print(e.response['Error']['Code'])
            
            return False
        
        return response
    
    def start_application(self, application_name):
        
        application=self.describe_application(
            application_name=application_name
        )
        input_id = application["ApplicationDetail"]["InputDescriptions"][0]["InputId"]
        
        try:
            self.kinesis_analytics.start_application(
                ApplicationName=application_name,
                InputConfigurations=[
                    {
                       "Id": input_id,
                       "InputStartingPositionConfiguration": {
                           "InputStartingPosition": "NOW"
                       }
                    }
                ]
            )
            
            # Wait until application starts running
            application=self.describe_application(
                application_name=application_name
            )
            status = application["ApplicationDetail"]["ApplicationStatus"]
            print (f"current status: {status}")
            
            sys.stdout.write('Starting ')
            while status != "RUNNING":
                sys.stdout.write('.')
                sys.stdout.flush()
                time.sleep(1)
                application=self.describe_application(
                    application_name=application_name
                )
                status = application["ApplicationDetail"]["ApplicationStatus"]
            sys.stdout.write('RUNNING')
            sys.stdout.write(os.linesep)
            
        except ClientError as e:
            logging.error(e)
            
            if e.response['Error']['Code'] == 'ResourceInUseException':
                print(e.response['message'])
            else:
                print(e.response['Error']['Code'])
            
            return False
        
        print (f'kinesis application "{application_name}" start!!')
        return True
    
    def stop_application(self, application_name):
        
        try:
            self.kinesis_analytics.stop_application(
                ApplicationName=application_name
            )
        except ClientError as e:
            logging.error(e)
            
            if e.response['Error']['Code'] == 'ResourceInUseException':
                print(e.response['message'])
            else:
                print(e.response['Error']['Code'])
    
        # Wait until application stops running
        response = self.describe_application(
            application_name=application_name
        )
        status = response["ApplicationDetail"]["ApplicationStatus"]
        sys.stdout.write('Stopping ')

        while status != "READY":
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(1)
            response = self.describe_application(
                application_name=application_name
            )
            status = response["ApplicationDetail"]["ApplicationStatus"]

        sys.stdout.write(os.linesep)
        
        print (f'STOP: kinesis application "{application_name}"')
        
    def delete_application(self, application_name):
        
        response = self.describe_application(
            application_name=application_name
        )
        
        try:
            self.kinesis_analytics.delete_application(
                ApplicationName=application_name,
                CreateTimestamp=response['ApplicationDetail']['CreateTimestamp']
            )
        
        except ClientError as e:
            logging.error(e)
            
            if e.response['Error']['Code'] == 'ResourceInUseException':
                print(e.response['message'])
            else:
                print(e.response['Error']['Code'])
        
        print (f'kinesis application "{application_name}" was deleted successfully!.')

        
