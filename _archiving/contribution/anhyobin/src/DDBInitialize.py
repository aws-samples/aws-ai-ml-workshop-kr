import json
import csv
import random
import boto3
from botocore.vendored import requests

def lambda_handler(event, context):
    try:
        # Set user pidx from S3
        s3 = boto3.resource('s3')
        bucket = s3.Bucket('gamingonaws2018')
        obj = bucket.Object(key = 'userList.csv')
        
        response = obj.get()
        users = response['Body'].read().split()
        
        # Set DynamoDB
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('UserProfile')
        
        uclass = ['warrior', 'mage', 'healer']
        
        if event['RequestType'] == 'Delete':
            print 'Send response to CFN.'
            send_response(event, context, "SUCCESS", {"Message": "CFN deleted!"})
        else:
            for user in users:
                response = table.put_item(
                    Item = {
                        'pidx': int(user),
                        'ulevel': 1,
                        'uclass': random.choice(uclass),
                        'utimestamp': '2000-01-01 00:00:00.000000'
                    }
                )
            print 'Send response to CFN.'
            send_response(event, context, "SUCCESS",  {"Message": "CFN created!"})
    
        print 'End of Lambda function.'
    except:
        send_response(event, context, "FAILED", {"Message": "Lambda failed!"})
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