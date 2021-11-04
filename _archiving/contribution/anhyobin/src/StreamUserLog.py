from __future__ import print_function
import boto3
import json

client = boto3.client('firehose')

def lambda_handler(event, context):
    for record in event['Records']:
        if(record['eventName'] == 'MODIFY' or record['eventName'] == 'INSERT'):
            # Read data from DynamoDB Streams
            pidx = record['dynamodb']['NewImage']['pidx']['N']
            uclass = record['dynamodb']['NewImage']['uclass']['S']
            ulevel = record['dynamodb']['NewImage']['ulevel']['N']
            utimestamp = record['dynamodb']['NewImage']['utimestamp']['S']
            

            # Dict to store data
            raw_data = {}
            raw_data['pidx'] = int(pidx)
            raw_data['uclass'] = uclass
            raw_data['ulevel'] = int(ulevel)
            raw_data['utimestamp'] = utimestamp

            # Convert to JSON and Put to Kinesis Firehose
            data = json.dumps(raw_data) + '\n'
            response = client.put_record(
                DeliveryStreamName = 'stream-userprofile',
                Record = {
                    'Data' : data
                }
            )
            print(data + ' has been ingested to Firehose.')