import boto3
import os
import json

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['CONNECTIONS_TABLE'])


def handler(event, context):
    # Store connection ID in DynamoDB
    connection_id = event['requestContext']['connectionId']

    table.put_item(Item={
        'connectionId': connection_id
    })

    return {
        'statusCode': 200,
        'body': 'Connected'
    }
