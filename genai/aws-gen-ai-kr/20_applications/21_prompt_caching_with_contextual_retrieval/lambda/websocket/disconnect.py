import boto3
import os
import json

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['CONNECTIONS_TABLE'])


def handler(event, context):
    # Remove connection ID from DynamoDB
    connection_id = event['requestContext']['connectionId']

    table.delete_item(Key={
        'connectionId': connection_id
    })

    return {
        'statusCode': 200,
        'body': 'Disconnected'
    }
