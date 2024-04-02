import boto3
import json
import os

BEDROCK_REGION_NAME=os.environ.get('BEDROCK_REGION_NAME', 'us-west-2')

bedrock_runtime = boto3.client(
  service_name='bedrock-runtime',
  region_name=BEDROCK_REGION_NAME
)

MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'


def lambda_handler(event, context):
    print(event)

    prompt = event['body'] 

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "top_k": 50,
        "top_p": 0.92,
        "temperature": 0.9
    }

    # Run Bedrock API
    response = bedrock_runtime.invoke_model(
        modelId=MODEL_ID,
        contentType='application/json',
        accept='application/json',
        body=json.dumps(body)
    )

    print(response)

    response_body = json.loads(response.get('body').read())
    output = response_body['content'][0]['text']
    print(output)
    
    return {
        'statusCode': 200,
        'body': output 
    }
