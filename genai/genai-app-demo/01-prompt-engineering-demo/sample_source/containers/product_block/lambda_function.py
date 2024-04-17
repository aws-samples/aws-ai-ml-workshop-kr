import boto3
import json
import base64
import os

BEDROCK_REGION_NAME=os.environ.get('BEDROCK_REGION_NAME', 'us-west-2')
BUCKET_NAME = os.environ.get('BUCKET_NAME')

bedrock_runtime = boto3.client(
  service_name='bedrock-runtime',
  region_name=BEDROCK_REGION_NAME
)

s3 = boto3.client('s3')

MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'


def lambda_handler(event, context):
    print(event)

    image_name = json.loads(event['body'])['name']
    response = s3.get_object(Bucket=BUCKET_NAME, Key=image_name)
    print(response)
    image_bytes = response['Body'].read()
    base64_data = base64.b64encode(image_bytes).decode('utf-8')

    # Content-Type
    _, extension = os.path.splitext(image_name)
    if extension == '.png':
        content_type = "image/png"
    elif extension == '.jpeg':
        content_type = "image/jpeg"
    else:
        content_type = "image/jpeg"

    command = """위의 이미지는 상품에 대한 정보가 담겨있습니다.
출력은 다음 키를 이용하여 JSON 형식으로 제공하세요: suitable, info
첫째, 이미지에서 나이키 신발이 있는지 식별합니다.
둘째, 나이키 신발이 있으면 suitable 의 값을 문자열 False 로 합니다.
셋째, 나이키 신발이 없으면 suitable 의 값을 문자열 True 로 합니다.
넷째, 신발이 있으나 나이키 신발인지 식별할 수 없으면 또는 이미지에서 신발을 식별할 수 없으면 suitable 의 값을 문자열 None 으로 합니다.
다섯째, 이미지에서 신발을 식별할 수 있으면 식별된 신발의 한글 상품 정보를 info 의 값으로 제공합니다.
여섯째, 이미지에서 신발을 식별할 수 없으면 info 의 값은 문자열 None 입니다.
"""
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": content_type,
                            "data": base64_data
                        }
                    },
                    {
                        "type": "text",
                        "text": command
                    }                       
                ]
            }
        ],
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
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
