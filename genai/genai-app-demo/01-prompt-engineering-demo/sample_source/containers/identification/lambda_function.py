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
위 이미지에서 신발, 가방을 식별하고 그에 대한 설명을 상품이라는 항목으로 설명합니다.
만약 신발, 가방에 대한 식별을 할 수 없다면 이미지에 대한 설명을 기타라는 항목으로 설명합니다. 
신발, 가방의 브랜드를 알 수 있으면 그 브랜드에 대한 설명을 추가합니다.
비슷한 아디다스 제품을 추천합니다.
답변은 아래와 같은 JSON 형식으로 출력합니다.
{
    "goods": 상품 설명,
    "brand": 브랜드 설명,
    "etc": 기타 설명,
    "similar": {
        제품 이름: 제품 설명
    }
}
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
