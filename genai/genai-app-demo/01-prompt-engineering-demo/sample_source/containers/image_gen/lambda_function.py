import boto3
import json
import base64
import os

from io import BytesIO


BEDROCK_REGION_NAME=os.environ.get('BEDROCK_REGION_NAME', 'us-west-2')
BUCKET_NAME = os.environ.get('BUCKET_NAME')

bedrock_runtime = boto3.client(
  service_name='bedrock-runtime',
  region_name=BEDROCK_REGION_NAME
)

s3 = boto3.client('s3')

SDXL_MODEL_ID = 'stability.stable-diffusion-xl-v1'
TITAN_MODEL_ID = 'amazon.titan-image-generator-v1'


def titan_generate_image(prompt, negative_text, quality):
    image_name = 'rendered_image.png'

    body = {
        'textToImageParams': {
            'text': prompt
        },
        'taskType': 'TEXT_IMAGE',
        'imageGenerationConfig': {
            'cfgScale': 8,
            'seed': 0,
            'quality': 'premium',
            'width': 512,
            'height': 512,
            'numberOfImages': 1,
            'quality': quality
        }
    }

    if negative_text:
        body['textToImageParams']['negativeText'] = negative_text

    response = bedrock_runtime.invoke_model(
        modelId=TITAN_MODEL_ID,
        contentType='application/json',
        accept='application/json',
        body=json.dumps(body)
    )

    print(response)

    response_body = json.loads(response.get('body').read())
    finish_reason = response_body.get("error")

    if finish_reason is not None:
        raise Exception(finish_reason)

    base64_str = response_body.get("images")[0]
    decoded_bytes = BytesIO(base64.b64decode(base64_str))

    return image_name, decoded_bytes


def sdxl_generate_image(prompt, negative_prompts, style_preset):
    image_name = 'rendered_image.jpg'

    body = {
        'text_prompts': (
            [{'text': prompt, 'weight': 1.0}]
            + [{'text': negprompt, 'weight': -1.0} for negprompt in negative_prompts]
        ),
        'cfg_scale': 5,
        'seed': 5450,
        'steps': 70,
        'width': 512, 
        'height': 512,
        'style_preset': style_preset
    }

    response = bedrock_runtime.invoke_model(
        modelId=SDXL_MODEL_ID,
        contentType='application/json',
        accept='application/json',
        body=json.dumps(body)
    )
    response_body = json.loads(response.get('body').read())

    print(response_body['result'])

    base64_str = response_body['artifacts'][0].get('base64')
    decoded_bytes = BytesIO(base64.b64decode(base64_str))
    # img = Image.open(decoded_bytes)
    # img.save(f'/tmp/{image_name}')
    # img.save(image_name)

    return image_name, decoded_bytes


def lambda_handler(event, context):
    print(event)

    model_type = json.loads(event['body'])['model_type']
    prompt = json.loads(event['body'])['prompt']

    if model_type == 'sdxl':
        negative_prompts = json.loads(event['body'])['negative_prompts']
        style_preset = json.loads(event['body'])['style_preset']
        image_name, image_obj = sdxl_generate_image(prompt, negative_prompts, style_preset)
    else:
        negative_text = json.loads(event['body'])['negative_text']
        quality = json.loads(event['body'])['quality']
        image_name, image_obj = titan_generate_image(prompt, negative_text, quality)

    s3.upload_fileobj(image_obj, BUCKET_NAME, f'images/{image_name}') 

    return {
        'statusCode': 200,
        'body': image_name
    }


if __name__ == '__main__':
    # prompt = 'Dog in a forest'
    prompt = 'a boy wearing a nike shoes is playing basketball in a backyard with his friends.'
    negative_prompts = [
        # 'poorly rendered',
        # 'poor background details',
        # 'bad quality',
        # 'bad detail',
        # 'blurry-image',
        # 'bad contrast',
        # 'bad anatomy',
        # 'duplicate',
        # 'watermark',
        # 'extra detail',
        # 'chaotic distribution of objects',
        # 'distortion',
        # 'bad detail facial details'
    ]

    # 3d-model analog-film anime cinematic comic-book digital-art enhance fantasy-art isometric line-art low-poly modeling-compound neon-punk origami photographic pixel-art tile-texture
    style_preset = 'photographic' 

    # sdxl_generate_image(prompt, negative_prompts, style_preset)
    # titan_generate_image(prompt, negative_text='')

    event = {
        'body': '{"model_type": "titan", "prompt": "a boy with a rain coat", "negative_text": ""}'
    }

    lambda_handler(event, None)
