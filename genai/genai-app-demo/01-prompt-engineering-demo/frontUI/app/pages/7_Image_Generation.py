import streamlit as st
import requests
import boto3
import os


ALB_URL = os.environ.get('ALB_URL')
BUCKET_NAME = os.environ.get('BUCKET_NAME')

API_URL = f'http://{ALB_URL}/image'

s3 = boto3.client('s3')

def generate_response(q_text):
    data = {
        'model_type': 'titan',
        'prompt': q_text,
        'negative_text': ''        
    }
    response = requests.post(API_URL, json=data)
    print(response.text)

    return response.text

st.set_page_config(
    page_title='Gen AI - Image Generation',
    page_icon = 'images/aws_favi.png',
    # layout = 'wide'    
)
st.title('이미지 생성')


q_input = st.text_area('Enter your text', 'cherry blossom trees around a lake', height=100)

result = []
with st.form('text_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Loading...'):
            # people under cherry blossom tree
            # data = {
            #     'model_type': 'sdxl',
            #     'prompt': q_input,
            #     'negative_prompts': [],
            #     'style_preset': 'photographic'
            # }
            data = {
                'model_type': 'titan',
                'prompt': q_input,
                'negative_text': '',
                'quality': 'premium'
            }
            response = requests.post(API_URL, json=data)
            file_name = response.text

            image_object = s3.get_object(Bucket=BUCKET_NAME, Key=f'images/{file_name}')    
            st.image(image_object['Body'].read())



