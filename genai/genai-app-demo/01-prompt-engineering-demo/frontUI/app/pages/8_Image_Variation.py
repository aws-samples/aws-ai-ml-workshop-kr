import streamlit as st
import requests
import boto3
import os


ALB_URL = os.environ.get('ALB_URL')
BUCKET_NAME = os.environ.get('BUCKET_NAME')

API_URL = f'http://{ALB_URL}/variation'

s3 = boto3.client('s3')

st.set_page_config(
    page_title='Gen AI - Image Variation',
    page_icon = 'images/aws_favi.png',
    # layout = 'wide'    
)
st.title('이미지 변형')
st.write('소스 이미지를 변형하여 이미지를 수정합니다')

uploaded_file = st.file_uploader("파일을 선택하세요", type=['png', 'jpg'])
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    st.image(bytes_data)

q_input = st.text_area('Enter your text', '', height=100)

with st.form('submit_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Loading...'):
                s3.upload_fileobj(
                    uploaded_file,
                    BUCKET_NAME,
                    f'images/{uploaded_file.name}'
                )

                data = {
                    'name': f'images/{uploaded_file.name}',
                    'prompt': q_input,
                    'negative_text': '',
                    'quality': 'premium'
                }
                response = requests.post(API_URL, json=data)
                print(response)
                result = response.json()

                print(result)
                col0, col1 = st.columns(2)
                image_object_0 = s3.get_object(Bucket=BUCKET_NAME, Key=f'images/{result[0]}')    
                image_object_1 = s3.get_object(Bucket=BUCKET_NAME, Key=f'images/{result[1]}')    
                col0.image(image_object_0['Body'].read())
                col1.image(image_object_1['Body'].read())

                image_object_2 = s3.get_object(Bucket=BUCKET_NAME, Key=f'images/{result[2]}')
                st.image(image_object_2['Body'].read()) 