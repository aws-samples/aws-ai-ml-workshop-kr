import streamlit as st
import requests
import boto3
import os


ALB_URL = os.environ.get('ALB_URL')
BUCKET_NAME = os.environ.get('BUCKET_NAME')

API_URL = f'http://{ALB_URL}/replace'

s3 = boto3.client('s3')

st.set_page_config(
    page_title='Gen AI - Image Variation',
    page_icon = 'images/aws_favi.png',
    # layout = 'wide'    
)
st.title('이미지 교체')
st.write('주변 배경과 일치하도록 변경하여 이미지를 수정합니다.')

uploaded_file = st.file_uploader("파일을 선택하세요", type=['png', 'jpg'])
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    st.image(bytes_data)

m_input = st.text_area( 
    '이미지에서 남기고 싶은 오브젝트를 서술합니다 예) car, phone, bag',
    '', 
    # height=100
)
q_input = st.text_area(
    '남기고 싶은 오프젝트 이외에 배경에 대해서 정의합니다', 
    '', 
    # height=100
)

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
                    'mask_prompt': m_input
                }
                response = requests.post(API_URL, json=data)
                result = response.text

                print(result)
                image_object = s3.get_object(Bucket=BUCKET_NAME, Key=f'images/{result}')
                st.image(image_object['Body'].read()) 

