import streamlit as st
import requests
import boto3
import os


ALB_URL = os.environ.get('ALB_URL')
BUCKET_NAME = os.environ.get('BUCKET_NAME')

API_URL = f'http://{ALB_URL}/shortform'


s3 = boto3.client('s3')

st.set_page_config(
    page_title='Gen AI - ShortForm',
    page_icon = 'images/aws_favi.png',
    layout = 'wide'    
)
st.title('비디오 숏폼')

st.image("images/shortform.png")

uploaded_file = st.file_uploader("파일을 선택하세요", type=['mp4'])


if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    st.video(bytes_data)

    s3.upload_fileobj(
        uploaded_file,
        BUCKET_NAME,
        f'videos/{uploaded_file.name}'
    )

    data = {'name': uploaded_file.name}
    response = requests.post(API_URL, json=data)

    data = response.json()

    print(data)

    st.info(data['summary'])

    video_object = s3.get_object(Bucket=BUCKET_NAME, Key=data['shortened'])    
    st.video(video_object['Body'].read())


