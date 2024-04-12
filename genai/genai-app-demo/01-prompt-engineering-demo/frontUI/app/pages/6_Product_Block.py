import streamlit as st
import requests
import boto3
import os


ALB_URL = os.environ.get('ALB_URL')
BUCKET_NAME = os.environ.get('BUCKET_NAME')

API_URL = f'http://{ALB_URL}/block'

s3 = boto3.client('s3')

st.set_page_config(
    page_title='Gen AI - Product Block',
    page_icon = 'images/aws_favi.png',
    # layout = 'wide'    
)
st.title('상품 적합/부적합')
st.write('나이키 신발이라면 상품으로 올리기에 부적합, 그외 브랜드라면 적합')


uploaded_file = st.file_uploader("파일을 선택하세요", type=['png', 'jpg'])

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    st.image(bytes_data)

    s3.upload_fileobj(
        uploaded_file,
        BUCKET_NAME,
        f'images/{uploaded_file.name}'
    )

    data = {'name': f'images/{uploaded_file.name}'}
    response = requests.post(API_URL, json=data)
    print(response)
    result = response.json()

    print(result)

    if result['suitable'] == 'True':
        st.success('적합')
    elif result['suitable'] == 'False':
        st.error('부적합')
    else:
        st.warning('식별 불가')

    st.info(f'적합: {result["suitable"]}\n\nInfo: {result["info"]}')


