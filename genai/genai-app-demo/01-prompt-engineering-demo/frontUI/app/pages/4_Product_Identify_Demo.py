import streamlit as st
import requests
import boto3
import os


ALB_URL = os.environ.get('ALB_URL')
BUCKET_NAME = os.environ.get('BUCKET_NAME')

API_URL = f'http://{ALB_URL}/identify'

s3 = boto3.client('s3')

st.set_page_config(
    page_title='Gen AI - Identification',
    page_icon = 'images/aws_favi.png',
    layout = 'wide'    
)
st.title('상품 식별')


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

    if 'goods' in result and result['goods']:
        temp = []
        if 'similar' in result and result['similar']:
            similar = result['similar']
            for name, desc in similar.items():
                temp.append(f'{name}: {desc}')

        st.info(f'상품\n\n {result['goods']}\n\n브랜드: {result['brand']}\n\n추천\n\n {'\n'.join(temp)}')
    elif 'etc' in result and result['etc']:
        st.info(f'기타: {result['etc']}')
