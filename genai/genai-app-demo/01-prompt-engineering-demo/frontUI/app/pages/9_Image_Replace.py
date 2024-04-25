import streamlit as st
import requests
import boto3
import os
import io

from streamlit_cropper import st_cropper
from PIL import Image, ImageDraw


ALB_URL = os.environ.get('ALB_URL')
BUCKET_NAME = os.environ.get('BUCKET_NAME')

API_URL = f'http://{ALB_URL}/replace'

s3 = boto3.client('s3')

st.set_page_config(
    page_title='Gen AI - Image Replace',
    page_icon = 'images/aws_favi.png',
    # layout = 'wide'    
)
st.title('이미지 교체')
st.write('주변 배경과 일치하도록 변경하여 이미지를 수정합니다.')

if "mask_enable" not in st.session_state:
    st.session_state.mask_enable = False

uploaded_file = st.file_uploader("파일을 선택하세요", type=['png', 'jpg'])
if uploaded_file is not None:
    # print(st.session_state.mask_enable)
    st.checkbox("이미지 마스크 지정", key="mask_enable")
    if st.session_state.mask_enable:
        img = Image.open(uploaded_file)
        width, height = img.size

        cropped_box = st_cropper(
            img,
            realtime_update=True,
            box_color='#0000FF',
            aspect_ratio=None,
            return_type='box'
        )

        left = cropped_box['left']
        top = cropped_box['top']
        right = left + cropped_box['width']
        bottom = top + cropped_box['height']
        shape = (left, top, right, bottom) 

        masked_image = Image.new('RGB', (width, height), color=(255, 255, 255))
        temp_image = ImageDraw.Draw(masked_image)
        temp_image.rectangle(shape, fill=(0, 0, 0))

        # st.image(masked_image)

        st.write('이미지에서 남기고 싶은 오브젝트')
        cropped_image = img.crop(shape)
        _ = cropped_image.thumbnail((150,150))
        st.image(cropped_image)
    else:
        bytes_data = uploaded_file.getvalue()
        st.image(bytes_data)

        m_input = st.text_area( 
            '이미지에서 남기고 싶은 오브젝트를 지정합니다 예) car, phone, bag',
            '', 
            # height=100
        )

    q_input = st.text_area(
        '지정한 오프젝트가 보일 배경에 대해서 정의합니다', 
        '', 
        # height=100
    )

with st.form('submit_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Loading...'):
            uploaded_file.seek(0)
            s3.upload_fileobj(
                uploaded_file,
                BUCKET_NAME,
                f'images/{uploaded_file.name}'
            )

            if st.session_state.mask_enable:
                in_mem_file = io.BytesIO()
                masked_image.save(in_mem_file, format='PNG')
                in_mem_file.seek(0)

                file_name, ext = os.path.splitext(uploaded_file.name)
                masked_image_name = f'images/masked_{file_name}.png'
                s3.upload_fileobj(
                    in_mem_file,
                    BUCKET_NAME,
                    masked_image_name
                )

                data = {
                    'name': f'images/{uploaded_file.name}',
                    'prompt': q_input,
                    'mask_image': masked_image_name
                }
            else:
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

