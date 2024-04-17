import streamlit as st
import requests
import os


ALB_URL = os.environ.get('ALB_URL')
API_URL = f'http://{ALB_URL}/prompt'


def generate_response(q_text):
    data = {'body': q_text}
    response = requests.post(API_URL, json=data)
    print(response.text)

    return response.text


st.set_page_config(
    page_title='Gen AI - Text',
    page_icon = 'images/aws_favi.png',
    # layout = 'wide'    
)

st.title('무엇이든 물어보세요.')

q_input = st.text_area('Enter your text', '', height=200)

result = []
with st.form('text_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Loading...'):
            response = generate_response(q_input)
            result.append(response)

if len(result):
    st.info(response)
