import streamlit as st
import requests
import os


ALB_URL = os.environ.get('ALB_URL')
API_URL = f'http://{ALB_URL}/comment'


st.set_page_config(
    page_title='Gen AI - Comment Generation',
    page_icon = 'images/aws_favi.png',
    layout = 'wide'    
)
st.title('상품 리뷰 답변 생성')
st.write('구매자 만족도를 POSITIVE, NEGATIVE, MIXED 로 구분하고, 이에 대응하는 답변을 생성해 줍니다.')
st.markdown('\n')

st.code("""원단 느낌도 좋고 색상도 훌륭합니다.어깨/팔 모양이 박시합니다.
사이즈가 확실히 줄어들기 때문에 오버사이즈 룩을 원하시면 한 사이즈 크게 주문하세요.""")
st.code("""스웨터는 절대 아니에요.도톰한 티셔츠 느낌을 줍니다.
제가 계속 쓸 수 있을지 모르겠어요.저는 작은 것을 주문했어요.""")
st.code("""이 제품은 태그 없이 도착했고 이미 보풀이 있는 것 같았습니다.소매에서 얼룩이 하나 발견되었습니다.추천하지 않습니다.""")


# Review
review_input = st.text_area('Review', '')

with st.form('comment_generation_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Loading...'):
            response = requests.post(API_URL, json={'body': review_input}) 

            data = response.json()
            st.info(f'구매자 만족도: {data["Sentiment"]}')
            st.info(f'AI가 생성한 답변: {data["Generated"]}')


