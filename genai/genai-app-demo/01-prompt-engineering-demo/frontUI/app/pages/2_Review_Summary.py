import streamlit as st
import requests
import os


ALB_URL = os.environ.get('ALB_URL')
API_URL = f'http://{ALB_URL}/summary'


def generate_response(q_text):
    data = {'body': q_text}
    response = requests.post(API_URL, json=data)

    return response.text


st.set_page_config(
    page_title='Gen AI - Summarization',
    page_icon = 'images/aws_favi.png',
    # layout = 'wide'
)
st.title('상품 리뷰 요약')
st.write('판매자에게 유용한 정보를 색상, 핏, 소재, 세탁, 가격으로 요약합니다.')

q_input = st.text_area('Enter reviews', '', height=200)

result = []
with st.form('review_summary_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Loading...'):
            response = generate_response(q_input)
            result.append(response)

if len(result):
    st.info(response)

st.markdown('\n')
st.markdown('### 후기')
st.code("""
1. 핏이 좋고 편안합니다.하지만 세탁 후 색이 번졌어요.그리고 권장대로 세탁했어요.
이제 사방에 커다란 분홍색 얼룩이 생겼어요.
2. 너무 편안하고 귀엽고 스타일리시합니다.마음에 들어요!
3. 이 스웨터가 마음에 들어요.소재는 훌륭하지만 조금 짧았습니다.
4. 분명히 위험하다는 건 알았지만 다른 사람들에게서 봤을 때 위글 공간이 조금 더 있을지도 모른다고 생각했지만 XL이 생겼고 예상대로 더 오버사이즈였으면 좋겠어요.
그러니 덩치가 큰 제 딸들에게는 생각처럼 오버사이즈가 아니라는 걸 명심하세요!아직 엄청 귀엽고 부드러워요. 하지만 이거 자연 건조할게요!!!
5. 마음에 들어요, 멋진 오버사이즈 핏, 귀여운 색상!!
""")
