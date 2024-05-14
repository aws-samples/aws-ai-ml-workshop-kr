import streamlit as st
import bedrock as glib 

col1, col2 = st.columns(2)
image = None

with col1:
    img_file = st.file_uploader('이미지를 업로드 하세요.', type=['png', 'jpg', 'jpeg'])
    prompt = st.text_area("prompt 입력")

    if st.button("Submit", type="primary"):
        response = glib.invoke(img_file, prompt)
        image = response
    
with col2:    
    if image:
        st.image(response, caption='생성된 이미지')