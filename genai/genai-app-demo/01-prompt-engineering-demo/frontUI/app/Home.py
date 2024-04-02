import streamlit as st 

st.set_page_config(
    page_title="AWS Gen AI",
    page_icon = "images/aws_favi.png",
    layout = "wide"
)

home_title = "AWS Gen AI"
home_introduction = "안녕하세요. AWS Gen AI 워크샾 데모 입니다."

st.markdown(
    "<style>#MainMenu{visibility:hidden;}</style>",
    unsafe_allow_html=True
)

st.markdown(f"# {home_title} <span style=color:#2E9BF5><font size=5>Demo</font></span>",unsafe_allow_html=True)

st.image("images/aws_thumb.jpg")

st.markdown("\n")
st.markdown("#### Demo")
st.write(home_introduction)

st.image("images/architecture.png")

