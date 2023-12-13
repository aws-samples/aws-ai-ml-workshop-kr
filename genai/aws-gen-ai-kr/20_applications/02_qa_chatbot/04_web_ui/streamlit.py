import streamlit as st  # 모든 streamlit 명령은 "st" alias로 사용할 수 있습니다.
import bedrock as glib  # 로컬 라이브러리 스크립트에 대한 참조
from langchain.callbacks import StreamlitCallbackHandler

st.title("💬 Knox Manage API reference")   #page 제목
index = glib.get_info()
st.subheader("Index ver: "+index, divider='blue')
st.caption("Welcome to the reference for the Knox Manage Open API. The Knox Manage Open API provides a broad set of operations and resources that: 1) User, device, organization, group management 2) Apply policies to users, groups, organizations, and devices 3) User authentication, etc.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a knox chatbot who can search the knox API documentation. How can I help you?"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 유저가 쓴 chat을 query라는 변수에 담음
query = st.chat_input("Serach documentation")
if query:
    # Session에 메세지 저장
    st.session_state.messages.append({"role": "user", "content": query})
    # UI에 출력
    st.chat_message("user").write(query)
    # Streamlit callback handler로 bedrock streaming 받아오는 컨테이너 설정
    st_cb = StreamlitCallbackHandler(st.container(), collapse_completed_thoughts=True)
    # bedrock.py의 invoke 함수 사용
    response = glib.invoke(query=query, streaming_callback=st_cb)
    # response 로 메세지, 링크, 레퍼런스(source_documents) 받아오게 설정된 것을 변수로 저장
    msg = response[0]
    link = response[1]
    ref = response[2]
    # Session 메세지 저장
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.session_state.messages.append({"role": "assistant", "content": link})
    # UI 출력
    st.chat_message("assistant").write(msg)
    st.chat_message("assistant").write(link)
    st.chat_message("assistant").write(ref)
    # Thinking을 complete로 수동으로 바꾸어 줌
    st_cb._complete_current_thought()