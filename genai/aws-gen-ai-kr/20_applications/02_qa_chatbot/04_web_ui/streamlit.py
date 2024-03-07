import streamlit as st  # 모든 streamlit 명령은 "st" alias로 사용할 수 있습니다.
import bedrock as glib  # 로컬 라이브러리 스크립트에 대한 참조
from langchain.callbacks import StreamlitCallbackHandler

st.title("AWS Q&A Bot with Advanced RAG!")  # page 제목
st.markdown('''- This chatbot is implemented using Amazon Bedrock Claude v2.1.''')

st.markdown('''- Integrated advanced RAG technology: **Hybrid Search, ReRanker, and Parent Document** techniques.''')

st.markdown('''- The original data is stored in Amazon OpenSearch, and the embedding model utilizes Amazon Titan.''')

st.markdown(
    '''- You can find the source code in [this Github](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/20_applications/02_qa_chatbot/04_web_ui)''')
col1, col2, col3 = st.columns([2, 1, 1])
with col2:
    parent = st.toggle("Partent_docs")
with col3:
    reranker = st.toggle("Reranker")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
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
    st_cb = StreamlitCallbackHandler(
        st.container(), collapse_completed_thoughts=True)
    # bedrock.py의 invoke 함수 사용
    response = glib.invoke(query=query, streaming_callback=st_cb, parent=parent, reranker=reranker)
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
    # st.chat_message("assistant").write(ref)
    # Thinking을 complete로 수동으로 바꾸어 줌
    st_cb._complete_current_thought()
