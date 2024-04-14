import streamlit as st  # 모든 streamlit 명령은 "st" alias로 사용할 수 있습니다.
import bedrock as glib  # 로컬 라이브러리 스크립트에 대한 참조
from langchain.callbacks import StreamlitCallbackHandler

def context_showing_tab(contexts):
    tab_titles = []
    tab_contents = {}
    for i, context in enumerate(contexts):
        title = str(context[0])
        tab_titles.append(title)
        tab_contents[title] = context[1][0]
    tabs = st.tabs(tab_titles)
    for i, tab in enumerate(tabs):
        with tab:
            st.header(tab_titles[i])
            st.write(tab_contents[tab_titles[i]])

def multi_answer_column(answers):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('''### option 1 ''')
        st.write(answers[0])
    with col2:
        st.markdown('''### option 2 ''')
        st.write(answers[1])
    with col3:
        st.markdown('''### option 3 ''')
        st.write(answers[2])
    with col4:
        st.markdown('''### option 4 ''')
        st.write(answers[3])

st.set_page_config(layout="wide")
st.title("AWS Q&A Bot with Advanced RAG!")  # page 제목

st.markdown('''- This chatbot is implemented using Amazon Bedrock Claude v2.1.''')
st.markdown('''- Integrated advanced RAG technology: **Hybrid Search, ReRanker, and Parent Document** techniques.''')
st.markdown('''- The original data is stored in Amazon OpenSearch, and the embedding model utilizes Amazon Titan.''')
st.markdown('''
            - You can find the source code in 
            [this Github](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/20_applications/02_qa_chatbot/04_web_ui)
            ''')
# Store the initial value of widgets in session state
if "showing_option" not in st.session_state:
    st.session_state.showing_option = "Separately"

with st.sidebar: # Sidebar 모델 옵션
    with st.container(height=170):
        st.radio(
            "Set showing method 👇",
            ["Separately", "All at once"],
            captions = ["blah blah", "blah blah blah"],
            key="showing_option",
        )

    st.title("Set parameter for your Bot 👇")

    # semantic = st.toggle("Semantic", disabled=st.session_state.showing_option=="All at once")
    # lexical = st.toggle("Lexical", disabled=st.session_state.showing_option=="All at once")
    
    # hybrid = st.slider('Alpha value of Hybrid Search: lexical(0.0) / semantic(1.0)', 0.0, 1.0, 0.5)
    
    reranker = st.toggle("Reranker", disabled=st.session_state.showing_option=="All at once")
    
    # ragfusion = st.toggle("RAG-Fusion", disabled=st.session_state.showing_option=="All at once")
    # hyde = st.toggle("HyDE", disabled=st.session_state.showing_option=="All at once")
    
    parent = st.toggle("Parent_docs", disabled=st.session_state.showing_option=="All at once")

### 1) 'Separately' 옵션 선택한 경우 ###
if st.session_state.showing_option == "Separately":
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]
    # 지난 답변 출력
    for msg in st.session_state.messages:
        # 지난 답변에 대한 컨텍스트 출력
        if msg["role"] == "assistant_context": 
            with st.chat_message("assistant"):
                with st.expander("정확도 별 답변 보기 ⬇️"):
                    context_showing_tab(contexts=msg["content"])
        else:
            st.chat_message(msg["role"]).write(msg["content"])
    
    # 유저가 쓴 chat을 query라는 변수에 담음
    query = st.chat_input("Search documentation")
    if query:
        # Session에 메세지 저장
        st.session_state.messages.append({"role": "user", "content": query})
        
        # UI에 출력
        st.chat_message("user").write(query)
        
        # Streamlit callback handler로 bedrock streaming 받아오는 컨테이너 설정
        st_cb = StreamlitCallbackHandler(
            st.chat_message("assistant"), 
            collapse_completed_thoughts=True
            )
        # bedrock.py의 invoke 함수 사용
        response = glib.invoke(
            query=query, 
            streaming_callback=st_cb, 
            parent=parent, 
            reranker=reranker
            )
        # response 로 메세지, 링크, 레퍼런스(source_documents) 받아오게 설정된 것을 변수로 저장
        answer = response[0]
        contexts1 = response[1] # semantic
        contexts2 = response[2] # lexical
        contexts3 = response[3] # reranker
        contexts4 = response[4] # similar_docs

        # UI 출력
        st.chat_message("assistant").write(answer)
        
        with st.chat_message("assistant"): 
            with st.expander("정확도 별 답변 보기 ⬇️"): # 정확도 별 답변 보기 (semantic)
                context_showing_tab(contexts1)
                
        # with st.chat_message("assistant"): 
        #     with st.expander("정확도 별 답변 보기 (lexical) ⬇️"):
        #         context_showing_tab(contexts2)

        # with st.chat_message("assistant"): 
        #     with st.expander("정확도 별 답변 보기 (reranker) ⬇️"):
        #         context_showing_tab(contexts3)

        # Session 메세지 저장
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.messages.append({"role": "assistant_context", "content": contexts1})
        # st.session_state.messages.append({"role": "assistant_context", "content": contexts2})
        # st.session_state.messages.append({"role": "assistant_context", "content": contexts3})
        
        # Thinking을 complete로 수동으로 바꾸어 줌
        st_cb._complete_current_thought()

### 2) 'All at once' 옵션 선택한 경우 ###
else:
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]
    # 지난 답변 출력
    for msg in st.session_state.messages:
        if msg["role"] == "assistant_column":
            answers = msg["content"]
            multi_answer_column(answers)
        else:
            st.chat_message(msg["role"]).write(msg["content"])
    
    # 유저가 쓴 chat을 query라는 변수에 담음
    query = st.chat_input("Search documentation")
    if query:
        # Session에 메세지 저장
        st.session_state.messages.append({"role": "user", "content": query})
        
        # UI에 출력
        st.chat_message("user").write(query)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('''### option 1 ''')
        with col2:
            st.markdown('''### option 2 ''')
        with col3:
            st.markdown('''### option 3 ''')
        with col4:
            st.markdown('''### option 4 ''')
        
        with col1:
            # Streamlit callback handler로 bedrock streaming 받아오는 컨테이너 설정
            st_cb = StreamlitCallbackHandler(
                st.chat_message("assistant"), 
                collapse_completed_thoughts=True
                )
            answer1 = glib.invoke(
                query=query, 
                streaming_callback=st_cb, 
                parent=False, 
                reranker=False
                )[0]
            st.write(answer1)
            st_cb._complete_current_thought() # Thinking을 complete로 수동으로 바꾸어 줌
        with col2:
            st_cb = StreamlitCallbackHandler(
                st.chat_message("assistant"), 
                collapse_completed_thoughts=True
                )
            answer2 = glib.invoke(
                query=query, 
                streaming_callback=st_cb, 
                parent=True, 
                reranker=False
                )[0]
            st.write(answer2)
            st_cb._complete_current_thought() 
        with col3:
            st_cb = StreamlitCallbackHandler(
                st.chat_message("assistant"), 
                collapse_completed_thoughts=True
                )
            answer3 = glib.invoke(
                query=query, 
                streaming_callback=st_cb, 
                parent=False, 
                reranker=True
                )[0]
            st.write(answer3)
            st_cb._complete_current_thought() 
        with col4:
            st_cb = StreamlitCallbackHandler(
                st.chat_message("assistant"), 
                collapse_completed_thoughts=True
            )
            answer4 = glib.invoke(
                query=query, 
                streaming_callback=st_cb, 
                parent=True, 
                reranker=True
                )[0]
            st.write(answer4)
            st_cb._complete_current_thought()

        # Session 메세지 저장
        answer = [answer1, answer2, answer3, answer4]
        st.session_state.messages.append({"role": "assistant_column", "content": answer})
        