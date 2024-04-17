import base64
import streamlit as st  # 모든 streamlit 명령은 "st" alias로 사용할 수 있습니다.
import bedrock as glib  # 로컬 라이브러리 스크립트에 대한 참조
from langchain.callbacks import StreamlitCallbackHandler
import re

##################### Functions ########################
# 'Separately' 옵션 선택 시 나오는 중간 Context를 탭 형태로 보여주는 UI -- 현재 사용하고 있지 않음
def show_context_with_tab(contexts):
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

# 'Separately' 옵션 선택 시 나오는 중간 Context를 expander 형태로 보여주는 UI
def show_context_with_expander(contexts):
    for context in contexts:
        # Contexts 내용 출력
        page_content = context.page_content
        st.markdown(page_content)
                    
        # Image, Table 이 있을 경우 파싱해 출력
        metadata = context.metadata
        category = "None"
        if "category" in context.metadata:
            category = metadata["category"]
            if category == "Table":
                text_as_html = metadata["text_as_html"]
                st.markdown(text_as_html, unsafe_allow_html=True)
            elif category == "Image":
                image_base64 = metadata["image_base64"]
                st.image(base64.b64decode(image_base64))
            else: 
                pass
                
# 'All at once' 옵션 선택 시 4개의 컬럼으로 나누어 결과 표시하는 UI
# TODO: HyDE, RagFusion 추가 논의 필요
def show_answer_with_multi_columns(answers): 
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('''### `Lexical` ''')
        st.markdown(":green[: Alpha 값이 0.0인 경우]")
        st.write(answers[0])
    with col2:
        st.markdown('''### `Semantic` ''')
        st.markdown(":green[: Alpha 값이 1.0인 경우]")
        st.write(answers[1])
    with col3:
        st.markdown('''### + `Reranker` ''')
        st.markdown(":green[Alpha 값은 왼쪽 사이드바에서 설정하신 값으로 적용됩니다.]")
        st.write(answers[2])
    with col4:
        st.markdown('''### + `Parent_docs` ''') 
        st.markdown(":green[Alpha 값은 왼쪽 사이드바에서 설정하신 값으로 적용됩니다.]")
        st.write(answers[3])

####################### Application ###############################
st.set_page_config(layout="wide")
st.title("AWS Q&A Bot with Advanced RAG!")  # page 제목

st.markdown('''- This chatbot is implemented using Amazon Bedrock Claude v3 Sonnet.''')
st.markdown('''- Integrated advanced RAG technology: **Hybrid Search, ReRanker, and Parent Document, HyDE, Rag Fusion** techniques.''')
st.markdown('''- The original data is stored in Amazon OpenSearch, and the embedding model utilizes Amazon Titan.''')
st.markdown('''
            - You can find the source code in 
            [this Github](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/20_applications/02_qa_chatbot/04_web_ui)
            ''')
# Store the initial value of widgets in session state
if "showing_option" not in st.session_state:
    st.session_state.showing_option = "Separately"
if "search_mode" not in st.session_state:
    st.session_state.search_mode = "Hybrid"
if "hyde_or_ragfusion" not in st.session_state:
    st.session_state.hyde_or_ragfusion = "None"
disabled = st.session_state.showing_option=="All at once"

with st.sidebar: # Sidebar 모델 옵션
    # st.title("Choose UI 👇")
    with st.container(height=190):
        st.radio(
            "Choose UI between 2 options:",
            ["Separately", "All at once"],
            captions = ["아래에서 설정한 파라미터 조합으로 하나의 검색 결과가 도출됩니다.", "여러 옵션들을 한 화면에서 한꺼번에 볼 수 있습니다."],
            key="showing_option",
        )
    st.title("Set parameters for your Bot 👇")

    with st.container(height=380):
        search_mode = st.radio(
            "Choose a search mode:",
            ["Lexical", "Semantic", "Hybrid"],
            captions = [
                "키워드의 일치 여부를 기반으로 답변을 생성합니다.",
                "키워드의 일치 여부보다는 문맥의 의미적 유사도에 기반해 답변을 생성합니다.", 
                "아래의 Alpha 값을 조정하여 Lexical/Semantic search의 비율을 조정합니다."
                ],
            key="search_mode",
            disabled=disabled
            )
        alpha = st.slider('Alpha value for Hybrid search', 0.0, 0.51, 1.0, disabled=st.session_state.search_mode != "Hybrid")
        # st.write("Alpha=0.0 이면 Lexical search, Alpha=1.0 이면 Semantic search")
        if search_mode == "Lexical":
            alpha = 0.0
        elif search_mode == "Semantic":
            alpha = 1.0
    
    col1, col2 = st.columns(2)
    with col1:
        reranker = st.toggle("Reranker", disabled=disabled)
    with col2:
        parent = st.toggle("Parent_docs", disabled=disabled)

    with st.container(height=230):
        hyde_or_ragfusion = st.radio(
            "Choose a RAG option:",
            ["None", "HyDE", "RAG-Fusion"],
            captions = ["blah blah", "blah blah", "blah blah blah"],
            key="hyde_or_ragfusion",
            disabled=disabled
            ) 
        hyde = hyde_or_ragfusion == "HyDE"
        ragfusion = hyde_or_ragfusion == "RAG-Fusion"

###### 1) 'Separately' 옵션 선택한 경우 ######
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
                with st.expander("Context 확인하기 ⬇️"):
                    # show_context_with_tab(contexts=msg["content"]) ## TODO: 임시적으로 주석 처리 - score 나오면 주석 해제
                    show_context_with_expander(contexts=msg["content"])
        elif msg["role"] == "assistant_column":
            # 'Separately' 옵션일 경우 multi column 으로 보여주지 않고 첫 번째 답변만 출력
            st.chat_message(msg["role"]).write(msg["content"][0]) 
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
            reranker=reranker,
            hyde = hyde,
            ragfusion = ragfusion,
            alpha = alpha
            )
        # response 로 메세지, 링크, 레퍼런스(source_documents) 받아오게 설정된 것을 변수로 저장
        answer = response[0]
        contexts = response[1] 

        # UI 출력
        st.chat_message("assistant").write(answer)
        
        with st.chat_message("assistant"): 
            with st.expander("Context 확인하기 ⬇️ "): # TODO: "정확도 별 답변 보기 ⬇️" 로 수정 필요 
                show_context_with_expander(contexts)

        # Session 메세지 저장
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.messages.append({"role": "assistant_context", "content": contexts})
        
        # Thinking을 complete로 수동으로 바꾸어 줌
        st_cb._complete_current_thought()

###### 2) 'All at once' 옵션 선택한 경우 ######
else:
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]
    # 지난 답변 출력
    for msg in st.session_state.messages:
        if msg["role"] == "assistant_column":
            answers = msg["content"]
            show_answer_with_multi_columns(answers)
        elif msg["role"] == "assistant_context": 
            pass # 'All at once' 옵션 선택 시에는 context 로그를 출력하지 않음
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
            st.markdown('''### `Lexical` ''')
            st.markdown(":green[: Alpha 값이 0.0]으로, 키워드의 정확한 일치 여부를 판단하는 Lexical search 결과입니다.")
        with col2:
            st.markdown('''### `Semantic` ''')
            st.markdown(":green[: Alpha 값이 1.0]으로, 키워드 일치 여부보다는 문맥의 의미적 유사도에 기반한 Semantic search 결과입니다.")
        with col3:
            st.markdown('''### + `Reranker` ''')
            st.markdown(":green[Alpha 값은 왼쪽 사이드바에서 설정하신 값으로 적용됩니다.]")
        with col4:
            st.markdown('''### + `Parent_docs` ''')
            st.markdown(":green[Alpha 값은 왼쪽 사이드바에서 설정하신 값으로 적용됩니다.]")
        
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
                reranker=False,
                hyde = False,
                ragfusion = False,
                alpha = 0 # Lexical
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
                parent=False, 
                reranker=False,
                hyde = False,
                ragfusion = False,
                alpha = 1.0 # Semantic
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
                reranker=True, # Add Reranker option
                hyde = False,
                ragfusion = False,
                alpha = alpha # Hybrid
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
                parent=True, # Add Parent_docs option
                reranker=True, # Add Reranker option
                hyde = False,
                ragfusion = False,
                alpha = alpha # Hybrid
                )[0]
            st.write(answer4)
            st_cb._complete_current_thought()

        # Session 메세지 저장
        answers = [answer1, answer2, answer3, answer4]
        st.session_state.messages.append({"role": "assistant_column", "content": answers})
