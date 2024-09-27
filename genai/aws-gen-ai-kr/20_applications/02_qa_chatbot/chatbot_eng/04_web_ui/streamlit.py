import base64
import json
import streamlit as st  # 모든 streamlit 명령은 "st" alias로 사용할 수 있습니다.
import bedrock as glib  # 로컬 라이브러리 스크립트에 대한 참조
from langchain.callbacks import StreamlitCallbackHandler
from langchain_core.documents import Document


##################### Functions ########################

def show_context_with_tab(contexts):
    # 탭 생성
    tabs = st.tabs(["Document " + str(i+1) for i in range(len(contexts))])
    
    for i, (tab, doc) in enumerate(zip(tabs, contexts)):
        with tab:
            if isinstance(doc, Document):
                # 메타데이터 표시
                st.subheader("Metadata")
                for key, value in doc.metadata.items():
                    st.text(f"{key}: {value}")
                
                # 내용 표시
                st.subheader("Content")
                st.markdown(doc.page_content)
            else:
                st.write(f"Unexpected document type: {type(doc)}")

####################### Application ###############################
st.set_page_config(layout="wide")
st.title("AWS Q&A Bot with Advanced RAG!")  # page 제목

st.markdown('''- This chatbot is implemented using Amazon Bedrock Claude v3.5 Sonnet.''')
st.markdown('''- Integrated advanced RAG technology: **Hybrid Search, ReRanker, and Parent Document** techniques.''')
st.markdown('''- The original data is stored in Amazon OpenSearch, and the embedding model utilizes Amazon Titan v2.''')
st.markdown('''
            - You can find the source code in 
            [this Github](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/20_applications/02_qa_chatbot/04_web_ui)
            ''')
# Store the initial value of widgets in session state
if "showing_option" not in st.session_state:
    st.session_state.showing_option = "Separately"
if "search_mode" not in st.session_state:
    st.session_state.search_mode = "Hybrid search"
if "hyde_or_ragfusion" not in st.session_state:
    st.session_state.hyde_or_ragfusion = "None"
disabled = st.session_state.showing_option=="All at once"


with st.sidebar: # Sidebar 모델 옵션

    st.markdown('''### Set parameters for your Bot 👇''')

    with st.container(border=True):
        search_mode = st.radio(
            "Choose a search mode:",
            ["Lexical search", "Semantic search", "Hybrid search"],
            captions = [
                "Generates answers based on the matching of keywords.",
                "Generates answers based on the semantic similarity of the context.", 
                "Adjusts the ratio of Lexical/Semantic search by adjusting the Alpha value below."
                ],
            key="search_mode",
            disabled=disabled
            )
        alpha = st.slider('Alpha value for Hybrid search ⬇️', 0.0, 1.0, 0.51, 
                          disabled=st.session_state.search_mode != "Hybrid search",
                          help="""Alpha=0.0 means Lexical search, \nAlpha=1.0 means Semantic search."""
                          )
        if search_mode == "Lexical search":
            alpha = 0.0
        elif search_mode == "Semantic search":
            alpha = 1.0
    
    col1, col2 = st.columns(2)
    with col1:
        reranker = st.toggle("Reranker", 
                             help="""A model that re-evaluates and re-ranks the initial search results. \nIt brings more relevant results to the top by considering contextual information and query relevance.""",
                             disabled=disabled)
    with col2:
        parent = st.toggle("Parent Docs", 
                           help="""An option that displays the source of information referred to by the answer generation model when generating an answer to a query.""", 
                           disabled=disabled)

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
                with st.expander("Context Check ⬇️"):
                    show_context_with_tab(contexts=msg["content"])
                    
        elif msg["role"] == "assistant_column":
            # 'Separately' 옵션일 경우 multi column 으로 보여주지 않고 첫 번째 답변만 출력
            st.chat_message(msg["role"]).write(msg["content"][0]) 
        else:
            st.chat_message(msg["role"]).write(msg["content"])
    
    # 유저가 쓴 chat을 query라는 변수에 담음
    query = st.chat_input("Search documentation")
    

if query:
    print("Query received")
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)
    
    print("Setting up StreamlitCallbackHandler")
    st_callback = StreamlitCallbackHandler(st.container())
    
    print("Calling glib.invoke")
    try:
        result = glib.invoke(
            query=query, 
            streaming_callback=st_callback,
            parent=parent, 
            reranker=reranker,
            alpha=alpha
        )
        print(f"Result type: {type(result)}")
        print(f"Result content: {result}")
        
        answer = result[0]
        context = result[1]
        
        st.chat_message("assistant").write(answer)
        with st.chat_message("assistant"):
            with st.expander("Context Check ⬇️"):
                show_context_with_tab(context)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.messages.append({"role": "assistant_context", "content": context})

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        st.error(f"An error occurred: {str(e)}")