import sys, os
module_path = "../../.."
sys.path.append(os.path.abspath(module_path))

import inspect
import pandas as pd
import streamlit as st
from typing import Callable, TypeVar
from streamlit.delta_generator import DeltaGenerator
from src_streamlit.genai_anaysis import llm_call, genai_analyzer
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from utils import bedrock
from utils.bedrock import bedrock_model, bedrock_info, bedrock_utils
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#from langchain.callbacks import StreamlitCallbackHandler




##################### Functions ########################
def get_dataset():
    
    df = pd.read_csv("../dataset/app_power_consumption.csv")
    column_info = pd.read_csv("../dataset/column_info.csv")

    return df, column_info
    

def get_bedrock_model():

    boto3_bedrock = bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
        region=os.environ.get("AWS_DEFAULT_REGION", None),
    )

    llm_text = bedrock_model(
        model_id=bedrock_info.get_model_id(model_name="Claude-V3-5-Sonnet"),
        bedrock_client=boto3_bedrock,
        stream=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        inference_config={
            'maxTokens': 2048,
            'stopSequences': ["\n\nHuman"],
            'temperature': 0.01,
            #'topP': ...,
        }
        #additional_model_request_fields={"top_k": 200}
    )

    return llm_text

def add_history(role, content):

    message = bedrock_utils.get_message_from_string(
        role=role,
        string=content
    )
    st.session_state["messages"].append(message)

T = TypeVar("T")
def get_streamlit_cb(parent_container: DeltaGenerator):
    
    def decor(fn: Callable[..., T]) -> Callable[..., T]:
        ctx = get_script_run_ctx()

        def wrapper(*args, **kwargs) -> T:
            add_script_run_ctx(ctx=ctx)
            return fn(*args, **kwargs)

        return wrapper

    st_cb = StreamlitCallbackHandler(parent_container=parent_container)

    for name, fn in inspect.getmembers(st_cb, predicate=inspect.ismethod):
        if name.startswith("on_"):
            setattr(st_cb, name, decor(fn))

    return st_cb

def print_history():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)



####################### Application ###############################
st.set_page_config(page_title="GenAI-driven Analytics 💬", page_icon="💬", layout="centered") ## layout [centered or wide]
st.title("GenAI-driven Analytics 💬")
st.markdown('''- This chatbot is implemented using Amazon Bedrock Claude v3.5 Sonnet.''')
st.markdown('''
            - You can find the source code in 
            [this Github](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/20_applications/02_qa_chatbot/04_web_ui)
            ''')

####################### Initialization ###############################
df, column_info = get_dataset()
llm_text = get_bedrock_model()

# Store the initial value of widgets in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

if "analyzer" not in st.session_state:
    st.session_state["analyzer"] = genai_analyzer(
        llm=llm_text,
        df=df,
        column_info=column_info,
        streamlit=True,
    )

if user_input := st.chat_input():

    add_history("user", user_input)
    st.chat_message("user").write(user_input)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["agent", "ask_reformulation", "code_generation_for_chart", "chart_generation", "chart_description"])
    tabs = {
        "agent": tab1,
        "ask_reformulation": tab2,
        "code_generation_for_chart": tab3,
        "chart_generation": tab4,
        "chart_description": tab5
    }

    assistant_placeholder = st.empty()
    with assistant_placeholder.container():
        with st.chat_message("assistant"):
            st_callback = get_streamlit_cb(assistant_placeholder)
            st.session_state["assistant_placeholder"] = assistant_placeholder
            st.session_state["tabs"] = tabs
            st.session_state["analyzer"].invoke(
                ask=user_input,
                st_callback=st_callback
            )

    
    # with st.chat_message("assistant"):
        
    #     st.session_state["chat_container"] = st.empty()
    #     #st.session_state["chat_container"] = st.session_state["chat_container"].container()
        
    #     st_callback = get_streamlit_cb(st.session_state["chat_container"])

    #     st.session_state["analyzer"].invoke(
    #         ask=user_input,
    #         session_state=st.session_state,
    #         st_callback=st_callback
    #     )



   
                

#         #answer = st.write_stream(stream)
        
#         #response["chat_container"] = chat_container
#         #outputparser(**response)
#         #print (response)


# #         stream_response = st.session_state["chain"].stream(
# #             {"question": user_input}
# #         )  # 문서에 대한 질의
# #         ai_answer = ""
# #         for chunk in stream_response:
# #             ai_answer += chunk
# #             chat_container.markdown(ai_answer)
# #         add_history("ai", ai_answer)
