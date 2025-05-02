import sys, os
module_path = ".."
sys.path.append(os.path.abspath(module_path))

import inspect
import streamlit as st
from typing import Callable, TypeVar
from streamlit.delta_generator import DeltaGenerator
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

##################### Title ########################
st.set_page_config(page_title="GenAI-driven Analytics 💬", page_icon="💬", layout="centered") ## layout [centered or wide]
st.title("GenAI-driven Analytics 💬")
st.markdown('''- This chatbot is implemented using Amazon Bedrock Claude v3.5 Sonnet.''')
st.markdown('''
            - You can find the source code in 
            [this Github](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/20_applications/09_genai_analytics)
            ''')



from main import execution
import io

##################### Functions ########################
def display_chat_history():

    node_names = ["agent", "ask_reformulation", "code_generation_for_chart", "chart_generation", "chart_description"]
    st.session_state["history_ask"].append(st.session_state["recent_ask"])

    recent_answer = {}
    for node_name in node_names:
        print ("node_name", node_name)
        if node_name != "chart_generation": recent_answer[node_name] = st.session_state["ai_results"][node_name].get("text", "None")
        else:
            if st.session_state["ai_results"][node_name] != {}:
                recent_answer[node_name] = io.BytesIO(st.session_state["ai_results"][node_name])
            else: recent_answer[node_name] = "None"
        st.session_state["ai_results"][node_name] = {} ## reset
    st.session_state["history_answer"].append(recent_answer)

    for user, assistant in zip(st.session_state["history_ask"], st.session_state["history_answer"]):
        
        with st.chat_message("user"):
            st.write(user)
            
        tabs = st.tabs(node_names)
        with st.chat_message("assistant"):
            for tab, node_name in zip(tabs, node_names):
                with tab:
                    if node_name == "chart_generation" and assistant[node_name] != "None": st.image(assistant[node_name])
                    else: st.write(assistant[node_name])

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

####################### Initialization ###############################
# Store the initial value of widgets in session state, (session_state: global var.)
if "messages" not in st.session_state: st.session_state["messages"] = []
if "history_ask" not in st.session_state: st.session_state["history_ask"] = []

for i in range(10):
    if f"ph{i}" not in st.session_state: st.session_state[f"ph{i}"] = st.empty()
    
####################### Application ###############################
if len(st.session_state["messages"]) > 0: display_chat_history()

if user_input := st.chat_input(): # block below will begin when user inputs their ask in contatiner    
    st.chat_message("user").write(user_input)
    st.session_state["recent_ask"] = user_input
    
    # ## Use tab 
    # tab1, tab2, tab3, tab4, tab5 = st.tabs(["agent", "ask_reformulation", "code_generation_for_chart", "chart_generation", "chart_description"])
    # tabs = {
    #     "agent": tab1,
    #     "ask_reformulation": tab2,
    #     "code_generation_for_chart": tab3,
    #     "chart_generation": tab4,
    #     "chart_description": tab5
    # }
    
    with st.chat_message("assistant"):
        with st.spinner(f'Thinking...'):
            execution(user_query=user_input)
            st.write("Done")