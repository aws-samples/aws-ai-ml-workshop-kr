from abc import ABC
from typing import Dict
import streamlit as st

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_aws import ChatBedrock
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_core.messages import HumanMessage
from langchain_community.utilities import SQLDatabase


import pprint
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig

from .langgraph_lib import *

# Make Graph
app = make_graph()
pp = pprint.PrettyPrinter(width=200, compact=True)


def get_chat_response_text_2_sql(content: str) -> tuple[str, str, str]:
    
    container = st.empty()
    text = ""
    
    config = RunnableConfig(recursion_limit=100, configurable={"thread_id": "TODO"})
    inputs = GraphState(question=content)
    
    try:
        for output in app.stream(inputs, config=config):
            for key, value in output.items():
                text += f"\n🔹 [NODE] {key} \n"
                text += "\n"
                container.markdown(text)
                for k, v in value.items():
                    text += f"📌 {k}: {v}"
                    container.markdown(text)
                
                text += "\n"
                container.markdown(text)

    except GraphRecursionError as e:
        print(f"⚠️ Recursion limit reached: {e}")
    
    print(text)
    return text