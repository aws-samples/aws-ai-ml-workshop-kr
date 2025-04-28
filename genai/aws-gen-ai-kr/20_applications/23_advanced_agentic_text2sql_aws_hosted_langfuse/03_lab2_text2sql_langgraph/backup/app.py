import random
from typing import List, Tuple, Union

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import src.chat_service as chat_svc


def set_page_config() -> None:
    """
    Set the Streamlit page configuration.
    """
    st.set_page_config(page_title="🤖 Text2SQL with Bedrock", layout="wide")
    st.title("🤖 Text2SQL with Bedrock")


def init_chat_data() -> None:
    """
    Reset the chat session and initialize a new conversation chain.
    """
init_message = {
        "role": "assistant",
        "content": "안녕하세요! 무엇을 도와드릴까요?",
    }

st.session_state.messages = []
st.session_state["langchain_messages"] = []
st.session_state.messages.append(init_message)




def print_formatted_text(text):
    st.markdown(f"""
    <div style='white-space: normal;'>
    {text}
    </div>
    """, unsafe_allow_html=True)
    

def display_history_messages() -> None:
    """
    Display chat messages and uploaded images in the Streamlit app.
    """
    for message in st.session_state.messages:
        message_role = message["role"]
        with st.chat_message(message_role):
            message_content = message["content"]
            print("#Display history")
            print(message_content)
            # print_formatted_text(message_content)
            st.markdown(message_content)


def main() -> None:
    """
    Main function to run the Streamlit app.
    """
    # Set page config
    set_page_config()

    # Initialize chat data
    if "messages" not in st.session_state:
        init_chat_data()

    # Generate a unique widget key only once
    if "widget_key" not in st.session_state:
        st.session_state["widget_key"] = str(random.randint(1, 1000000))


    # Get user input message
    content = st.chat_input()

    # Set new chat button
    # st.sidebar.button("Start New Chat", on_click=init_chat_data, type="primary")

    # Display all history messages
    display_history_messages()

    # Store user message
    if content:
        st.session_state.messages.append({"role": "user", "content": content})
        with st.chat_message("user"):
            st.markdown(content)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        # Get response
        with st.chat_message("assistant"):
            response = chat_svc.get_chat_response_text_2_sql(content=content)

        # Store LLM generated responses
        print("# response")
        print(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
        st.rerun()


if __name__ == "__main__":
    main()
