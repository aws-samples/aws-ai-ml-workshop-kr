import streamlit as st
import chat
import logging
import sys

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("streamlit")

# title
st.set_page_config(
    page_title='Drug Discovery Agent',
    page_icon='💊',
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

with st.sidebar:
    st.title("Menu")
    
    st.markdown(
        "Implementing various types of Agents using Strands Agent SDK. "
        "For detailed code, please refer to [Github](https://github.com/hsr87/drug-discovery-agent)."
    )

    # model selection box
    # model selection box
    modelName = st.selectbox(
        '🖊️ Choose your foundation model for analysis',
        ('Claude 3.7 Sonnet', 'Claude 3.5 Sonnet', 'Claude 3.5 Haiku'), index=0
    )
    
    # extended thinking of claude 3.7 sonnet
    select_reasoning = st.checkbox('Reasoning (only Claude 3.7 Sonnet)', value=False)
    reasoningMode = 'Enable' if select_reasoning and modelName == 'Claude 3.7 Sonnet' else 'Disable'
    logger.info(f"reasoningMode: {reasoningMode}")

    chat.update(modelName, reasoningMode)
    
    clear_button = st.button("Reset Conversation", key="clear")

st.title('💊 Drug Discovery Agent')  

if clear_button is True:
    chat.initiate()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

# Display chat messages from history on app rerun
def display_chat_messages():
    """Print message history
    @returns None
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "images" in message:                
                for url in message["images"]:
                    logger.info(f"url: {url}")

                    file_name = url[url.rfind('/') + 1:]
                    st.image(url, caption=file_name, use_container_width=True)
            st.markdown(message["content"])

display_chat_messages()

# Greet user
if not st.session_state.greetings:
    with st.chat_message("assistant"):
        intro = "Thank you for using Drug Discovery Agent based on Amazon Bedrock. You can enjoy comfortable conversations."
        st.markdown(intro)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True

if clear_button or "messages" not in st.session_state:
    st.session_state.messages = []        
    st.session_state.greetings = False
    st.rerun()

    chat.clear_chat_history()
       
# Always show the chat input
if prompt := st.chat_input("Enter your message."):
    with st.chat_message("user"):  # display user message in chat message container
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})  # add user message to chat history
    prompt = prompt.replace('"', "").replace("'", "")
    logger.info(f"prompt: {prompt}")

    with st.chat_message("assistant"):
        sessionState = ""
        chat.references = []
        chat.image_url = []
        response = chat.run_multi_agent_system(prompt, "Enable", st)

    st.session_state.messages.append({"role": "assistant", "content": response})
