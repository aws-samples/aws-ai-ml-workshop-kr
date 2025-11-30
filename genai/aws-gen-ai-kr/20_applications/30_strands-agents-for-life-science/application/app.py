import streamlit as st
import chat
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("streamlit")

# Page configuration
st.set_page_config(
    page_title='생명과학 연구 AI 어시스턴트',
    page_icon='🧬',
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

with st.sidebar:
    st.title("🧬 생명과학 연구 AI 어시스턴트")

    st.markdown(
        """
        **Strands Agents & Amazon Bedrock AgentCore** 워크샵 웹 데모에 오신 것을 환영합니다!
        ---

        자세한 코드는 [Github](https://github.com/hsr87/strands-agents-for-life-science)를 참조하세요.
        """
    )

    st.markdown("---")

    # Model selection
    modelName = st.selectbox(
        '🤖 모델 선택',
        ('Claude 3.7 Sonnet', 'Claude 3.5 Sonnet', 'Claude 3.5 Haiku'),
        index=0
    )

    # Extended thinking (reasoning mode)
    select_reasoning = st.checkbox(
        '🧠 Extended Thinking 활성화 (Claude 3.7 Sonnet 전용)',
        value=False
    )
    reasoningMode = 'Enable' if select_reasoning and modelName == 'Claude 3.7 Sonnet' else 'Disable'
    logger.info(f"reasoningMode: {reasoningMode}")

    chat.update(modelName, reasoningMode)

    st.markdown("---")

    # Clear conversation button
    clear_button = st.button("🔄 대화 초기화", key="clear")

# Main title
st.title('🧬 생명과학 연구 AI 어시스턴트')

# Clear conversation if button clicked
if clear_button is True:
    chat.initiate()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

# Display chat messages from history
def display_chat_messages():
    """Display message history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

display_chat_messages()

# Greet user
if not st.session_state.greetings:
    with st.chat_message("assistant"):
        intro = """
        안녕하세요! **생명과학 연구 AI 어시스턴트**입니다.

        저는 다음과 같은 질문에 답변할 수 있습니다:

        **📚 외부 데이터베이스 검색 예시:**
        - "HER2 단백질에 대한 최신 연구 논문을 찾아주세요"

        **💾 내부 데이터베이스 검색 예시:**
        - "데이터베이스에 어떤 테이블들이 있는지 알려주세요"

        **🧬 단백질 설계 예시:**
        - "다음 항체 서열을 최적화해주세요: EVQLVETGGGLVQPGGSLRLSCAASGFTLNSYGISWVRQAPGKGPEWVS - 안정성과 결합 친화력을 향상시키는 방향으로 최적화해주세요"

        무엇을 도와드릴까요?
        """
        st.markdown(intro)
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True

# Reset conversation if clear button clicked
if clear_button or "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False
    st.rerun()
    chat.clear_chat_history()

# Chat input
if prompt := st.chat_input("질문을 입력하세요..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    prompt = prompt.replace('"', "").replace("'", "")
    logger.info(f"prompt: {prompt}")

    # Generate and display assistant response
    with st.chat_message("assistant"):
        response = chat.run_multi_agent_system(prompt, "Enable", st)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
