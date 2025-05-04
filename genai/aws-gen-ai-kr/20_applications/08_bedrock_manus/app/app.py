import sys, os
module_path = ".."
sys.path.append(os.path.abspath(module_path))

import inspect
import streamlit as st
from typing import Callable, TypeVar
from streamlit.delta_generator import DeltaGenerator
from src.config.agents import AGENT_LLM_MAP
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

##################### Title ########################
st.set_page_config(page_title="GenAI-driven Analytics 💬", page_icon="💬", layout="wide")
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
    node_names = ["coordinator", "planner", "supervisor", "coder", "reporter"]
    node_descriptions = {
        "coordinator": "전체 프로세스 조정 및 최종 응답 생성",
        "planner": "분석 계획 수립 및 작업 분배",
        "supervisor": "코드 및 결과물 검증",
        "coder": "데이터 처리 및 시각화 코드 작성",
        "reporter": "분석 결과 해석 및 보고서 작성"
    }
    
    st.session_state["history_ask"].append(st.session_state["recent_ask"])

    recent_answer = {}
    for node_name in node_names:
        print("node_name", node_name)
        if node_name != "chart_generation": 
            recent_answer[node_name] = st.session_state["ai_results"][node_name].get("text", "None")
        else:
            if st.session_state["ai_results"][node_name] != {}:
                recent_answer[node_name] = io.BytesIO(st.session_state["ai_results"][node_name])
            else: 
                recent_answer[node_name] = "None"
        st.session_state["ai_results"][node_name] = {} ## reset
    st.session_state["history_answer"].append(recent_answer)

    for i, (user, assistant) in enumerate(zip(st.session_state["history_ask"], st.session_state["history_answer"])):
        
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.write(user)
        
        # 응답 표시 - 전체 화면으로
        with st.chat_message("assistant"):
            # 가장 중요한 결과를 먼저 표시
            st.write(assistant["coordinator"])
            
            # 구분선 추가
            st.divider()
            
            st.subheader("Process Details")
            
            # 각 에이전트의 결과를 순차적으로 표시
            for node_name in node_names:
                if node_name != "coordinator":  # coordinator는 이미 표시했으므로 제외
                    with st.expander(f"🤖 {node_name.upper()}: {node_descriptions[node_name]}", expanded=False):
                        if node_name == "chart_generation" and assistant[node_name] != "None":
                            st.image(assistant[node_name])
                        else:
                            st.write(assistant[node_name])

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
# Store the initial value of widgets in session state
if "messages" not in st.session_state: st.session_state["messages"] = []
if "history_ask" not in st.session_state: st.session_state["history_ask"] = []
if "history_answer" not in st.session_state: st.session_state["history_answer"] = []
if "ai_results" not in st.session_state: st.session_state["ai_results"] = {"coordinator": {}, "planner": {}, "supervisor": {}, "coder": {}, "reporter": {}}
if "current_agent" not in st.session_state: st.session_state["current_agent"] = ""
    
####################### Application ###############################
if len(st.session_state["history_ask"]) > 0: 
    display_chat_history()

if user_input := st.chat_input(): # 사용자 입력 받기
    st.chat_message("user").write(user_input)
    st.session_state["recent_ask"] = user_input
    
    node_names = ["coordinator", "planner", "supervisor", "coder", "reporter"]
    tool_node_names = ["coder", "reporter"]
    node_descriptions = {
        "coordinator": "전체 프로세스 조정 및 최종 응답 생성",
        "planner": "분석 계획 수립 및 작업 분배",
        "supervisor": "코드 및 결과물 검증",
        "coder": "데이터 처리 및 시각화 코드 작성",
        "reporter": "분석 결과 해석 및 보고서 작성"
    }
    
    # 응답 프로세스 시작
    with st.chat_message("assistant"):
        # 초기 메시지
        main_response = st.empty()
        main_response.write("분석 작업을 시작합니다...")        
        # 각 단계별 진행 상황을 표시할 expander 생성
        if "process_containers" not in st.session_state:
            st.session_state["process_containers"] = {}
            st.session_state["tool_containers"] = {}
            st.session_state["reasoning_containers"] = {}
            
        for node_name in node_names:
            with st.expander(f"🔄 {node_name.upper()}: {node_descriptions[node_name]}", expanded=True):
                
                # Create two columns: left for Agent message, right for Reasoning and Tool
                left_col, right_col = st.columns([1, 1])
                
                # Left column - Agent message
                with left_col:
                    st.markdown(f"💬 Agent message:")
                    st.session_state["process_containers"][node_name] = st.empty()
                    st.session_state["process_containers"][node_name].info(f"Waiting...")
                
                # Right column - Reasoning and Tool
                with right_col:
                    if AGENT_LLM_MAP[node_name] == "reasoning":
                        st.markdown(f"🧠 Reasoning:")
                        st.session_state["reasoning_containers"][node_name] = st.empty()
                        st.session_state["reasoning_containers"][node_name].info(f"Reasoning not used yet")
                        st.markdown("---")
                    
                    # 에이전트가 사용하는 툴 결과를 표시할 컨테이너
                    if node_name in tool_node_names:
                        st.markdown(f"🔧 Tool message:")
                        st.markdown(f"  - Input:")
                        st.session_state["tool_containers"][node_name] = {}
                        st.session_state["tool_containers"][node_name]["input"] = st.empty()
                        st.markdown(f"  - Output:")
                        st.session_state["tool_containers"][node_name]["output"] = st.empty()
                        st.session_state["tool_containers"][node_name]["input"].info(f"Tool not used yet")
                        st.session_state["tool_containers"][node_name]["output"].info(f"Tool not used yet")
                
                st.markdown("---")  # Divider between agent sections
        
        # 차트를 위한 컨테이너 (필요한 경우)
        chart_container = st.empty()
        
        with st.spinner('분석 중...'):
            # 실행 및 결과 처리
            exe_results = execution(user_query=user_input)

            last_message = ""
            for history in exe_results["history"]:
                st.session_state["process_containers"][history["agent"]].write(history["message"])
                last_message = history["message"]
            
            ## 각 에이전트의 결과 업데이트
            #for node_name in node_names:
            #    result_text = st.session_state["ai_results"].get(node_name, {}).get("text", "처리 완료")
            #    st.session_state["process_containers"][node_name].write(result_text)
                
            
            # 메인 응답 업데이트 (coordinator의 결과)
            #coordinator_result = st.session_state["ai_results"].get("coordinator", {}).get("text", "분석이 완료되었습니다.")
            #st.session_state["process_containers"][last_node].write(result_text)
            main_response.write(last_message)
        
        # 세션 상태 업데이트 및 히스토리 저장
        #display_chat_history()