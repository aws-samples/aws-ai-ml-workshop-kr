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

for i in range(10):
    if f"ph{i}" not in st.session_state: st.session_state[f"ph{i}"] = st.empty()
    
####################### Application ###############################
if len(st.session_state["history_ask"]) > 0: 
    display_chat_history()

if user_input := st.chat_input(): # 사용자 입력 받기
    st.chat_message("user").write(user_input)
    st.session_state["recent_ask"] = user_input
    
    node_names = ["coordinator", "planner", "supervisor", "coder", "reporter"]
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
        for node_name in node_names:
            with st.expander(f"🔄 {node_name.upper()}: {node_descriptions[node_name]}", expanded=True):
                st.session_state["process_containers"][node_name] = st.empty()
                st.session_state["process_containers"][node_name].info(f"Waiting...")

                # 에이전트가 사용하는 툴 결과를 표시할 컨테이너
                st.markdown(f"🔧 툴 사용 현황")
                st.session_state["tool_containers"][node_name] = {}
                st.session_state["tool_containers"][node_name]["input"] = st.empty()
                st.session_state["tool_containers"][node_name]["output"] = st.empty()
                st.session_state["tool_containers"][node_name]["input"].info(f"Tool not used yet")
                st.session_state["tool_containers"][node_name]["output"].info(f"Tool not used yet")
                
                #st.session_state["tool_containers"][node_name] = st.empty()
                #st.session_state["tool_containers"][node_name].info(f"Tool not used yet")
        
        # 차트를 위한 컨테이너 (필요한 경우)
        chart_container = st.empty()
        
        with st.spinner('분석 중...'):
            # 실행 및 결과 처리
            execution(user_query=user_input)
            
            # 각 에이전트의 결과 업데이트
            for node_name in node_names:
                if node_name == "chart_generation" and st.session_state["ai_results"].get(node_name, {}) != {}:
                    chart_data = io.BytesIO(st.session_state["ai_results"][node_name])
                    st.session_state["process_containers"][node_name].image(chart_data)
                    # 메인 영역에도 차트 표시
                    chart_container.image(chart_data)
                else:
                    result_text = st.session_state["ai_results"].get(node_name, {}).get("text", "처리 완료")
                    st.session_state["process_containers"][node_name].write(result_text)
            
            # 메인 응답 업데이트 (coordinator의 결과)
            coordinator_result = st.session_state["ai_results"].get("coordinator", {}).get("text", "분석이 완료되었습니다.")
            main_response.write(coordinator_result)
        
        # 세션 상태 업데이트 및 히스토리 저장
        display_chat_history()

# import sys, os
# module_path = ".."
# sys.path.append(os.path.abspath(module_path))

# import inspect
# import streamlit as st
# from typing import Callable, TypeVar
# from streamlit.delta_generator import DeltaGenerator
# from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
# from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# ##################### Title ########################
# st.set_page_config(page_title="GenAI-driven Analytics 💬", page_icon="💬", layout="wide")  # layout을 wide로 변경
# st.title("GenAI-driven Analytics 💬")
# st.markdown('''- This chatbot is implemented using Amazon Bedrock Claude v3.5 Sonnet.''')
# st.markdown('''
#             - You can find the source code in 
#             [this Github](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/20_applications/09_genai_analytics)
#             ''')

# from main import execution
# import io

# ##################### Functions ########################
# def display_chat_history():
#     node_names = ["coordinator", "planner", "supervisor", "coder", "reporter"]
#     st.session_state["history_ask"].append(st.session_state["recent_ask"])

#     recent_answer = {}
#     for node_name in node_names:
#         print ("node_name", node_name)
#         if node_name != "chart_generation": recent_answer[node_name] = st.session_state["ai_results"][node_name].get("text", "None")
#         else:
#             if st.session_state["ai_results"][node_name] != {}:
#                 recent_answer[node_name] = io.BytesIO(st.session_state["ai_results"][node_name])
#             else: recent_answer[node_name] = "None"
#         st.session_state["ai_results"][node_name] = {} ## reset
#     st.session_state["history_answer"].append(recent_answer)

#     for i, (user, assistant) in enumerate(zip(st.session_state["history_ask"], st.session_state["history_answer"])):
        
#         # 사용자 메시지 표시
#         with st.chat_message("user"):
#             st.write(user)
        
#         # 응답을 컬럼으로 분할
#         left_col, right_col = st.columns([1, 1])
        
#         # 왼쪽 컬럼 - 주요 LLM 출력
#         with left_col:
#             with st.chat_message("assistant"):
#                 # 기본으로 표시할 노드 선택 (예: agent 또는 chart_description)
#                 main_output = assistant["agent"]
#                 st.write(main_output)
                
#                 # 차트 이미지가 있으면 표시
#                 if assistant["chart_generation"] != "None":
#                     st.image(assistant["chart_generation"])
        
#         # 오른쪽 컬럼 - 프로세스 상세 정보
#         with right_col:
#             with st.container():
#                 st.subheader("Process Details")
#                 # 탭으로 모든 프로세스 단계 표시
#                 tabs = st.tabs(node_names)
#                 for tab, node_name in zip(tabs, node_names):
#                     with tab:
#                         if node_name == "chart_generation" and assistant[node_name] != "None":
#                             st.image(assistant[node_name])
#                         else:
#                             st.write(assistant[node_name])

# T = TypeVar("T")
# def get_streamlit_cb(parent_container: DeltaGenerator):
    
#     def decor(fn: Callable[..., T]) -> Callable[..., T]:
#         ctx = get_script_run_ctx()

#         def wrapper(*args, **kwargs) -> T:
#             add_script_run_ctx(ctx=ctx)
#             return fn(*args, **kwargs)

#         return wrapper

#     st_cb = StreamlitCallbackHandler(parent_container=parent_container)

#     for name, fn in inspect.getmembers(st_cb, predicate=inspect.ismethod):
#         if name.startswith("on_"):
#             setattr(st_cb, name, decor(fn))

#     return st_cb

# ####################### Initialization ###############################
# # Store the initial value of widgets in session state
# if "messages" not in st.session_state: st.session_state["messages"] = []
# if "history_ask" not in st.session_state: st.session_state["history_ask"] = []
# if "history_answer" not in st.session_state: st.session_state["history_answer"] = []
# if "ai_results" not in st.session_state: st.session_state["ai_results"] = {"coordinator": {}, "planner": {}, "supervisor": {}, "coder": {}, "reporter": {}}

# for i in range(10):
#     if f"ph{i}" not in st.session_state: st.session_state[f"ph{i}"] = st.empty()
    
# ####################### Application ###############################
# if len(st.session_state["history_ask"]) > 0: 
#     display_chat_history()

# if user_input := st.chat_input(): # 사용자 입력 받기
#     st.chat_message("user").write(user_input)
#     st.session_state["recent_ask"] = user_input
    
#     # 처리 중 UI 구성 - 컬럼 레이아웃 사용
#     left_col, right_col = st.columns([1, 1])
    
#     # 왼쪽 컬럼 - 메인 응답을 위한 준비
#     with left_col:
#         with st.chat_message("assistant"):
#             main_response_container = st.empty()
#             main_response_container.write("처리 중...")
            
#             # 이미지를 위한 자리 확보
#             image_container = st.empty()
    
#     # 오른쪽 컬럼 - 프로세스 단계 모니터링
#     with right_col:
#         with st.container():
#             st.subheader("Processing Steps")
#             # 각 프로세스 단계를 위한 빈 컨테이너 생성
#             if "process_containers" not in st.session_state:
#                 st.session_state["process_containers"] = {}
#                 node_names = ["coordinator", "planner", "supervisor", "coder", "reporter"]
#                 for node_name in node_names:
#                     st.session_state["process_containers"][node_name] = st.empty()
#                     st.session_state["process_containers"][node_name].info(f"{node_name}: Waiting...")

#     with st.spinner(f'Thinking...'):
#         # 여기서 execution 함수를 호출하고 각 단계마다 해당 컨테이너 업데이트
#         execution(user_query=user_input)
        
#         # 결과 업데이트 - 실제 구현에서는 각 단계에서 컨테이너를 직접 업데이트 필요
#         for node_name in node_names:
#             if node_name != "chart_generation":
#                 result_text = st.session_state["ai_results"][node_name].get("text", "None")
#                 process_containers[node_name].write(f"{node_name}: {result_text}")
#             else:
#                 if st.session_state["ai_results"][node_name] != {}:
#                     chart_data = io.BytesIO(st.session_state["ai_results"][node_name])
#                     process_containers[node_name].image(chart_data)
#                 else:
#                     process_containers[node_name].info(f"{node_name}: No chart generated")
        
#         # 메인 출력 업데이트
#         main_response_container.write(st.session_state["ai_results"]["agent"].get("text", "처리 완료"))
        
#         # 차트 표시 (있는 경우)
#         if st.session_state["ai_results"]["chart_generation"] != {}:
#             chart_data = io.BytesIO(st.session_state["ai_results"]["chart_generation"])
#             image_container.image(chart_data)
        
#         # 세션 상태 업데이트 및 히스토리 저장
#         display_chat_history()

