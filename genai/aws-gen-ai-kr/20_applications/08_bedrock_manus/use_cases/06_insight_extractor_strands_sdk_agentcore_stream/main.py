"""
Entry point script for the LangGraph Demo.
"""
import os
import sys
import shutil
import asyncio
import streamlit as st
from src.workflow import run_agent_workflow, run_agent_workflow_streaming

def remove_artifact_folder(folder_path="./artifacts/"):
    """
    ./artifact/ 폴더가 존재하면 삭제하는 함수
    
    Args:
        folder_path (str): 삭제할 폴더 경로
    """
    if os.path.exists(folder_path):
        print(f"'{folder_path}' 폴더를 삭제합니다...")
        try:
            # 폴더와 그 내용을 모두 삭제
            shutil.rmtree(folder_path)
            print(f"'{folder_path}' 폴더가 성공적으로 삭제되었습니다.")
        except Exception as e:
            print(f"오류 발생: {e}")
    else:
        print(f"'{folder_path}' 폴더가 존재하지 않습니다.")

def execution(user_query):
    remove_artifact_folder()
    result = asyncio.run(
        run_agent_workflow(
            user_input=user_query,
            debug=False
        )
    )

    # Print the conversation history
    print("\n=== Conversation History ===")
    print ("result", result)
    for history in result["history"]:
        print ("===")
        print (f'agent: {history["agent"]}')
        print (f'message: {history["message"]}')

    return result

async def execution_streaming(user_query):
    """Execute workflow with streaming support"""
    remove_artifact_folder()
    
    print("\n=== Starting Streaming Execution ===")
    
    final_result = None
    async for event in run_agent_workflow_streaming(user_input=user_query, debug=False):
        if event.get("type") == "workflow_complete":
            final_result = event.get("result")
            print(f"\n=== Workflow Complete ===")
        elif event.get("event_type") == "text_chunk":
            # Print streaming text chunks in real-time
            print(event.get("data", ""), end="", flush=True)
        else:
            # Print other events
            print(f"\n[EVENT] {event}")
    
    # Print the conversation history from global state
    print("\n\n=== Conversation History ===")
    from src.graph.nodes import _global_node_states
    shared_state = _global_node_states.get('shared', {})
    history = shared_state.get('history', [])
    
    if history:
        for hist_item in history:
            print("===")
            print(f'agent: {hist_item["agent"]}')
            print(f'message: {hist_item["message"]}')
    else:
        print("No conversation history found in global state")
    
    return final_result
    

if __name__ == "__main__":

    remove_artifact_folder()

    if len(sys.argv) > 1: user_query = " ".join(sys.argv[1:])
    else: user_query = input("Enter your query: ")
    
    user_query = '''
        이것은 아마존 상품판매 데이터를 분석하고 싶습니다.
        분석대상은 "./data/Dat-fresh-food-claude.csv" 파일 입니다.
        데이터를 기반으로 마케팅 인사이트 추출을 위한 분석을 진행해 주세요.
        분석은 기본적인 데이터 속성 탐색 부터, 상품 판매 트렌드, 변수 관계, 변수 조합 등 다양한 분석 기법을 수행해 주세요.
        데이터 분석 후 인사이트 추출에 필요한 사항이 있다면 그를 위한 추가 분석도 수행해 주세요.
        분석 리포트는 상세 분석과 그 것을 뒷받침 할 수 있는 이미지 및 차트를 함께 삽입해 주세요.
        최종 리포트는 pdf 형태로 저장해 주세요.
    '''
    
    # Use streaming execution
    result = asyncio.run(execution_streaming(user_query))