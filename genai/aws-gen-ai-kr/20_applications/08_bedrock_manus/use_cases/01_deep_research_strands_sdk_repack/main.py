
"""
Entry point script for the Strands Agent Demo.
"""
# Load environment variables FIRST before any other imports
from dotenv import load_dotenv
load_dotenv()

import os
import shutil
import asyncio
import argparse
from src.utils.strands_sdk_utils import strands_utils
from src.graph.builder import build_graph

# Import event queue for unified event processing
from src.utils.event_queue import clear_queue 

def remove_artifact_folder(folder_path="./artifacts/"):
    """
    ./artifact/ 폴더가 존재하면 삭제하는 함수

    Args:
        folder_path (str): 삭제할 폴더 경로
    """
    if os.path.exists(folder_path):
        print(f"'{folder_path}' 폴더를 삭제합니다...")
        try:
            shutil.rmtree(folder_path)
            print(f"'{folder_path}' 폴더가 성공적으로 삭제되었습니다.")
        except Exception as e: print(f"오류 발생: {e}")
    else:
        print(f"'{folder_path}' 폴더가 존재하지 않습니다.")

def _setup_execution():
    """Initialize execution environment"""
    remove_artifact_folder()
    clear_queue()
    print("\n=== Starting Queue-Only Event Stream ===")

def _print_conversation_history():
    """Print final conversation history"""
    print("\n=== Conversation History ===")
    from src.graph.nodes import _global_node_states
    shared_state = _global_node_states.get('shared', {})
    history = shared_state.get('history', [])

    if history:
        for hist_item in history:
            print(f"[{hist_item['agent']}] {hist_item['message']}")
    else:
        print("No conversation history found")

async def graph_streaming_execution(payload):
    """Execute full graph streaming workflow using new graph.stream_async method"""

    _setup_execution()

    # Get user query from payload
    user_query = payload.get("user_query", "")
        
    # Build graph and use stream_async method
    graph = build_graph()
    
    #########################
    ## modification START  ##
    #########################

    # Stream events from graph execution
    async for event in graph.stream_async(
        {
            "request": user_query,
            "request_prompt": f"Here is a user request: <user_request>{user_query}</user_request>"
        }
    ):
        yield event

    #########################
    ## modification END    ##
    #########################
    
    _print_conversation_history()
    print("=== Queue-Only Event Stream Complete ===")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Strands Agent Demo')
    parser.add_argument('--user_query', type=str, help='User query for the agent')
    
    args, unknown = parser.parse_known_args()

    #########################
    ## modification START  ##
    #########################

    # Use argparse values if provided, otherwise use predefined values
    if args.user_query:
        payload = {
            "user_query": args.user_query,
        }
    else:
        # Full comprehensive analysis query (main version):
        payload = {
            #"user_query": "AWS Cloud Market Analysis: Growth Prospects in Korea (2025-2030) 조사해줘. 결과는 pdf로 만들어줘. research와 coder는 정말 간단하게 1번씩만 호출 하고 조사도 아주 간단하게만 해. reporter 에이전트 테스트 중이거든."
            #"user_query": "너가 작성할 것은 moon market 의 판매 현황 보고서야. 세일즈 및 마케팅 관점으로 분석을 해주고, 차트 생성 및 인사이트도 뽑아서 pdf 파일로 만들어줘. 분석대상은 './data/Dat-fresh-food-claude.csv' 파일 입니다. planning 할때 coder 분석항목은 1-2개로 최대한 간단하게 하고 Validator는 제외해줘. 왜냐하면 리포터 에이전트 테스트 중이거든. Validator가 있으면 시간을 많이 먹어 "
            "user_query": "You’re a researcher analyzing new and emerging technology trends. Your goal is to produce Part 1 of a set of reports for a CIO/CTO of a specific company to track and prioritize emerging technologies. Part 1 focuses on a broad landscape analysis that can help them understand all emerging technologies they should keep track of, and use the content to engage other executives"
        }

    #########################
    ## modification END    ##
    #########################

    remove_artifact_folder()

    # Use full graph streaming execution for real-time streaming with graph structure
    async def run_streaming():
        async for event in graph_streaming_execution(payload):
            strands_utils.process_event_for_display(event)

    asyncio.run(run_streaming())