
"""
Entry point script for the Strands Agent Demo.
"""
import os
import shutil
import asyncio
import argparse
from dotenv import load_dotenv
from src.utils.strands_sdk_utils import strands_utils
from src.graph.builder import build_graph

# Load environment variables
load_dotenv()

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

def _print_token_usage_summary():
    """Print final token usage statistics"""
    print("\n" + "="*60)
    print("=== Token Usage Summary ===")
    print("="*60)

    from src.graph.nodes import _global_node_states
    shared_state = _global_node_states.get('shared', {})
    token_usage = shared_state.get('token_usage', {})

    if token_usage:
        total_input = token_usage.get('total_input_tokens', 0)
        total_output = token_usage.get('total_output_tokens', 0)
        total = token_usage.get('total_tokens', 0)
        cache_read = token_usage.get('cache_read_input_tokens', 0)
        cache_write = token_usage.get('cache_write_input_tokens', 0)

        print(f"\nTotal Tokens: {total:,}")
        print(f"  - Regular Input:  {total_input:>8,} (100% cost)")
        print(f"  - Cache Read:     {cache_read:>8,} (10% cost - 90% discount)")
        print(f"  - Cache Write:    {cache_write:>8,} (125% cost - 25% extra)")
        print(f"  - Output:         {total_output:>8,}")

        by_agent = token_usage.get('by_agent', {})
        if by_agent:
            print("\n" + "-"*60)
            print("Token Usage by Agent:")
            print("-"*60)

            # Sort agents for consistent display
            for agent_name in sorted(by_agent.keys()):
                usage = by_agent[agent_name]
                input_tokens = usage.get('input', 0)
                output_tokens = usage.get('output', 0)
                agent_cache_read = usage.get('cache_read', 0)
                agent_cache_write = usage.get('cache_write', 0)
                agent_total = input_tokens + output_tokens + agent_cache_read + agent_cache_write

                print(f"\n  [{agent_name}] Total: {agent_total:,}")
                print(f"    - Regular Input:  {input_tokens:>8,} (100% cost)")
                print(f"    - Cache Read:     {agent_cache_read:>8,} (10% cost - 90% discount)")
                print(f"    - Cache Write:    {agent_cache_write:>8,} (125% cost - 25% extra)")
                print(f"    - Output:         {output_tokens:>8,}")

        print("="*60)
    else:
        print("No token usage data available")
        print("="*60)

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
    _print_token_usage_summary()
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
            "user_query": "너가 작성할 것은 moon market 의 판매 현황 보고서야. 세일즈 및 마케팅 관점으로 분석을 해주고, 차트 생성 및 인사이트도 뽑아서 docx 파일로 만들어줘. 분석대상은 './data/Dat-fresh-food-claude.csv' 파일 입니다."
            #"user_query": "너가 작성할 것은 moon market 의 판매 현황 보고서야. 세일즈 및 마케팅 관점으로 분석을 해주고, 차트 생성 및 인사이트도 뽑아서 docx 파일로 만들어줘. 분석대상은 './data/Dat-fresh-food-claude.csv' 파일 입니다. planning 할때 coder 분석항목은 1-2개로 최대한 간단하게 하고 Validator는 제외해줘. 왜냐하면 리포터 에이전트 테스트 중이거든. Validator가 있으면 시간을 많이 먹어 "
        }

    #########################
    ## modification END    ##
    #########################

    # Use full graph streaming execution for real-time streaming with graph structure
    async def run_streaming():
        async for event in graph_streaming_execution(payload):
            strands_utils.process_event_for_display(event)

    asyncio.run(run_streaming())