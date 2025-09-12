
import os
import shutil
import asyncio
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from src.graph.builder import build_graph
from src.utils.strands_sdk_utils import strands_utils

# Import event queue for unified event processing
from src.utils.event_queue import clear_queue

app = BedrockAgentCoreApp()

def remove_artifact_folder(folder_path="./artifacts/"): # ./artifact/ 폴더가 존재하면 삭제하는 함수
    if os.path.exists(folder_path):
        print(f"'{folder_path}' 폴더를 삭제합니다...")
        try:
            shutil.rmtree(folder_path)
            print(f"'{folder_path}' 폴더가 성공적으로 삭제되었습니다.")
        except Exception as e: 
            print(f"오류 발생: {e}")
    else:
        print(f"'{folder_path}' 폴더가 존재하지 않습니다.")

def _setup_execution():
    """Initialize execution environment"""
    remove_artifact_folder()
    clear_queue()
    print("\n=== Starting AgentCore Runtime Event Stream ===")

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

@app.entrypoint
async def graph_streaming_execution(payload):
    """
    Execute full graph streaming workflow through AgentCore Runtime
    Queue-only event processing compatible with AgentCore API
    """
    # Get user query from payload
    user_query = payload.get("prompt", "")

    if not user_query:
        # Use default query if none provided
        user_query = "너가 작성할 것은 moon market 의 판매 현황 보고서야. 세일즈 및 마케팅 관점으로 분석을 해주고, 차트 생성 및 인사이트도 뽑아서 pdf 파일로 만들어줘. 분석대상은 './data/Dat-fresh-food-claude.csv' 파일 입니다. Coder 에이전트가 할일은 최대한 작게 해줘. 왜냐하면 reporter 에이전트 테스트 중이라 빨리 코더 단계를 넘어 가야 하거든. 부탁해."

    _setup_execution()

    # Build graph and use stream_async method
    graph = build_graph()
    event_count = 0

    # Stream events from graph execution
    async for event in graph.stream_async({
        "request": user_query,
        "request_prompt": f"Here is a user request: <user_request>{user_query}</user_request>"
    }):
        event_count += 1
        # Add AgentCore runtime metadata
        event["event_id"] = event_count
        event["runtime_source"] = "bedrock_manus_agentcore"

        # Mark final event
        if event.get("type") == "workflow_complete":
            event["total_events"] = event_count
            event["message"] = "All events processed through global queue via AgentCore Runtime"

        yield event

    _print_conversation_history()
    print("=== AgentCore Runtime Event Stream Complete ===")


if __name__ == "__main__":
    app.run()

