

import os
import shutil
import asyncio
import argparse
import json
from src.graph.builder import build_graph
from src.utils.strands_sdk_utils import strands_utils

# Import event queue for unified event processing
from src.utils.event_queue import clear_queue

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
    print("\n=== Starting Local Runtime Event Stream ===")

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
    """
    Execute full graph streaming workflow in local environment
    Direct event processing without AgentCore API
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
    events_list = []

    # Stream events from graph execution
    async for event in graph.stream_async({
        "request": user_query,
        "request_prompt": f"Here is a user request: <user_request>{user_query}</user_request>"
    }):
        event_count += 1
        # Add local runtime metadata
        event["event_id"] = event_count
        event["runtime_source"] = "bedrock_manus_local"

        # Print event for local debugging
        print(f"Event {event_count}: {event.get('type', 'unknown')}")

        # Store events for final processing
        events_list.append(event)

        # Mark final event
        if event.get("type") == "workflow_complete":
            event["total_events"] = event_count
            event["message"] = "All events processed locally without AgentCore Runtime"

    _print_conversation_history()
    print("=== Local Runtime Event Stream Complete ===")

    # Return final result for local execution
    return {
        "total_events": event_count,
        "final_message": "Local execution completed successfully",
        "events": events_list
    }

def main_local_execution(payload):
    """
    Main function for local execution
    Synchronous wrapper for async graph execution
    """
    try:
        # Run async function in event loop
        result = asyncio.run(graph_streaming_execution(payload))
        return result
    except Exception as e:
        print(f"Local execution error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bedrock Manus Multi-Agent Local Runtime")
    parser.add_argument("payload", type=str, help="JSON payload with prompt")
    args = parser.parse_args()

    try:
        payload = json.loads(args.payload)
        result = main_local_execution(payload)
        print("\n=== Final Result ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print("Error: Invalid JSON payload")
    except Exception as e:
        print(f"Execution error: {e}")
