

"""
Entry point script for the LangGraph Demo.
"""
import os
import shutil
import argparse
import asyncio
from src.workflow import run_graph_streaming_workflow
from bedrock_agentcore.runtime import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

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

@app.entrypoint
async def graph_streaming_execution(payload):

    print ("111111")
    user_query = payload.get("prompt")
    print ("222222")

    """Execute full graph streaming workflow with real-time events"""
    remove_artifact_folder()

    # Import event queue for processing tool events
    from src.utils.event_queue import get_event, has_events

    print("\n=== Starting Graph Streaming Execution ===")
    print("Real-time streaming events from full graph:")

    try:
        async for event in run_graph_streaming_workflow(user_input=user_query, debug=False):
            # Yield node-level event first
            print(f"Node event: {event}")
            yield event
            
            # Check and yield any queued events from tools (like coder_agent_tool)
            while has_events():
                queue_event = get_event()
                if queue_event:
                    print(f"Queue event: {queue_event}")
                    yield queue_event

    except Exception as e:
        # Handle errors gracefully in streaming context
        error_response = {"error": str(e), "type": "stream_error"}
        print(f"Streaming error: {error_response}")
        yield error_response

    # async for event in run_graph_streaming_workflow(user_input=user_query, debug=False):

if __name__ == "__main__":
    app.run()



