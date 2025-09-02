
"""
AgentCore Runtime for Bedrock-Manus Multi-Agent System
Converted from main.py to use AgentCore Runtime API with unified event streaming
"""
import os
import shutil
import asyncio
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from src.workflow import run_graph_streaming_workflow
from src.utils.strands_sdk_utils import strands_utils

# Import event queue for unified event processing
from src.utils.event_queue import get_event, has_events, clear_queue

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

async def _cleanup_workflow(workflow_task):
    """Handle workflow completion and cleanup"""
    if not workflow_task.done():
        try:
            await asyncio.wait_for(workflow_task, timeout=1.0)
        except asyncio.TimeoutError:
            workflow_task.cancel()
            try:
                await workflow_task
            except asyncio.CancelledError:
                pass

async def _yield_pending_events():
    """Yield any pending events from queue"""
    while has_events():
        event = get_event()
        if event:
            yield event

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

    # Start workflow in background
    async def run_workflow():
        try:
            result = await run_graph_streaming_workflow(user_input=user_query)
            print(f"Workflow completed: {result}")
        except Exception as e:
            print(f"Workflow error: {e}")
            # Put error in queue for client to receive
            from src.utils.event_queue import put_event
            put_event({
                "type": "error",
                "event_type": "error",
                "source": "workflow",
                "message": f"Workflow error: {str(e)}"
            })

    workflow_task = asyncio.create_task(run_workflow())
    event_count = 0

    try:
        # Main event loop - monitor queue until workflow completes
        while not workflow_task.done():
            async for event in _yield_pending_events():
                event_count += 1
                # Add AgentCore runtime metadata
                event["event_id"] = event_count
                event["runtime_source"] = "bedrock_manus_agentcore"

                # Process event for display (optional - can be removed for pure runtime)
                #strands_utils.process_event_for_display(event)

                yield event
            await asyncio.sleep(0.01)

    finally:
        await _cleanup_workflow(workflow_task)

        # Process remaining events
        async for event in _yield_pending_events():
            event_count += 1
            event["event_id"] = event_count
            event["runtime_source"] = "bedrock_manus_agentcore"
            event["final_event"] = True

            # Process event for display (optional)
            #strands_utils.process_event_for_display(event)

            yield event

    # Final completion
    completion_event = {
        "type": "workflow_complete", 
        "event_type": "completion",
        "message": "All events processed through global queue via AgentCore Runtime",
        "total_events": event_count,
        "runtime_source": "bedrock_manus_agentcore"
    }

    yield completion_event

    _print_conversation_history()
    print("=== AgentCore Runtime Event Stream Complete ===")


if __name__ == "__main__":
    app.run()



