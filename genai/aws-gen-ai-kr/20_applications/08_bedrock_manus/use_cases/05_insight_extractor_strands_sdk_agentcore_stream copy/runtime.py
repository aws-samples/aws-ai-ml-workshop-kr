
"""
AgentCore Runtime for Bedrock-Manus Multi-Agent System
Unified event streaming through global queue - compatible with AgentCore Runtime API
"""
import os
import shutil
import asyncio
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from src.workflow import run_graph_streaming_workflow

app = BedrockAgentCoreApp()

def remove_artifact_folder(folder_path="./artifacts/"):
    """
    ./artifact/ 폴더가 존재하면 삭제하는 함수

    Args:
        folder_path (str): 삭제할 폴더 경로
    """
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
        except Exception as e:
            print(f"아티팩트 폴더 삭제 오류: {e}")

@app.entrypoint
async def unified_streaming_execution(payload):
    """
    Unified event streaming for AgentCore Runtime
    All events (nodes + tools) processed through global queue
    """
    user_query = payload.get("prompt", "")

    if not user_query:
        yield {"type": "error", "message": "No prompt provided"}
        return

    # Clean artifacts folder
    remove_artifact_folder()

    # Import event queue for unified event processing
    from src.utils.event_queue import get_event, has_events, clear_queue

    # Clear any existing events in queue
    clear_queue()

    # Start workflow in background - it will put all events in global queue
    async def run_workflow_background():
        """Run workflow and consume its events (since nodes already use put_event)"""
        try:
            async for _ in run_graph_streaming_workflow(user_input=user_query, debug=False):
                # We ignore these events since nodes already put them in global queue
                pass
        except Exception as e:
            # Put error in queue for client to receive
            from src.utils.event_queue import put_event
            put_event({
                "type": "error",
                "event_type": "error",
                "source": "workflow",
                "message": f"Workflow error: {str(e)}"
            })

    workflow_task = asyncio.create_task(run_workflow_background())

    try:
        workflow_complete = False
        event_count = 0

        # Main event loop - only monitor global queue
        while not workflow_complete:
            # Check for events in global queue
            if has_events():
                event = get_event()
                if event:
                    event_count += 1
                    # Add event metadata for runtime tracking
                    event["event_id"] = event_count
                    event["runtime_source"] = "bedrock_manus_runtime"
                    yield event

            # Check if workflow is complete
            if workflow_task.done():
                workflow_complete = True

            # Small delay to prevent busy waiting
            await asyncio.sleep(0.01)

    finally:
        # Wait for workflow to complete gracefully
        if not workflow_task.done():
            try:
                await asyncio.wait_for(workflow_task, timeout=2.0)
            except asyncio.TimeoutError:
                workflow_task.cancel()
                try:
                    await workflow_task
                except asyncio.CancelledError:
                    pass

        # Process any final remaining events in queue
        while has_events():
            event = get_event()
            if event:
                event_count += 1
                event["event_id"] = event_count
                event["runtime_source"] = "bedrock_manus_runtime"
                event["final_event"] = True
                yield event

    # Final completion event
    yield {
        "type": "workflow_complete", 
        "event_type": "completion",
        "message": "All events processed through unified global queue",
        "total_events": event_count,
        "runtime_source": "bedrock_manus_runtime"
    }


if __name__ == "__main__":
    app.run()
