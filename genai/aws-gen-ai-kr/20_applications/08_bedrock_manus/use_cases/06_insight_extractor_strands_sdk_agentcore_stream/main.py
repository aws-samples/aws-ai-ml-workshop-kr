"""
Entry point script for the LangGraph Demo.
"""
import os
import json
import shutil
import asyncio
from src.workflow import run_graph_streaming_workflow

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

async def graph_streaming_execution(user_query):
    """Execute full graph streaming workflow - queue-only event processing"""
    remove_artifact_folder()
    
    # Import event queue for unified event processing
    from src.utils.event_queue import get_event, has_events, clear_queue
    import asyncio
    
    # Clear any existing events in queue
    clear_queue()
    
    
    print("\n=== Starting Queue-Only Event Stream ===")
    print("All events (nodes + tools) processed through global queue")
    
    # Start workflow in background - it will put all events in global queue
    async def run_workflow_background():
        """Run workflow and consume its events (since nodes already use put_event)"""
        try:
            async for _ in run_graph_streaming_workflow(user_input=user_query, debug=False):
                # We ignore these events since nodes already put them in global queue
                pass
        except Exception as e: print(f"Workflow error: {e}")
    
    workflow_task = asyncio.create_task(run_workflow_background())
    
    try:
        workflow_complete = False
        
        # Main event loop - only monitor global queue
        while not workflow_complete:
            # Check for events in global queue
            if has_events():
                event = get_event()
                if event:
                    #print(f"DEBUG: Queue event: {event}")
                    yield event
            
            # Check if workflow is complete
            if workflow_task.done(): workflow_complete = True
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.01)
            
    finally:
        # Wait for workflow to complete
        
        if not workflow_task.done():
            try:
                await asyncio.wait_for(workflow_task, timeout=1.0)
            except asyncio.TimeoutError:
                workflow_task.cancel()
                try: await workflow_task
                except asyncio.CancelledError: pass
        
        # Process any final remaining events in queue
        while has_events():
            event = get_event()
            if event:
                print(f"DEBUG: Final queue event: {event}")
                yield event
    
    # Final workflow complete event
    yield {"type": "workflow_complete", "message": "All events processed through global queue"}
    
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
    
    print("\n=== Queue-Only Event Stream Complete ===")

if __name__ == "__main__":

    remove_artifact_folder()

    # Use predefined query for testing
    user_query = '''
        이것은 아마존 상품판매 데이터를 분석하고 싶습니다.
        분석대상은 "./data/Dat-fresh-food-claude.csv" 파일 입니다.
        데이터를 기반으로 마케팅 인사이트 추출을 위한 분석을 진행해 주세요.
        분석은 기본적인 데이터 속성 탐색 부터, 상품 판매 트렌드, 변수 관계, 변수 조합 등 다양한 분석 기법을 수행해 주세요.
        데이터 분석 후 인사이트 추출에 필요한 사항이 있다면 그를 위한 추가 분석도 수행해 주세요.
        분석 리포트는 상세 분석과 그 것을 뒷받침 할 수 있는 이미지 및 차트를 함께 삽입해 주세요.
        최종 리포트는 pdf 형태로 저장해 주세요.
    '''
    
    # Use full graph streaming execution for real-time streaming with graph structure
    async def run_streaming():
        # Initialize colored callbacks for terminal display
        from src.utils.strands_sdk_utils import ColoredStreamingCallback
        callback_reasoning = ColoredStreamingCallback('cyan')
        callback_default = ColoredStreamingCallback('purple')
        callback_red = ColoredStreamingCallback('red')
        
        def process_event_for_display(event):
            """Process events for colored terminal output"""
            if event:
                source = event.get("source", "unknown")
                if event.get("event_type") == "text_chunk":
                    if source == "coder_tool": 
                        callback_red.on_llm_new_token(event.get('data', ''))
                    else: 
                        callback_default.on_llm_new_token(event.get('data', ''))
                        
                elif event.get("event_type") == "reasoning":
                    if source == "coder_tool": 
                        callback_red.on_llm_new_token(event.get('reasoning_text', ''))
                    else: 
                        callback_reasoning.on_llm_new_token(event.get('reasoning_text', ''))
        
        async for event in graph_streaming_execution(user_query):
            # Process event for terminal display
            process_event_for_display(event)
            # Event is yielded from graph_streaming_execution and consumed here for display
    
    asyncio.run(run_streaming())