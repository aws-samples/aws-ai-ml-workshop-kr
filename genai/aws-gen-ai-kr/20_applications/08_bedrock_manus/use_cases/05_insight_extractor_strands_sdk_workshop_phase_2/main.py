
"""
Entry point script for the Strands Agent Demo.
"""
import os
import shutil
import asyncio
import argparse
from dotenv import load_dotenv
from src.workflow import run_graph_streaming_workflow
from src.utils.strands_sdk_utils import strands_utils

# Load environment variables
load_dotenv()

# Observability
from opentelemetry import trace, context
from src.utils.agentcore_observability import set_session_context, add_span_event

# Import event queue for unified event processing
from src.utils.event_queue import get_event, has_events, clear_queue 

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

async def _cleanup_workflow(workflow_task):
    """Handle workflow completion and cleanup"""
    if not workflow_task.done():
        try:
            await asyncio.wait_for(workflow_task, timeout=1.0)
        except asyncio.TimeoutError:
            workflow_task.cancel()
            try: await workflow_task
            except asyncio.CancelledError: pass

async def _yield_pending_events():
    """Yield any pending events from queue"""
    while has_events():
        event = get_event()
        if event: yield event

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
    """Execute full graph streaming workflow - queue-only event processing"""

    #_get_env()
    _setup_execution()

    # Get user query from payload
    user_query = payload.get("user_query", "")
    session_id = payload.get("session-id", "default")
    context_token = set_session_context(session_id)

    try:
        # Get tracer for main application
        tracer = trace.get_tracer(
            instrumenting_module_name=os.getenv("TRACER_MODULE_NAME", "insight_extractor_agent"),
            instrumenting_library_version=os.getenv("TRACER_LIBRARY_VERSION", "1.0.0")
        )
        with tracer.start_as_current_span("insight_extractor_session") as span:   
            # Start workflow in background
            async def run_workflow():
                try:
                    result = await run_graph_streaming_workflow(user_input=user_query)
                    print(f"Workflow completed: {result}")
                except Exception as e:
                    print(f"Workflow error: {e}")

            workflow_task = asyncio.create_task(run_workflow())

            try:
                # Main event loop - monitor queue until workflow completes
                while not workflow_task.done():
                    async for event in _yield_pending_events():
                        yield event
                    await asyncio.sleep(0.01)

            finally:
                await _cleanup_workflow(workflow_task)

                # Process remaining events
                async for event in _yield_pending_events():
                    yield event

            # Final completion
            yield {"type": "workflow_complete", "message": "All events processed through global queue"}
            _print_conversation_history()
            print("=== Queue-Only Event Stream Complete ===")

            # Add Event
            add_span_event(span, "user_query", {"user-query": str(user_query)}) 

    finally:
        context.detach(context_token)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Strands Agent Demo')
    parser.add_argument('--user_query', type=str, help='User query for the agent')
    parser.add_argument('--session_id', type=str, default='insight-extractor-1', help='Session ID')

    args = parser.parse_args()


    #########################
    ## modification START  ##
    #########################

    # Use argparse values if provided, otherwise use predefined values
    if args.user_query:
        payload = {
            "user_query": args.user_query,
            "session-id": args.session_id
        }
    else:
        # Use predefined query for testing
        payload = {
            "user_query": "너가 작성할 것은 moon market 의 판매 현황 보고서야. 세일즈 및 마케팅 관점으로 분석을 해주고, 차트 생성 및 인사이트도 뽑아서 pdf 파일로 만들어줘. 분석대상은 './data/Dat-fresh-food-claude.csv' 파일 입니다.",
            "session-id": "insight-extractor-1"
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

    #user_query = "너가 작성할 것은 moon market 의 판매 현황 보고서야. 세일즈 및 마케팅 관점으로 분석을 해주고, 차트 생성 및 인사이트도 뽑아서 pdf 파일로 만들어줘. 분석대상은 './data/Dat-fresh-food-claude.csv' 파일 입니다. Coder 에이전트가 할일은 최대한 작게 해줘. 왜냐하면 reporter 에이전트 테스트 중이라 빨리 코더 단계를 넘어 가야 하거든. 부탁해."
    #user_query = "너가 작성할 것은 moon market 의 판매 현황 보고서야. 세일즈 및 마케팅 관점으로 분석을 해주고, 차트 생성 및 인사이트도 뽑아서 pdf 파일로 만들어줘. 분석대상은 './data/Dat-fresh-food-claude.csv' 파일 입니다."

    # user_query = '''

    #     분석대상은 "./data/Dat-fresh-food-claude.csv" 파일 입니다.
    #     데이터를 기반으로 마케팅 인사이트 추출을 위한 분석을 진행해 주세요.
    #     분석은 기본적인 데이터 속성 탐색 부터, 상품 판매 트렌드, 변수 관계, 변수 조합 등 다양한 분석 기법을 수행해 주세요.
    #     데이터 분석 후 인사이트 추출에 필요한 사항이 있다면 그를 위한 추가 분석도 수행해 주세요.
    #     분석 리포트는 상세 분석과 그 것을 뒷받침 할 수 있는 이미지 및 차트를 함께 삽입해 주세요.
    #     최종 리포트는 pdf 형태로 저장해 주세요.
    # '''
