
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

# Observability
from opentelemetry import trace, context
from src.utils.agentcore_observability import set_session_context, add_span_event

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
    session_id = payload.get("session-id", "default")
    context_token = set_session_context(session_id)

    try:
        # Get tracer for main application
        tracer = trace.get_tracer(
            instrumenting_module_name=os.getenv("TRACER_MODULE_NAME", "insight_extractor_agent"),
            instrumenting_library_version=os.getenv("TRACER_LIBRARY_VERSION", "1.0.0")
        )
        with tracer.start_as_current_span("insight_extractor_session") as span:   
            
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

            # Add Event
            add_span_event(span, "user_query", {"user-query": str(user_query)}) 

    finally:
        context.detach(context_token)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Strands Agent Demo')
    parser.add_argument('--user_query', type=str, help='User query for the agent')
    parser.add_argument('--session_id', type=str, default='insight-extractor-1', help='Session ID')

    args, unknown = parser.parse_known_args()


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
        # Full comprehensive analysis query (main version):
        payload = {
            "user_query": "너가 작성할 것은 moon market 의 판매 현황 보고서야. 세일즈 및 마케팅 관점으로 분석을 해주고, 차트 생성 및 인사이트도 뽑아서 pdf 파일로 만들어줘. 분석대상은 './data/Dat-fresh-food-claude.csv' 파일 입니다.",
            "session-id": "insight-extractor-1"
        }
        
        # Quick test version (commented - for chart improvements validation):
        # payload = {
        #     "user_query": "데이터 './data/Dat-fresh-food-claude.csv'에서 총 판매 금액 계산하고, 차트 3개 정도 만들어서 PDF 리포트 작성해줘. 차트는 1) 카테고리별 매출, 2) 월별 매출 추이, 3) 프로모션별 매출 정도로 해줘. 차트 라벨 개선 테스트용이니 간단하게.",
        #     "session-id": "insight-extractor-1"
        # }

    #########################
    ## modification END    ##
    #########################

    remove_artifact_folder()

    # Use full graph streaming execution for real-time streaming with graph structure
    async def run_streaming():
        async for event in graph_streaming_execution(payload):
            strands_utils.process_event_for_display(event)

    asyncio.run(run_streaming())

