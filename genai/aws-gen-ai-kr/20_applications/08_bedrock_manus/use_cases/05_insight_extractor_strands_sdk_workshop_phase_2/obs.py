


"""
Entry point script for the Strands Agent Demo.
"""
import os
import shutil
import asyncio
from src.graph.builder import build_graph
from src.utils.strands_sdk_utils import strands_utils

# Observability
from opentelemetry import trace, context
from src.utils.agentcore_observability import set_session_context

# Import event queue for unified event processing
from src.utils.event_queue import clear_queue 

# Env.
from dotenv import load_dotenv
load_dotenv()

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

def _get_env():

    #load_dotenv()

    # Display the OTEL-related environment variables
    otel_vars = [
        "OTEL_PYTHON_DISTRO",
        "OTEL_PYTHON_CONFIGURATOR",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_EXPORTER_OTLP_LOGS_HEADERS",
        "OTEL_RESOURCE_ATTRIBUTES",
        "AGENT_OBSERVABILITY_ENABLED",
        "OTEL_TRACES_EXPORTER"
    ]

    print("OpenTelemetry Configuration:")
    for var in otel_vars:
        value = os.getenv(var)
        if value: print(f"{var}={value}")
    print("=======================")

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
    """Execute full graph streaming workflow - queue-only event processing"""

    _get_env()
    _setup_execution()

    # Get user query from payload
    user_query = payload.get("user_query", "")
    session_id = payload.get("session-id", "default")

    context_token = set_session_context(session_id)
    print ("context_token", context_token)

    try:
        # Get tracer for main application
        tracer = trace.get_tracer(
            os.getenv("TRACER_MODULE_NAME", "insight_extractor_agent"),
            os.getenv("TRACER_LIBRARY_VERSION", "1.0.0")
        )
        with tracer.start_as_current_span("insight_extractor_session") as main_span:

            main_span.add_event( # add_event 사용 시 (CloudWatch)GenAI Observability의 trace의 event에서 바로 확인 x, "Traces"의 event에서는 확인 가능
                "user_query",
                {"user-query": str(user_query)}
            ) # attribute.name에 _(under bar)가 들어가면 안된다. 


            # Build graph and use stream_async method
            graph = build_graph()
            
            # Stream events from graph execution
            async for event in graph.stream_async({
                "request": user_query,
                "request_prompt": f"Here is a user request: <user_request>{user_query}</user_request>"
            }):
                yield event
            _print_conversation_history()
            print("=== Queue-Only Event Stream Complete ===")

    finally:
        context.detach(context_token)



if __name__ == "__main__":



    # Use predefined query for testing
    payload = {
        "user_query": "너가 작성할 것은 moon market 의 판매 현황 보고서야. 세일즈 및 마케팅 관점으로 분석을 해주고, 차트 생성 및 인사이트도 뽑아서 pdf 파일로 만들어줘. 분석대상은 './data/Dat-fresh-food-claude.csv' 파일 입니다.",
        "session-id": "insight-extractor-2"
    }

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
    remove_artifact_folder()

    # Use full graph streaming execution for real-time streaming with graph structure
    async def run_streaming():
        async for event in graph_streaming_execution(payload):
            strands_utils.process_event_for_display(event)

    asyncio.run(run_streaming())
