import boto3
import json
from src.utils.strands_sdk_utils import strands_utils
from botocore.config import Config

def parse_sse_data(sse_bytes):
    if not sse_bytes or len(sse_bytes) == 0:
        return None

    try:
        text = sse_bytes.decode('utf-8').strip()
        if not text or text == '': return None

        if text.startswith('data: '):
            json_text = text[6:].strip()
            if json_text: return json.loads(json_text)
        else:
            return json.loads(text)

    except Exception as e:
        pass

    return None

config = Config(
    read_timeout=900,  # 타임아웃 늘리기
    connect_timeout=300,
    retries={
        'max_attempts': 3,
        'mode': 'adaptive'
    }
)

client = boto3.client('bedrock-agentcore', region_name='us-west-2', config=config)

#payload = json.dumps({
#    "input": {"prompt": "Explain machine learning in simple terms"}
#})
payload=json.dumps({"prompt": '이것은 아마존 상품판매 데이터를 분석하고 싶습니다. 분석대상은 "./data/Dat-fresh-food-claude.csv" 파일 입니다. 데이터를 기반으로 마케팅 인사이트 추출을 위한 분석을 진행해 주세요. 분석은 기본적인 데이터 속성 탐색 부터, 상품 판매 트렌드, 변수 관계, 변수 조합 등 다양한 분석 기법을 수행해 주세요. 데이터 분석 후 인사이트 추출에 필요한 사항이 있다면 그를 위한 추가 분석도 수행해 주세요. 분석 리포트는 상세 분석과 그 것을 뒷받침 할 수 있는 이미지 및 차트를 함께 삽입해 주세요. 최종 리포트는 pdf 형태로 저장해 주세요.'})
boto3_response = client.invoke_agent_runtime(
    agentRuntimeArn='arn:aws:bedrock-agentcore:us-west-2:615299776985:runtime/bedrock_manus_runtime-4EkPSt4ohK',
    runtimeSessionId='dfmeoagmreaklgmrkleafremoigrmtesogmtrskhmtkrlshmt-14',  # Must be 33+ chars
    payload=payload,
    qualifier="DEFAULT" # Optional
)
if "text/event-stream" in boto3_response.get("contentType", ""):
    content = []
    for event in boto3_response["response"].iter_lines(chunk_size=1):
        event = parse_sse_data(event)
        if event is None:  # None 체크 추가
            continue
        else:
            strands_utils.process_event_for_display(event)
else:
    try:
        events = []
        for event in boto3_response.get("response", []):
            print ("6", event)
            events.append(event)
    except Exception as e:
        events = [f"Error reading EventStream: {e}"]
    display(Markdown(json.loads(events[0].decode("utf-8"))))


response_body = response['response'].read()
response_data = json.loads(response_body)
print("Agent Response:", response_data)