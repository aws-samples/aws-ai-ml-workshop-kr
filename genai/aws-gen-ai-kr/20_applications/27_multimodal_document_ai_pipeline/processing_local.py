import json
import os
import base64
import glob

from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from langchain_core.prompts import PromptTemplate
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import StreamingStdOutCallbackHandler
import boto3

"""
local_processing.py

이 스크립트는 다음 단계로 이루어집니다:

1) .env 파일 로드
2) PDF 파티셔닝(텍스트/이미지/테이블 등 추출, GPU 사용여부 등)
3) Prompt Template 정의
4) AWS Bedrock(Claude) 클라이언트 설정
5) 체인 구성(Runnable 파이프라인)
6) PDF 요소들에 대해 체인 실행 & 질의응답 생성
7) 최종 JSONL 파일로 저장

환경 변수:
  - AWS_REGION
  - AWS_ACCESS_KEY_ID
  - AWS_SECRET_ACCESS_KEY

사용 라이브러리:
  - dotenv, unstructured, langchain-core, langchain-aws, boto3 등
"""

# -------------------------------------------------------------------
# 1) .env 파일 로드
# .env 파일에서 환경 변수를 로드합니다.
# -------------------------------------------------------------------


load_dotenv()


# -------------------------------------------------------------------
# 2) PDF 파티셔닝 함수 정의
# PDF 파일에서 이미지, 테이블, 그리고 텍스트 조각을 추출합니다.
# -------------------------------------------------------------------

def extract_elements_from_pdf(filepath, table_model=None):
    """
    Extracts elements from a PDF file using specified partitioning strategies.
    
    Args:
        filepath (str): The path to the PDF file to be processed.
        table_model (str, optional): The table detection model to use. 
                                   Options: "yolox", "table-transformer", "tatr"
                                   If None, infer_table_structure will be disabled.

    Returns:
        list: A list of extracted elements from the PDF.

    Keyword Args:
        filename (str): The path to the PDF file to be processed.
        extract_images_in_pdf (bool): Whether to extract images from the PDF. This flag utilize GPU resources when available for improved performance in image recognition and structure inference.  Defaults to False. 
        infer_table_structure (bool): Whether to infer table structures in the PDF. This flag utilize GPU resources when available for improved performance in image recognition and structure inference. Defaults to False.
        chunking_strategy (str): The strategy to use for chunking text. Defaults to "by_title".
        max_characters (int): The maximum number of characters in a chunk. Defaults to 4000.
        new_after_n_chars (int): The number of characters after which a new chunk is created. Defaults to 3800.
        combine_text_under_n_chars (int): The number of characters under which text is combined into a single chunk. Defaults to 2000.
    """
    # 테이블 모델 옵션 설정
    partition_kwargs = {
        "filename": filepath,
        "extract_images_in_pdf": True,
        "chunking_strategy": "by_title",  #"by_page"
        #"page_numbers": list(range(1, 7)),  # 1~6 페이지 명시
        "max_characters": 4000,
        "new_after_n_chars": 3800,
        "combine_text_under_n_chars": 2000,
        #"batch_size": 10,  # 한 번에 처리할 페이지 수 (메모리 사용량 조절)
        "extract_image_block_output_dir": "figures",  # 이미지 추출 디렉토리 지정
    }
    
    # 테이블 모델이 지정된 경우에만 테이블 구조 추론 활성화
    if table_model:
        partition_kwargs["infer_table_structure"] = True
        partition_kwargs["table_model"] = table_model
    else:
        partition_kwargs["infer_table_structure"] = False
        
    return partition_pdf(**partition_kwargs)

def encode_image_to_base64(image_path):
    """
    이미지 파일을 base64로 인코딩합니다.
    
    Args:
        image_path (str): 인코딩할 이미지 파일 경로
        
    Returns:
        str: base64로 인코딩된 이미지 데이터
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        print(f"이미지 인코딩 에러 {image_path}: {e}")
        return None

def get_extracted_images(figures_dir="figures"):
    """
    figures 디렉토리에서 추출된 이미지 파일들을 찾습니다.
    
    Args:
        figures_dir (str): 이미지가 저장된 디렉토리
        
    Returns:
        list: 이미지 파일 경로 리스트
    """
    if not os.path.exists(figures_dir):
        return []
    
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(figures_dir, extension)))
    
    return sorted(image_files)


# -------------------------------------------------------------------
# 3) 프롬프트 템플릿 정의
# -------------------------------------------------------------------
# 텍스트용 프롬프트
text_prompt = PromptTemplate.from_template(
    """Context information is below. You are only aware of this context and nothing else.
---------------------

{context}

---------------------
Given this context, generate only questions based on the below query.
You are an Teacher/Professor in {domain}. 
Your task is to provide exactly **{num_questions}** question(s) for an upcoming quiz/examination. 
You are not to provide more or less than this number of questions. 
The question(s) should be diverse in nature across the document. 
The purpose of question(s) is to test the understanding of the students on the context information provided.
You must also provide the answer to each question. The answer should be based on the context information provided only.

Restrict the question(s) to the context information provided only.
QUESTION and ANSWER should be written in Korean. response in JSON format which contains the `question` and `answer`.
DO NOT USE List in JSON format.
ANSWER should be a complete sentence.

#Format:
```json
{{
    "QUESTION": "테슬라가 공개한 차세대 로봇 '옵티머스 2.0'의 핵심 개선점 중 하나는 무엇입니까?",
    "ANSWER": "테슬라가 공개한 차세대 로봇 옵티머스 2.0의 핵심 개선점은 자체 설계한 근전도 센서를 활용해 정밀한 손동작을 구현한 것입니다."
}},
{{
    "QUESTION": "오픈AI가 발표한 GPT-5 연구 방향에서 가장 강조된 목표는 무엇입니까?",
    "ANSWER": "오픈AI가 발표한 GPT-5 연구 방향에서 가장 강조된 목표는 장기적 추론 능력 향상입니다."
}},
{{
    "QUESTION": "파이낸셜 타임즈 보고서에 따르면 2030년까지 글로벌 양자컴퓨팅 시장 규모는 얼마로 예상되나요?",
    "ANSWER": "파이낸셜 타임즈 보고서에 따르면 2030년까지 글로벌 양자컴퓨팅 시장 규모는 125억 달러로 예상됩니다."
}}
```
"""
)

# 이미지용 프롬프트 메시지 생성 함수
def create_image_prompt_message(image_base64, domain, num_img_questions, image_path=""):
    """
    이미지 분석을 위한 프롬프트 메시지를 생성합니다.
    
    Args:
        image_base64 (str): base64로 인코딩된 이미지 데이터
        domain (str): 분야/도메인
        num_questions (str): 생성할 질문 수
        image_path (str): 이미지 파일 경로 (포맷 감지용)
        
    Returns:
        HumanMessage: 이미지와 텍스트가 포함된 메시지
    """
    # 이미지 확장자로부터 포맷 감지
    image_format = "png"  # 기본값
    if image_path:
        ext = os.path.splitext(image_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            image_format = "jpeg"
        elif ext in ['.png']:
            image_format = "png"
        elif ext in ['.gif']:
            image_format = "gif"
        elif ext in ['.bmp']:
            image_format = "bmp"
    
    return HumanMessage(content=[
        {
            "type": "text",
            "text": f"""
Analyze this image and generate question-answer pairs.

You are a professor/teacher in the {domain} field.
Your task is to create exactly **{num_img_questions}** questions for an upcoming quiz/exam.
You must not create more or fewer questions than this number.

**MANDATORY RULES - VIOLATION WILL RESULT IN FAILURE:**
1. **EXACT DATA ONLY**: Use ONLY the exact numbers, dates, and text visible in the image. Do NOT interpret, convert, or modify any values.
2. **PRECISE READING**: Read dates, numbers, and labels character-by-character as they appear. For example, if you see "12.3일", it means December 3rd, NOT November 13th.
3. **NO ASSUMPTIONS**: Do not assume relationships, trends, or meanings beyond what is explicitly shown.
4. **VERIFY BEFORE WRITING**: Before writing each answer, mentally point to the exact location in the image where that information appears.
5. **CONSERVATIVE APPROACH**: If you cannot clearly read a specific value or date, do not create a question about it.

**DATA ACCURACY REQUIREMENTS:**
- Charts/Graphs: Only reference data points where both X-axis (date/time) AND Y-axis (value) are clearly visible
- Tables: Only reference cells where both row and column headers are clear
- Text: Only reference text that is completely legible
- Numbers: Copy numbers exactly as shown (including decimal points, units like bp, %, etc.)

**FORBIDDEN ACTIONS:**
- Converting date formats (e.g., 12.3 ≠ 11.13)
- Estimating values between data points
- Creating questions about unclear or partially visible content
- Using information from chart legends if the actual data is unclear

**Question Types to Focus On:**
- Direct reading of clearly visible data points
- Identification of clearly labeled chart/table elements
- Reading of section titles, page numbers, or menu items
- Comparison of clearly visible values (highest, lowest, specific dates)

Write questions and answers in Korean and respond in JSON format.
Do not use arrays/lists in the JSON format.


#Format:
```json
{{
    "QUESTION": "CDS 프리미엄 차트에서 12월 17일의 수치는 얼마입니까?",
    "ANSWER": "CDS 프리미엄 차트에서 12월 17일의 수치는 36.3bp입니다."
}},
{{
    "QUESTION": "목차에서 개선 방안의 첫 번째 항목은 무엇입니까?",
    "ANSWER": "목차에서 개선 방안의 첫 번째 항목은 '건전성 규제 완화'입니다."
}},
{{
    "QUESTION": "9월 전후 지표 악화 차트에서 외환 차익거래유인 최고점은 언제 기록되었습니까?",
    "ANSWER": "9월 전후 지표 악화 차트에서 외환 차익거래유인 최고점은 10월 2일경에 기록되었습니다."
}}
```
"""                
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": f"image/{image_format}",
                "data": image_base64
            }
        }
    ])


#-------------------------------------------------------------------
# 4) AWS Bedrock(Claude) 클라이언트 설정
# Get AWS credentials from environment variables
# Extracting Sentences from PDF to JSONL with AWS Bedrock Claude which is a LLM Professor Personality prepared test questions.
#-------------------------------------------------------------------

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")  # Default to us-east-1 if not specified
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
# Add debugging to check environment variables
print(f"AWS_REGION: {AWS_REGION}")
print(f"Extracting Sentences from PDF to JSONL with AWS Bedrock Claude which is a LLM Professor Personality prepared test questions.")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.environ.get('AWS_SESSION_TOKEN')

bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    aws_session_token=AWS_SESSION_TOKEN
)
#-------------------------------------------------------------------
# see https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
#--------------------------------------------------------------------
# Bedrock Claude 모델 설정
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    #model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    #model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",
    client=bedrock_client,
    model_kwargs={
        "temperature": 0,
        "max_tokens": 2000,
    },
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

    
    
#-------------------------------------------------------------------
# 5) 체인 구성
# def format_docs(input_dict): """ 체인에 전달할 입력을 표준화합니다. """ 
# def custom_json_parser(response): """ ChatBedrock 모델의 응답(문자열) 중 JSON 포맷 부분을 찾아 파싱합니다. """ 
# chain : Runnable 파이프라인(체인) 정의, RunnablePassthrough -> Prompt -> LLM -> StrOutputParser -> custom_json_parser 
#-------------------------------------------------------------------

def custom_json_parser(response):
    if hasattr(response, 'content'):
        response = response.content
    
    try:
        start = response.find('```json') + 7 if '```json' in response else 0
        end = response.find('```', start) if '```' in response[start:] else len(response)
        json_text = response[start:end].strip()
        json_text = json_text.rstrip(',')
        json_text = f'[{json_text}]'
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 에러: {e}")
        print(f"파싱 시도한 텍스트: {json_text}")
        return []

def format_docs(input_dict):
    return {
        "context": input_dict["context"],
        "domain": input_dict["domain"],
        "num_questions": input_dict["num_questions"]
    }

# 텍스트 처리용 체인
text_chain = (
    RunnablePassthrough(format_docs)
    | text_prompt
    | llm
    | StrOutputParser()
    | custom_json_parser
)

# 이미지 처리 함수
def process_image_with_llm(image_path, domain, num_img_questions):
    """
    이미지를 LLM으로 처리해서 Q&A 생성
    
    Args:
        image_path (str): 이미지 파일 경로
        domain (str): 분야/도메인
        num_img_questions (str): 생성할 질문 수
        
    Returns:
        list: 생성된 Q&A 쌍 리스트
    """
    try:
        # 이미지를 base64로 인코딩
        image_base64 = encode_image_to_base64(image_path)
        if not image_base64:
            return []
        
        # 이미지 프롬프트 메시지 생성
        message = create_image_prompt_message(image_base64, domain, num_img_questions, image_path)
        
        # LLM으로 이미지 처리
        response = llm.invoke([message])
        
        # 응답을 JSON으로 파싱
        parsed_response = custom_json_parser(response)
        
        # 이미지 소스 정보 추가
        for qa in parsed_response:
            qa['source'] = 'image'
            qa['image_path'] = os.path.basename(image_path)
        
        print(f"이미지 처리 완료: {os.path.basename(image_path)} - {len(parsed_response)}개 Q&A 생성")
        return parsed_response
        
    except Exception as e:
        print(f"이미지 처리 에러 {image_path}: {e}")
        return []


def main():
    
    # Get PDF path from environment variable or use default
    pdf_path = os.environ.get("PDF_PATH", "data/fsi_data.pdf")
    print(f"Processing PDF from: {pdf_path}")
    
    # Get domain from environment variable or use default
    domain = os.environ.get("DOMAIN", "International Finance")
    print(f"Using domain: {domain}")
    
    # Get number of questions from environment variable or use default
    num_questions = os.environ.get("NUM_QUESTIONS", "5")
    print(f"Number of questions to generate: {num_questions}")
    
    # Get number of image questions from environment variable or use default
    num_img_questions = os.environ.get("NUM_IMG_QUESTIONS", "1")
    
    # Get table model from environment variable or use default (None)
    table_model = os.environ.get("TABLE_MODEL", None)
    print(f"Table model: {table_model if table_model else 'None (disabled)'}")
    
    # 6-1) PDF 파일에서 요소 추출     
    elements = extract_elements_from_pdf(pdf_path, table_model=table_model)
    print(f"추출된 요소 수: {len(elements)}")
    
    # 6-2) 추출된 텍스트 요소 각각에 대해 text_chain 실행
    qa_pairs = []
    text_count = 0
    
    print("\n=== 텍스트 요소 처리 시작 ===")
    for element in elements:
        if element.text and element.text.strip():  # 빈 텍스트 제외
            try:
                response = text_chain.invoke({
                    "context": element.text,                
                    "domain": domain,
                    "num_questions": num_questions
                })
                qa_pairs.extend(response)
                text_count += 1
                print(f"텍스트 요소 {text_count} 처리 완료 - {len(response)}개 Q&A 생성")
            except Exception as e:
                print(f"텍스트 요소 처리 에러: {e}")
    
    print(f"텍스트 처리 완료: 총 {text_count}개 요소에서 {len(qa_pairs)}개 Q&A 생성")
    
    # 6-3) 추출된 이미지들 처리
    print("\n=== 이미지 요소 처리 시작 ===")
    image_files = get_extracted_images("figures")
    image_count = 0
    
    if image_files:
        print(f"발견된 이미지 파일: {len(image_files)}개")
        for image_path in image_files:
            image_qa = process_image_with_llm(image_path, domain, num_img_questions)
            qa_pairs.extend(image_qa)
            if image_qa:
                image_count += 1
    else:
        print("추출된 이미지가 없습니다.")
    
    print(f"이미지 처리 완료: 총 {image_count}개 이미지에서 {sum(1 for qa in qa_pairs if qa.get('source') == 'image')}개 Q&A 생성")
    
    # 6-4) JSONL로 결과 저장
    os.makedirs("data", exist_ok=True)  # data 디렉토리가 없으면 생성
    
    with open("data/qa_pairs.jsonl", "w", encoding='utf-8') as f:
        for item in qa_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n[INFO] QA 생성 완료! 총 {len(qa_pairs)}개 Q&A가 data/qa_pairs.jsonl 파일에 저장되었습니다.")
    print(f"- 텍스트에서 생성: {len([qa for qa in qa_pairs if qa.get('source') != 'image'])}개")
    print(f"- 이미지에서 생성: {len([qa for qa in qa_pairs if qa.get('source') == 'image'])}개")


if __name__ == "__main__":
    
    main()