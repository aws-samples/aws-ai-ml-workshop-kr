# processing.py - SageMaker Processing Job용 스크립트
from unstructured.partition.pdf import partition_pdf
from langchain_core.prompts import PromptTemplate
from langchain_aws import ChatBedrock
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import HumanMessage
import json 
import boto3
import os
import argparse  # 추가: 명령줄 인자 처리를 위한 모듈
import base64
import glob

"""
SageMaker Processing Job용 PDF 파싱 및 질문 생성 스크립트

이 스크립트는 다음 단계로 이루어집니다:
1) PDF 파티셔닝(텍스트/이미지/테이블 등 추출)
2) Prompt Template 정의
3) AWS Bedrock(Claude) 클라이언트 설정
4) 체인 구성(Runnable 파이프라인)
5) PDF 요소들에 대해 체인 실행 & 질의응답 생성
6) 최종 JSONL 파일로 S3에 저장

processing_local.py와 다른점은 LLM 객체 생성 부분을 전역에서 함수 내부로 이동하고 domain, num_questions, model_id을 인자로 받을 수 있도록 argparse 및 main 함수 수정

"""

# -------------------------------------------------------------------
# 1) SageMaker Processing Job용 경로 설정
# -------------------------------------------------------------------
# SageMaker Processing Job의 기본 경로
INPUT_DIR = '/opt/ml/processing/input'
OUTPUT_DIR = '/opt/ml/processing/output'

# 입출력 파일 경로 설정
PDF_FILE_PATH = os.path.join(INPUT_DIR, 'pdf/fsi_data.pdf')
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, 'qa_pairs.jsonl')

# 출력 디렉토리가 없으면 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"입력 디렉토리: {INPUT_DIR}")
print(f"출력 디렉토리: {OUTPUT_DIR}")
print(f"PDF 파일 경로: {PDF_FILE_PATH}")
print(f"출력 파일 경로: {OUTPUT_FILE_PATH}")

# -------------------------------------------------------------------
# 2) PDF 파티셔닝 함수 정의
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
        
    Available table_model options:
        - None: Disable table detection
        - "yolox": Default YOLOX (fast)
        - "yolox_quantized": Quantized YOLOX (ultra-fast)
        - "table-transformer": Microsoft model (high accuracy)
        - "tatr": Table Transformer improved (balanced)
        - "detectron2": Meta model (highest accuracy)
        - "detectron2_onnx": Meta ONNX (optimized)
        - "paddle": PaddleOCR (Chinese specialized)
    """
    # 테이블 모델 옵션 설정
    partition_kwargs = {
        "filename": filepath,
        "extract_images_in_pdf": True,
        "chunking_strategy": "by_title",  #see : https://docs.unstructured.io/api-reference/partition/chunking
        #"page_numbers": list(range(1, 7)),  # 1~6 페이지 명시
        "max_characters": 4000,
        "new_after_n_chars": 3800,
        "combine_text_under_n_chars": 2000,
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
        num_img_questions (str): 생성할 질문 수
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

# -------------------------------------------------------------------
# 4) 유틸리티 함수 정의
# -------------------------------------------------------------------
# SageMaker는 실행 역할의 권한을 사용하므로 별도의 자격증명이 필요 없음

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
print(f"AWS 리전: {AWS_REGION}")

def custom_json_parser(response):
    """
    응답에서 JSON 형식의 텍스트를 추출하고 파싱합니다.
    
    Args:
        response (str|obj): LLM의 응답
        
    Returns:
        list: 파싱된 JSON 객체 목록
    """
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
    """
    체인에 전달할 입력을 형식화합니다.
    
    Args:
        input_dict (dict): 입력 딕셔너리
        
    Returns:
        dict: 형식화된 입력 딕셔너리
    """
    return {
        "context": input_dict["context"],
        "domain": input_dict["domain"],
        "num_questions": input_dict["num_questions"]
    }

# 이미지 처리 함수
def process_image_with_llm(image_path, domain, num_img_questions, llm):
    """
    이미지를 LLM으로 처리해서 Q&A 생성
    
    Args:
        image_path (str): 이미지 파일 경로
        domain (str): 분야/도메인
        num_img_questions (str): 생성할 질문 수
        llm: LLM 객체
        
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







# -------------------------------------------------------------------
# 6) 메인 실행 함수
# -------------------------------------------------------------------
def main(domain="International Finance", num_questions="5", num_img_questions="1", model_id="anthropic.claude-3-5-sonnet-20240620-v1:0", table_model=None): #for arguments parsing
    print("PDF 질문 생성 작업 시작...")
    print(f"도메인: {domain}, 텍스트 질문 수: {num_questions}, 이미지 질문 수: {num_img_questions}, 모델: {model_id}")
    
    try:
        # AWS Bedrock 클라이언트 설정
        print("AWS Bedrock 클라이언트 설정 중...")
        bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=AWS_REGION
        )
        # Bedrock Claude 모델 설정
        llm = ChatBedrock(
            model_id=model_id,  # 전달받은 model_id 사용            
            client=bedrock_client,
            model_kwargs={
                "temperature": 0,
                "max_tokens": 2000,
            },
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # -------------------------------------------------------------------
        # 5) 체인 구성 : Runnable 파이프라인 정의
        # -------------------------------------------------------------------

        # 텍스트 처리용 체인 구성
        text_chain = (
            RunnablePassthrough(format_docs)
            | text_prompt
            | llm
            | StrOutputParser()
            | custom_json_parser
        )
        
        # 1) PDF 파일에서 요소 추출
        elements = extract_elements_from_pdf(PDF_FILE_PATH, table_model=table_model)
        print(f"추출된 요소 수: {len(elements)}")
        
        # 2) 추출된 텍스트 요소 각각에 대해 text_chain 실행
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
        
        # 3) 추출된 이미지들 처리
        print("\n=== 이미지 요소 처리 시작 ===")
        image_files = get_extracted_images("figures")
        image_count = 0
        
        if image_files:
            print(f"발견된 이미지 파일: {len(image_files)}개")
            for image_path in image_files:
                image_qa = process_image_with_llm(image_path, domain, num_img_questions, llm)
                qa_pairs.extend(image_qa)
                if image_qa:
                    image_count += 1
        else:
            print("추출된 이미지가 없습니다.")
        
        print(f"이미지 처리 완료: 총 {image_count}개 이미지에서 {sum(1 for qa in qa_pairs if qa.get('source') == 'image')}개 Q&A 생성")
        
        print(f"\n총 {len(qa_pairs)}개의 QA 쌍 생성 완료")
        
        # 4) JSONL로 결과 저장
        with open(OUTPUT_FILE_PATH, "w", encoding='utf-8') as f:
            for item in qa_pairs:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\n[INFO] QA 생성 완료! 총 {len(qa_pairs)}개 Q&A가 {OUTPUT_FILE_PATH}에 저장되었습니다.")
        print(f"- 텍스트에서 생성: {len([qa for qa in qa_pairs if qa.get('source') != 'image'])}개")
        print(f"- 이미지에서 생성: {len([qa for qa in qa_pairs if qa.get('source') == 'image'])}개")
        
    except Exception as e:
        print(f"처리 중 오류 발생: {str(e)}")
        raise




if __name__ == "__main__":
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='PDF에서 QA 쌍을 생성하는 스크립트')
    parser.add_argument('--domain', type=str, default='International Finance',
                        help='QA 생성을 위한 도메인 (예: "International Finance", "외환정책")')
    parser.add_argument('--num_questions', type=str, default='5',
                        help='각 PDF 텍스트 요소마다 생성할 질문 수')
    parser.add_argument('--num_img_questions', type=str, default='1',
                        help='각 이미지마다 생성할 질문 수')
    parser.add_argument('--model_id', type=str, default='anthropic.claude-3-5-sonnet-20240620-v1:0',
                        help='사용할 Bedrock 모델 ID')
    parser.add_argument('--table_model', type=str, default='yolox',
                        help='테이블 구조 추론 모델 (yolox, table-transformer, tatr, detectron2, etc). None이면 테이블 구조 추론 비활성화')
    
    args = parser.parse_args()
    
    # 파싱된 인자를 main 함수에 전달
    main(domain=args.domain, num_questions=args.num_questions, num_img_questions=args.num_img_questions, model_id=args.model_id, table_model=args.table_model)
