# AI-Powered 프레젠테이션 생성 with Prompt Chaining

## 개요
이 저장소에는 Amazon Bedrock 런타임에서 Anthropic의 Claude 3.5 Sonnet 모델을 활용해 프레젠테이션의 내용을 자동으로 작성하는 Python 예제 코드가 포함되어 있습니다. Prompt chaining 기법과 고급 프롬프트 기법을 활용해 긴 프레젠테이션 생성이 가능한 코드를 제공하고, cross region inference 기능 활용하는 방법도 제공합니다. 

## 주요 기능
* 템플릿 기반 프롬프트 생성 및 관리
* Claude 3.5 Sonnet 모델을 활용한 프레젠테이션 생성
* 전체 프레젠테이션 개요 및 개별 슬라이드 내용 생성
* 인풋 토큰, 아웃풋 토큰 사용량 추적
* Prompt Chaining을 통한 대용량 프레젠테이션 내용 생성
* Zero-shot prompting 기반의 고급 프롬프트 기법 활용
    * Role-based prompting: 특정 역할을 가정한 프롬프트 생성
    * Task-specific instructions: 구체적인 작업 지침 제공
    * Content structuring: 체계적인 콘텐츠 구조화
    * Output constraints: 출력 형식 및 내용에 대한 제약 설정
* Amazon Bedrock Cross region inference 기능을 활용한 추론
* 생성된 콘텐츠의 파일 저장 및 병합

## 문제 상황
2024/08 기준, Amazon Bedrock에서 Claude 3.5 Sonnet 모델은 단일 호출로 최대 4,096개의 아웃풋 토큰을 생성할 수 있습니다. 이보다 긴 내용을 생성하기 위해서는 여러 번의 API 호출이 필요합니다.

## 해결 방법
20장 분량의 고품질 프레젠테이션 내용을 생성하기 위해 Prompt Chaining 기법을 적용합니다. 이 과정은 다음과 같이 3단계로 구성됩니다.
각 단계의 결과를 다음 단계의 입력으로 사용하여 전체 프레젠테이션의 일관성을 유지합니다.
1. 전체 프레젠테이션 개요 작성
2. 개요에 기반한 1-10번 슬라이드 내용 생성
3. 개요에 기반한 11-20번 슬라이드 내용 생성


## 요구 사항
* Python 3.7 이상
* AWS 계정 및 자격 증명
* AWS CLI 설치 및 구성

## 사용 방법
1. cross_region_inference_prompt_util.py 파일, generate-powerpoint.ipynb 파일 설정
* Cross-region inference 기능을 사용할 경우
    * cross_region_inference_prompt_util.py 파일에서 primary_region과 inferenceProfileId 설정
    * generate-powerpoint.ipynb 파일에서 cross_region_inference_prompt_util.py 파일을 import
* 단일 리전에서 추론할 경우 
    * generate-powerpoint.ipynb 파일에서 prompt_util.py 파일을 import
2. generate-powerpoint.ipynb 파일 실행
3. 생성된 콘텐츠는 지정된 출력 디렉토리(results 폴더)에 저장됨

## 주요 함수
* invoke_claude(): Claude 모델 호출 및 텍스트 생성
* create_slide_prompt(): 슬라이드 생성용 프롬프트 작성
* generate_slide_content(): 개별 슬라이드 내용 생성
* generate_all_slide_content(): 전체 프레젠테이션 내용 생성
* merge_files(): 생성된 파일들을 하나로 병합

## 추가 리소스
* [Anthropic Claude 설명서](https://docs.anthropic.com/claude/docs/intro-to-claude)
* [AWS Bedrock 런타임 설명서](https://docs.aws.amazon.com/ko_kr/bedrock/latest/userguide/service_code_examples_bedrock-runtime.html)
* [Cross-region inference in Amazon Bedrock 설명서](https://aws.amazon.com/blogs/machine-learning/getting-started-with-cross-region-inference-in-amazon-bedrock/)