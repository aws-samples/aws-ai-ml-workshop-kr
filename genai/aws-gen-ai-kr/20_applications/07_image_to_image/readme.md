# How to run this application

![Image2image](https://github.com/aws-samples/aws-ai-ml-workshop-kr/assets/48855293/918d53b6-9682-41fa-9b5d-4f5575411e20)

## Structure

1. `bedrock.py`

- Amazon Bedrock에서 stable diffusion model을 불러오고 image to image 생성합니다.

2. `streamlit.py`

- 애플리케이션의 front-end 파일, 실행 시 `bedrock.py`을 import해서 사용합니다.

## Start

1. 폴더 접근

```
cd aws-ai-ml-workshop-kr/genai/aws-gen-ai-kr/20_applications/02_qa_chatbot/07_image_to_image/
```

2. Python 종속 라이브러리 설치

```
pip install -r requirements.txt
```

3. Streamlit 애플리케이션 작동

```
streamlit run streamlit.py --server.port 8080
```

3. 접속하기

- Streamlit 작동 시 표시되는 External link로 접속

## Application

1. 생성하고자 하는 기반 이미지를 업로드합니다.
2. 기반 이미지에 수정을 원하는 프롬프트를 입력합니다.
3. Submit을 눌러 제출하고 기다리면 이미지가 생성됩니다.
