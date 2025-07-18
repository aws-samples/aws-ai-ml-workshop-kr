# 워크샵: AI 메뉴보드 생성기 만들기

## 학습 목표
이 워크샵을 통해 다음을 학습합니다:
- Amazon Bedrock Nova Canvas를 사용한 이미지 생성
- Amazon Bedrock Nova Pro를 사용한 텍스트 번역
- Streamlit을 사용한 웹 애플리케이션 개발
- PIL을 사용한 이미지 처리 및 텍스트 오버레이

## 사전 준비사항
- Python 3.8 이상
- AWS 계정 및 자격 증명 설정
- Amazon Bedrock Nova 모델 액세스 권한

## 워크샵 단계

### 1단계: 환경 설정
```bash
# 가상 환경 생성
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate     # Windows

# 의존성 설치
python setup.py
```

### 2단계: AWS 자격 증명 설정
```bash
aws configure
```

### 3단계: Amazon Bedrock 모델 액세스 설정
1. AWS 콘솔에서 Amazon Bedrock 서비스로 이동
2. "Model access" 메뉴 선택
3. Nova Canvas와 Nova Pro 모델 액세스 요청

### 4단계: 애플리케이션 실행
```bash
streamlit run app.py
```

## 주요 코드 구조

### 1. AWS 클라이언트 초기화
```python
@st.cache_resource
def init_aws_clients():
    session = boto3.Session()
    bedrock = session.client('bedrock-runtime', region_name='us-east-1')
    return bedrock
```

### 2. 텍스트 번역 (Nova Pro)
```python
def translate_to_english(korean_text, bedrock_client):
    prompt = f"Translate the following Korean text to English: {korean_text}"
    # Nova Pro 모델 호출
```

### 3. 이미지 생성 (Nova Canvas)
```python
def generate_menuboard_image(prompt, bedrock_client):
    # Nova Canvas 모델 호출하여 이미지 생성
```

### 4. 텍스트 오버레이
```python
def add_text_overlay(image, text, position, font_size):
    # PIL을 사용한 텍스트 오버레이
```

## 실습 과제

### 기본 과제
1. 카페 메뉴보드 생성해보기
2. 레스토랑 메뉴보드 생성해보기
3. 다양한 폰트 크기와 색상 적용해보기

### 심화 과제
1. 다른 언어 번역 기능 추가
2. 이미지 필터 효과 추가
3. 메뉴 카테고리별 섹션 나누기
4. QR 코드 추가 기능 구현

## 문제 해결
일반적인 문제와 해결 방법은 README.md의 문제 해결 섹션을 참조하세요.

## 추가 학습 자료
- [Amazon Bedrock 개발자 가이드](https://docs.aws.amazon.com/bedrock/)
- [Streamlit 문서](https://docs.streamlit.io/)
- [PIL 문서](https://pillow.readthedocs.io/)
