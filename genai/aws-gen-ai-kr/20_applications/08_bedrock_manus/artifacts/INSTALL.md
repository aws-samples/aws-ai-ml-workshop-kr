# Bedrock-Manus Streamlit UI 설치 가이드

## 사전 요구사항

1. Python 3.8 이상 설치
2. pip 패키지 관리자
3. Bedrock-Manus 프레임워크

## 설치 단계

### 1. 소스코드 다운로드 및 배치

1. `app.py`, `requirements.txt` 파일을 Bedrock-Manus 프레임워크의 루트 디렉토리에 복사합니다.

```bash
cp app.py /path/to/bedrock-manus/
cp requirements.txt /path/to/bedrock-manus/
```

### 2. 필요한 패키지 설치

```bash
cd /path/to/bedrock-manus/
pip install -r requirements.txt
```

### 3. 환경 설정

Bedrock-Manus 프레임워크에서 사용하는 `.env` 파일이 올바르게 구성되어 있는지 확인하세요.

## 실행 방법

```bash
cd /path/to/bedrock-manus/
streamlit run app.py
```

실행 후 웹 브라우저에서 자동으로 열리는 URL로 접속하거나, 기본적으로 http://localhost:8501 로 접속하세요.

## 문제 해결

### 오류: "Bedrock-Manus 프레임워크를 불러올 수 없습니다."

이 오류는 `app.py`가 `src.workflow` 모듈을 찾을 수 없을 때 발생합니다. 다음을 확인하세요:

1. `app.py` 파일이 Bedrock-Manus 프레임워크의 루트 디렉토리에 위치해 있는지 확인
2. `src` 디렉토리가 존재하고 그 안에 `workflow.py` 파일이 있는지 확인
3. Python 경로 설정이 올바른지 확인

### 다른 오류가 발생하는 경우

1. 모든 필요한 패키지가 설치되어 있는지 확인하세요.
2. Streamlit 버전이 1.22.0 이상인지 확인하세요.
3. Langgraph 버전이 0.0.10 이상인지 확인하세요.
