# Bedrock-Manus Streamlit UI

Amazon Bedrock 기반 AI 자동화 프레임워크인 Bedrock-Manus를 위한 Streamlit 기반 사용자 인터페이스입니다.

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. 기존 Bedrock-Manus 프레임워크가 설치되어 있어야 합니다.

## 실행 방법

다음 명령어로 Streamlit 앱을 실행합니다:
```bash
streamlit run app.py
```

## 주요 기능

- 사용자 쿼리 입력을 위한 텍스트 영역
- Debug 모드 토글 (사이드바)
- Artifacts 폴더 정리 기능 (사이드바)
- 대화 기록 표시 (확장 가능한 섹션)
- 생성된 아티팩트 파일 다운로드 기능

## UI 구성

- 상단: 제목 및 소개
- 좌측 사이드바: 설정 옵션
- 메인 영역: 쿼리 입력 및 결과 표시
- 하단: 대화 기록 및 아티팩트 다운로드
