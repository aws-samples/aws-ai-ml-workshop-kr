# Bedrock-Manus Streamlit UI 개발 요약

## 프로젝트 소개

이 프로젝트는 "Bedrock-Manus: Amazon Bedrock 최적화 AI 자동화 프레임워크"에 대한 Streamlit 기반 사용자 인터페이스를 개발한 것입니다. 이 UI는 사용자가 쉽게 Bedrock-Manus 프레임워크의 기능을 활용할 수 있도록 설계되었습니다.

## 개발 내용

### 1. 프레임워크 분석
- Bedrock-Manus 프로젝트 구조 분석
- 주요 모듈 및 기능 파악 (workflow.py, graph.py 등)
- 에이전트 기반 워크플로우 시스템 이해

### 2. Streamlit UI 구현
- 사용자 친화적인 웹 인터페이스 설계
- 입력, 처리, 결과 표시 워크플로우 구현
- 아티팩트 파일 관리 및 다운로드 기능 구현
- 대화 기록 조회 기능 구현

### 3. 시각적 요소 개선
- 모던한 디자인 적용 (CSS 스타일링)
- 직관적인 UI 구성 (헤더, 사이드바, 메인 영역)
- 진행 상태 표시 기능 추가

### 4. 문서화
- README.md: 프로젝트 개요 및 구성 설명
- INSTALL.md: 설치 가이드
- USAGE.md: 사용자 매뉴얼
- SUMMARY.md: 프로젝트 요약

## 파일 목록

1. **app.py**: Streamlit UI 애플리케이션 메인 코드
2. **requirements.txt**: 필요한 Python 패키지 목록
3. **README.md**: 프로젝트 개요
4. **INSTALL.md**: 설치 가이드
5. **USAGE.md**: 사용자 매뉴얼
6. **SUMMARY.md**: 개발 요약 문서
7. **bedrock_manus_streamlit.tar.gz**: 모든 파일을 포함한 압축 파일

## 주요 기능

- 사용자 쿼리 입력 및 처리
- Debug 모드 토글 (사이드바)
- Artifacts 폴더 관리 기능
- 대화 기록 표시
- 생성된 아티팩트 파일 관리
- 이미지 및 텍스트 파일 미리보기
- 파일 다운로드 기능

## 설치 및 사용 방법

기본적인 설치 과정은 다음과 같습니다:

```bash
# 1. 압축 파일 해제
tar -xzf bedrock_manus_streamlit.tar.gz

# 2. 필요한 패키지 설치
pip install -r requirements.txt

# 3. Streamlit 앱 실행
streamlit run app.py
```

자세한 설치 및 사용 방법은 INSTALL.md 및 USAGE.md 파일을 참조하세요.

## 기술 스택

- Python 3.8+
- Streamlit 1.22.0+
- Langgraph 0.0.10+
- Amazon Bedrock APIs
