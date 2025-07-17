<h1 align="left"><b>Amazon Bedrock 기반 신약 개발 에이전트</b></h1>

- - -

## 개요

신약 개발 에이전트는 제약 연구자들이 과학 문헌, 임상 시험, 그리고 약물 데이터베이스를 탐색하는 데 도움을 주는 AI 기반 에이전트입니다. 이 도구는 Amazon Bedrock의 대규모 언어 모델을 활용하여 신약 개발, 표적 단백질, 질병 및 관련 연구에 대한 대화형 상호작용을 제공합니다.

## 주요 기능

- **대화형 채팅 인터페이스**: 신약 개발 주제에 대한 자연어 대화 참여
- **다양한 데이터 소스**: 여러 과학 데이터베이스의 정보 접근:
  - arXiv (과학 논문)
  - PubMed (생의학 문헌)
  - ChEMBL (생물활성 분자)
  - ClinicalTrials.gov (임상 시험)
  - Tavily를 통한 웹 검색

- **종합적 분석**: 다음에 대한 상세한 정보 제공:
  - 표적 단백질과 그 억제제
  - 질병 메커니즘
  - 약물 후보와 그 특성
  - 임상 시험 결과
  - 최신 연구 결과

## 시작하기

### 사전 요구사항
- 필요한 Python 패키지 (`pip install -r requirements.txt`로 설치)
- AWS 자격 증명 구성
- 외부 서비스용 API 키 (Tavily)

### 설치

1. 이 저장소를 클론합니다
2. 종속성을 설치합니다:
   ```
   pip install -r requirements.txt
   ```
3. 환경 변수를 설정합니다:
   - `.env.example`을 `.env`로 복사
   - `.env` 파일에 API 키를 추가
   - 사용할 폰트의 ttf 파일을 다운로드하여 `assets/`로 이동하고 `chat.py`의 `font_path`를 변경

### 애플리케이션 실행

1. MCP 서버 시작 (외부 데이터 소스에 연결하는 Model Context Protocol 서버):
   ```
   python application/launcher.py
   ```
   이 명령은 arXiv, PubMed, ChEMBL, ClinicalTrials.gov, Tavily를 위한 모든 필요한 MCP 서버를 실행합니다.

2. Streamlit 웹 인터페이스 시작:
   ```
   streamlit run application/app.py
   ```

3. 브라우저를 열고 터미널에 표시된 URL로 이동 (일반적으로 http://localhost:8501)

## 신약 개발 에이전트 사용법

1. **모델 선택**: 사용 가능한 기반 모델 중 선택 (Claude 4.0 Sonnet, Claude 3.7 Sonnet, Claude 3.5 Sonnet, 또는 Claude 3.5 Haiku)

2. **질문하기**: 다음과 같은 질문 예시:
   - "최근 뉴스, 최신 연구, 관련 화합물, 진행 중인 임상 시험을 포함한 HER2에 대한 보고서를 생성해주세요."
   - "BRCA1 억제제에 대한 최근 연구 논문을 찾아주세요"
   - "코로나바이러스 단백질을 표적으로 하는 가장 유망한 약물 후보는 무엇인가요?"
   - "HER2 표적 치료법의 작용 메커니즘을 요약해주세요"
   
3. **보고서 생성**: 에이전트는 특정 표적이나 질병에 대한 종합적인 보고서를 작성할 수 있습니다

## 아키텍처

신약 개발 에이전트는 다음을 사용하여 구축되었습니다:

- **Strands Agent SDK**: 특정 기능을 가진 AI 에이전트 생성
- **Streamlit**: 웹 인터페이스
- **MCP (Model Context Protocol)**: 외부 데이터 소스 연결
- **Amazon Bedrock**: Claude와 같은 강력한 언어 모델 접근

각 MCP 서버는 다양한 과학 데이터베이스에 접근하기 위한 전문화된 도구를 제공합니다:
- `mcp_server_arxiv.py`: arXiv에서 과학 논문 검색 및 검색
- `mcp_server_chembl.py`: ChEMBL에서 화학 및 생물활성 데이터 접근
- `mcp_server_clinicaltrial.py`: 임상 시험 검색 및 분석
- `mcp_server_pubmed.py`: PubMed에서 생의학 문헌 접근
- `mcp_server_tavily.py`: 최신 정보를 위한 웹 검색 수행

## 제한사항
- 이 저장소는 개념 증명(PoC) 및 데모 목적으로만 제작되었습니다. 상업적 또는 프로덕션 용도로는 사용하지 마세요.
- 에이전트는 속도 제한이 있을 수 있는 외부 API에 의존합니다
- 정보는 연결된 데이터베이스에서 사용 가능한 것으로 제한됩니다

## 향후 개선사항
- 추가 신약 개발 도구 및 데이터베이스와의 통합
- 분자 구조 및 상호작용의 향상된 시각화
- 독점 연구 데이터베이스 지원

## 기여자
- 류하선, Ph.D. (AWS AI/ML 전문 솔루션즈 아키텍트) | [메일](mailto:hasunyu@amazon.com) | [LinkedIn](https://www.linkedin.com/in/hasunyu/)
- 신경식 (AWS 솔루션즈 아키텍트)| [메일](mailto:kyungss@amazon.com) | [LinkedIn](https://www.linkedin.com/in/shinks)
- 최지선 (AWS 솔루션즈 아키텍트)| [메일](mailto:jschoii@amazon.com) | [LinkedIn](https://www.linkedin.com/in/jschoii/)

## 인용
- 이 저장소가 유용하다고 생각되시면 별표 ⭐를 주시고 인용해 주세요

## 참고 자료
- [Strands Agents SDK](https://strandsagents.com/0.1.x/)
- [Strands Agents Samples](https://github.com/strands-agents/samples/tree/main)
- [Strands Agents Samples - Korean](https://github.com/kyopark2014/strands-agent)
