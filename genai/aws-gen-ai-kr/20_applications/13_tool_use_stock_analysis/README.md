# 프롬프트 엔지니어링과 Tool Use를 활용한 미국 주식 분석 서비스

## 개요
이 저장소에는 Amazon Bedrock 런타임에서 Anthropic의 Claude 3 Sonnet 모델을 Converse API와 Tool Use를 활용해서 
특정 미국 주식의 전날 거래 데이터를 분석하고, 기술적 지표 기반, 뉴스 헤드라인 기반 시장 감성 분석을 통해 맞춤형 투자 보고서를 작성하는 애플리케이션의 Python 예제 코드가 포함되어 있습니다. 최신 거래 데이터와 기술적 지표 기반, 뉴스 헤드라인 기반 시장 감성 분석은 Alpha Vantage의 API를 사용했습니다. 

## 주요 기능
- [Basic 버전](./01_tool_use_stock_analysis.ipynb) 이 코드는 Alpha Vantage API를 활용하여 사용자가 입력한 주식 종목(ticker)의 주요 데이터를 가져옵니다. 구체적으로 전일 종가, 변동률, 당일 거래 가격 범위(최고가/최저가), 시가, 그리고 거래량 정보를 수집합니다. 이렇게 얻은 데이터는 Claude 3 Sonnet 모델에 전달되어 분석되며, 그 결과를 자연어 형태로 사용자에게 제공합니다.


- [Advanced 버전](./02_tool_use_stock_analysis.ipynb) 이 개선된 코드는 LLM의 활용도를 극대화하기 위해 여러 기능을 추가했습니다. 주요 개선 사항으로는 1) 전일 대비 변동폭과 거래량 등 기술적 지표의 분석 및 해석, 2) 뉴스 헤드라인을 기반으로 한 시장 감성 분석, 3) 시스템 및 사용자 프롬프트의 최적화가 있습니다. 이러한 개선을 통해 코드는 단순한 데이터 조회를 넘어서, Claude의 분석 능력을 최대한 활용하여 투자자들에게 더욱 가치 있고 통찰력 있는 정보를 제공합니다.
  

## Tool Use
프롬프트 엔지니어링은 강력한 도구이지만, 실시간 데이터 접근이나 복잡한 계산과 같은 영역에서는 한계를 보입니다. 이러한 한계를 극복하기 위해서는 새로운 접근 방식이 필요합니다. 그 중 하나가 Tool Use (Function Calling)을 사용하는 방법입니다. Tool Use는 LLM이 웹 검색, 코드 생성 및 실행 등 외부 도구를 활용할 수 있게 해주는 기능입니다. 외부 함수에서 얻은 결과값을 프롬프트에 결합해서 LLM에게 응답을 호출하는 방식입니다. 이를 통해 실시간 데이터 접근이나 복잡한 계산 문제를 해결할 수 있습니다. 

- 실시간 데이터 통합
  - API를 통한 최신 정보 접근
  - 외부 데이터베이스 연동
  - 실시간 웹 검색 및 정보 업데이트

- 정확한 계산 및 처리
  - 전문 계산 도구 활용 (ex. 대규모 행렬 계산, 통계적 분석 및 검증, 수치 시뮬레이션 수행 등)
  - 데이터베이스 쿼리 실행 (ex. 대용량 데이터 집계, 복잡한 조인 연산, 시계열 데이터 분석 등)
  - 외부 알고리즘 활용 (ex. ML 모델 실행, 패턴 매칭 및 검색, 최적화 알고리즘 적용 등)

Tool Use는 다음과 같은 단계로 동작 합니다.
1. 도구 정의 및 프롬프트 전송
  - 개발자가 도구 정의와 사용자 프롬프트를 모델에 전송
  - 도구 정의에는 이름, 설명, 입력 스키마가 포함됨
2. 모델의 도구 사용 결정
  - 모델이 프롬프트를 분석하여 도구 사용 필요성 판단
3. 도구 실행 및 결과 반환
  - 클라이언트가 도구를 실행하고 결과를 모델에 전달
4. 최종 응답 생성
  - 모델이 도구 실행 결과를 바탕으로 최종 응답 생성
  - 사용자의 원래 질문에 대한 완성된 답변 제공

Tool Use와 프롬프트 엔지니어링을 결합했을 때의 시너지 효과를 낼 수 있습니다.
- 프롬프트로 전략적 방향 설정
- Tool Use로 실시간, 정확한 데이터 처리
- 두 기술의 장점을 결합한 고도화된 솔루션 제공


## 코드를 사용하기 위한 요구 사항
- Python 3.7 이상
- AWS 계정 및 자격 증명
- AWS CLI 설치 및 구성


## 추가 리소스
- [Anthropic Tool Use 설명서](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Tool Use with the Amazon Bedrock Converse API 설명서](https://community.aws/content/2hW5367isgQOkkXLYjp4JB3Pe16/intro-to-tool-use-with-the-amazon-bedrock-converse-api)
- [Converse API Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html)
- [ToolConfiguration API Reference](https://docs.aws.amazon.com/ko_kr/bedrock/latest/APIReference/API_runtime_ToolConfiguration.html)