# 워크샵: RAG Evaluation Using RAGAS and Amazon Bedrock 

이 워크샵은 검색 증강 생성 (RAG, Retrieval Augmented Generation) 시스템을 만든 후에, 얼마나 질문에 답변을 정확히 하는지를 평가하는 것을 배우게 됩니다. <br>
워크샵을 통해 배울 수 있는 것은 아래와 같습니다. 
- RAG Evaluation 방법론
- [RAGAS](https://docs.ragas.io/en/stable/) RAG Evaluation Metrics
- RAG 평가를 위한 합성 데이터셋 생성 
- Test RAG 시스템 생성: [Amazon Bedrock Knowledge Base](https://aws.amazon.com/ko/bedrock/knowledge-bases/?nc1=h_ls) 사용
- Amazon Bedrock 를 사용한 RAGAS Metrics 사용 방법
- Custom RAG Metrics 를 정의하여 평가 방법

# 1. 문제 정의
주로 RAG 시스템을 만들고 나서, 질문/답변의 쌍으로 구성된 Test 셋을 구성하여 RAG 시스템에 실행하고 어느 정도 답변이 정확하게 나왔는지를 비체계적으로 수행 합니다. <br>
"RAG 시스넴을 체계적으로 평가 및 검증 (Evaluation)할 수 있을까?" 라는 질문을 많이 합니다. <br>이 워크샵은 아래의 솔루션의 방향으로 이 문제를 해결하고자 합니다.

# 2. 솔루션
## 2.1. RAG Evaluation 방법론
 RAG 시스템을 만들면서, 비체계적으로 평가 및 검증을 합니다.이를 어떻게 체계적으로 할까 고민을 했고, 업계에서 가장 많이 사용하는 것 중의 하나인 오픈 소스 평가 프레임워크인 [RAGAS](https://docs.ragas.io/en/stable/) 를 사용을 합니다. <br>
 또한 평가를 위한 테스트용의 RAG 시스템은 [Amazon Bedrock Knowledge Base](https://aws.amazon.com/ko/bedrock/knowledge-bases/?nc1=h_ls) 을 선택 했습니다. 이유는 UI 로 쉽게 클릭, 클릭 해서 빠르게 만드는 것이 가능해서 였습니다. 만드는 방법은 이 블로그 [Amazon Bedrock Knowledge base로 30분 만에 멀티모달 RAG 챗봇 구축하기 실전 가이드](https://aws.amazon.com/ko/blogs/tech/practical-guide-for-bedrock-kb-multimodal-chatbot/) 를 참조 히시면 됩니다. <br>
 평가할 원본 문서 (데이터) 도 많은 사람이 직관적으로 이해할 수 있고, 흥미가 있는 데이터가 필요 했고, Claude Sonnet3.5 모델에게 "전 세계의 음식 레서피를 만들어줘" 라고 프폼프트를 주고 답변을 받은 것을 pdf 문서로 변환하여 사용을 하였습니다. <br> 
 기본 코드 베이스는 Kihyeon Myung 님의 RAG Evaluation Using Bedrock[4] 를 참조 했습니다. 

## 2.2. RAGAS RAG Evaluation Metrics
RAG 애플리케이션에서 RAGAS는 RAG 시스템의 검색(retrieval) 및 생성(generation) 구성 요소를 모두 평가할 수 있습니다 [1].<br>

![ragas metrics](img/ragas_metrics.jpg)

그림 처럼 ragas score 는 크게 Generation, Retrieval 의 두개의 분류로 나눌 수 있고, 각 메트릭의 정의를 확인 하겠습니다 [2].<br>
그리고 아래 설명에 대한 실제 코드 예시 [01-ragas-bedrock-metrics.ipynb](notebook/00-warming-up/01-ragas-bedrock-metrics.ipynb) 도 참조하시면 좋습니다.
### 2.2.1. Generation: faithfulness
**Faithfulness**(충실도) 지표는 `LLM 응답`이 `검색된 컨텍스트`와 얼마나 사실적으로 일치하는지를 측정합니다. 0에서 1 사이의 값을 가지며, 높은 점수일수록 더 나은 일관성을 나타냅니다.

응답이 **충실하다(faithful)**고 판단되는 것은 응답의 모든 주장이 검색된 컨텍스트로부터 뒷받침될 수 있을 때입니다.

계산 방법:
1. 응답에 있는 모든 주장들을 식별합니다
2. 각 주장이 검색된 컨텍스트로부터 추론될 수 있는지 확인합니다
3. 공식을 사용하여 충실도 점수를 계산합니다

더 쉽게 설명하면:
- 이는 "LLM이 제공한 응답이 Retrieved Context의 내용과 얼마나 일치하는가"를 측정합니다
- 예를 들어:
  - Retrieved Context에 "김치는 배추, 고춧가루, 마늘이 들어갑니다"라고 되어있는데
  - LLM 이 "김치의 주재료는 배추, 고춧가루, 마늘입니다"라고 응답하면 높은 충실도 점수
  - 반면 "김치에는 당근이 들어갑니다"라고 응답하면 낮은 충실도 점수를 받게 됩니다

### 2.2.2. Generation: answer relevancy
`answer relevancy` 지표는 응답이 **사용자 입력 (예: 프롬프트의 Question)** 과 얼마나 관련이 있는지를 측정합니다. 높은 점수는 사용자 입력과의 더 나은 일치를 나타내며, 응답이 불완전하거나 불필요한 정보를 포함할 경우 낮은 점수가 주어집니다.

이 지표는 `user_input (사용자 입력)`과 `response(응답)`를 사용하여 다음과 같이 계산됩니다:
1. 응답을 기반으로 인공적인 질문들(기본값 3개)을 생성합니다. 이 질문들은 응답의 내용을 반영하도록 설계됩니다.
2. 사용자 입력의 임베딩(Eo)과 각 생성된 질문의 임베딩(Egi) 사이의 코사인 유사도를 계산합니다.
3. 이러한 코사인 유사도 점수들의 평균을 계산하여 **응답 관련성(Answer Relevancy)**을 구합니다.

더 쉽게 설명하면:
- 이 지표는 "시스템의 응답이 사용자의 질문이나 요청에 얼마나 잘 부합하는가"를 측정합니다
- 예를 들어, 사용자가 "김치 만드는 방법"을 물었는데:
  - 김치 만드는 과정을 자세히 설명하면 높은 점수
  - 다른 한식 요리법을 설명하거나 불필요한 정보를 포함하면 낮은 점수를 받게 됩니다

### 2.2.3. Retrieval: context precision
- Context Precision은 retrieved_contexts에서 관련된 청크들의 비율을 측정하는 지표입
- 문서나 텍스트를 검색할 때, 시스템은 여러 개의 작은 조각(청크)들을 가져옵니다. 
- 이때 우리가 원하는 정보와 관련 있는 조각들을 얼마나 정확하게 가져왔는지를 측정하는 것이 Context Precision입니다.
    - 예를 들어 설명하면: 당신이 "한국의 전통 음식"에 대해 검색했다고 가정해봅시다. 
    - 시스템이 10개의 텍스트 조각을 가져왔는데, 그 중 7개는 실제로 한국 음식에 대한 내용이고,  
    - 3개는 다른 나라의 음식이나 관련 없는 내용이라면 이 경우의 Context Precision은 7/10 = 0.7 또는 70%가 됩니다

### 2.2.4. Retrieval: context recall
Context Recall은 관련된 문서(또는 정보)들이 얼마나 성공적으로 검색되었는지를 측정합니다. 이는 중요한 결과들을 놓치지 않는 것에 초점을 맞춥니다. 높은 recall은 관련된 문서들이 더 적게 누락되었다는 것을 의미합니다. 간단히 말해서, recall은 중요한 정보를 놓치지 않는 것에 관한 것입니다. 이는 누락된 것이 없는지를 측정하는 것이기 때문에, context recall을 계산하기 위해서는 항상 비교할 수 있는 기준이 필요합니다.

더 쉽게 설명하면:
- Precision이 "가져온 정보가 얼마나 정확한가"를 측정한다면
- Recall은 "필요한 정보를 얼마나 빠짐없이 가져왔는가"를 측정합니다

예를 들어:
- 도서관에 한국 요리에 대한 책이 총 100권이 있다고 가정했을 때
- 검색 시스템이 80권을 찾아냈다면
- Recall은 80/100 = 0.8 또는 80%가 됩니다
- 즉, 필요한 정보의 80%를 찾아냈다는 의미입니다


## 2.3 합성 데이터셋 생성 
RAG 평가 합성 데이터 셋을 생성하기 위해서 원본 문서 [241231-Sonnet-recipe.pdf](data/241231-Sonnet-recipe.pdf) 를 사용합니다. <br>
이 문서는 Claude Sonnet3.5 를 사용하여 만든 문서 입니다.
3 페이지로 되어 있고, 예시로써 아래와 같은 음식의 레서피가 있습니다.
```
볶음요리로는 먼저 김치볶음밥이 있는데, 잘 익은 김치를 잘게 썰어 밥과 함께
볶다가 마지막에 참기름을 둘러 고소한 향을 더합니다. 제육볶음은 돼지고기를 고
추장 양념에 버무려 매콤달콤하게 볶아내며, 양파와 당근 등 채소를 함께 넣어
영양을 높입니다. 낙지볶음은 신선한 낙지를 손질해 고추장 양념에 볶아 매콤하게
만드는 요리입니다.
양식으로는 크림파스타가 대표적인데, 생크림과 마늘, 양파를 볶아 소스를 만들고
베이컨이나 새우를 더해 고소하게 완성합니다. 토마토파스타는 토마토소스에 마늘
과 양파를 볶아 넣고 바질을 곁들여 상큼한 맛을 냅니다. 미트소스스파게티는 다
진 쇠고기를 토마토소스와 함께 오래 끓여 깊은 맛을 만듭니다.
```
위의 문서를 작은 조각 (Chunking) 으로 만든 후에, 아래와 같은 question, ground_truth, question_type, contexts 로 구성된 "RAG 평가 합성 데이터 셋" 생성 합니다. 아래는 생성된 1개의 예시를 보여 주고 있습니다.
```
{'question': '라자냐는 어떤 재료로 만들어지나요?', 
'ground_truth': '라자냐는 파스타면, 미트소스, 베샤멜 소스, 치즈를 층층이 쌓아 오븐에 구워 만듭니다.', 
'question_type': 'simple', 
'contexts': '라자냐는 파스타\n면과 미트소스, 베샤멜 소스, 치즈를 층층이 쌓아 오븐에 구워내는 풍성한 요리\n입니다. 피자는 밀가루 반죽을 얇게 펴서 토마토소스를 바르고 모차렐라 치즈와\n각종 토핑을 올려 화덕에서 구워냅니다.\n멕시코 요리에서는 타코가 대표적인데, 또르티야에 구운 고기와 양파, 고수, 라\n임을 넣어 먹습니다. 엔칠라다는 또르티야에 고기와 치즈를 넣어 말아서 특제 소\n스를 부어 오븐에 구워냅니다.}
```
## 2.4 Test RAG 생성: Amazon Bedrock Knowledge Base
또한 평가를 위한 테스트용의 RAG 시스템은 [Amazon Bedrock Knowledge Base](https://aws.amazon.com/ko/bedrock/knowledge-bases/?nc1=h_ls) 을 선택 했습니다.만드는 방법은 이 블로그 [Amazon Bedrock Knowledge base로 30분 만에 멀티모달 RAG 챗봇 구축하기 실전 가이드](https://aws.amazon.com/ko/blogs/tech/practical-guide-for-bedrock-kb-multimodal-chatbot/) 를 참조 히시면 됩니다. <br>
아래는 이 워크샵의 코드 실행을 위해서, 생성한 Amazon Bedrock Knowledge Base 의 예시 화면 입니다. 여기서 Knowledge Base ID 를 API retrieve 에 인자로 제공해서 검색 결과를 가져 옵니다.
[amazon_kb](img/amazon_kb.png)
 
## 2.5 Amazon Bedrock 를 사용한 RAGAS Metrics 사용
RAGAS Metrics를 Amazon Bedrock 으로 사용하기 위해서, ragas.llms.LangchainLLMWrapper 을 사용을 해야 합니다 [3]. <br>
이에 대한 예시 코드 [00-ragas-bedrock-getting-started.ipynb)](notebook/00-warming-up/00-ragas-bedrock-getting-started.ipynb) 를 참조 하시면 됩니다.

## 2.6 Custom RAG Metrics 
RAGAS Metrics 을 사용을 해도 되지만, 사용자가 직접 정의 해서 사용해도 됩니다. 코드 에시는[4] 를 참조 했습니다. 

# 3. 결과
## 3.1. 5개의 샘플 데이터 평가 결과
아래는 RAGAS Metrics 을 Amazon Bedrock 의 Sonnet3.5 모델, Titan Text 임베딩 모델을 통해서 평가한 내용 입니다.
![ragas_result.png](img/ragas_result.png)

## 3.2. 1개의 샘플 데이터 평가 결과 
하나의 결과만을 자세히 보겠습니다.
```
## AnswerRelevancy:  0.44
## Faithfulness:  0.41
## ContextRecall:  1.0
## ContextPrecision:  0.49

## Question: 
 지중해 연안 국가들의 요리 중 채식주의자와 육식주의자 모두를 만족시킬 수 있는 메뉴 구성을 제안한다면, 어떤 요리들을 선택하고 그 이유는 무엇인가요?

## Ground_truth: 
 지중해 연안 국가들의 요리 중 채식주의자와 육식주의자 모두를 만족시킬 수 있는 메뉴 구성은 다음과 같습니다:

1. 프랑스의 라타투이: 가지, 토마토, 호박, 파프리카 등 다양한 채소로 만든 채식 요리로, 채식주의자에게 적합합니다.
2. 스페인의 가스파초: 토마토, 오이, 피망으로 만든 차가운 스프로, 채식주의자도 즐길 수 있는 상큼한 요리입니다.
3. 그리스의 무사카: 가지와 다진 고기를 층층이 쌓아 만드는 요리로, 육식주의자를 위한 선택입니다. 채식 버전으로 변형도 가능합니다.
4. 모로코의 쿠스쿠스: 밀가루로 만든 알갱이에 채소를 얹어 먹는 요리로, 고기를 별도로 제공하여 선택적으로 먹을 수 있게 합니다.
5. 스페인의 빠에야: 해산물과 채소를 넣은 쌀 요리로, 해산물을 먹는 사람과 채식주의자 모두에게 적합합니다.

이러한 구성은 다양한 식단 선호도를 고려하면서도 지중해 요리의 특색을 잘 보여줍니다. 채식 옵션과 육류 옵션을 동시에 제공하여 모든 식성의 사람들을 만족시킬 수 있으며, 각 요리의 재료와 조리법을 조절하여 개인의 선호에 맞출 수 있는 유연성도 갖추고 있습니다.

## Generated Answer: 
 주어진 맥락을 바탕으로 지중해 연안 국가들의 요리 중 채식주의자와 육식주의자 모두를 만족시킬 수 있는 메뉴 구성을 제안하겠습니다:

1. 스페인의 가스파초: 토마토, 오이, 피망으로 만든 차가운 스프로, 채식주의자에게 적합합니다.
2. 그리스의 돌마데스: 포도잎에 밥과 다진 고기를 싸서 만드는 요리로, 고기 대신 채소로 채워 채식 버전으로 만들 수 있습니다.
3. 프랑스의 라타투이: 가지, 토마토, 호박, 파프리카 등 다양한 채소로 만든 프로방스 지방의 대표적인 채식 요리입니다.
4. 스페인의 빠에야: 해산물과 채소를 넣은 쌀 요리로, 해산물을 먹는 페스코 베지테리언과 육식주의자 모두에게 적합합니다.
5. 그리스의 무사카: 가지와 다진 고기를 층층이 쌓아 만드는 요리로, 고기 대신 렌틸콩이나 버섯으로 대체하여 채식 버전을 만들 수 있습니다.
6. 모로코의 쿠스쿠스: 채소와 고기를 함께 먹는 요리지만, 고기를 제외하고 채소만으로도 맛있게 즐길 수 있습니다.

이러한 메뉴 구성은 다음과 같은 이유로 채식주의자와 육식주의자 모두를 만족시킬 수 있습니다:

1. 다양성: 여러 나라의 요리를 포함하여 다양한 맛과 텍스처를 제공합니다.
2. 유연성: 일부 요리는 채식 버전으로 쉽게 변형할 수 있어 선택의 폭이 넓습니다.
3. 영양 균형: 채소, 곡물, 단백질 등 다양한 영양소를 포함하고 있습니다.
4. 문화적 경험: 지중해 연안의 다양한 식문화를 경험할 수 있습니다.

이 메뉴 구성은 채식주의자와 육식주의자 모두에게 맛있고 만족스러운 식사 경험을 제공할 수 있을 것입니다.
```
## 3.3. 1개의 샘플 데이터 평가 결과 분석 (Claude Sonnet 3.5 제공)
위의 데이타와 지표에 대한 해석을 위해서 Claude Sonnet 3.5 에 물어 보았습니다. "다음은 RAGAS 로 측정한 데이타와 지표야. 여기서 왜 ContextRecall 이 1.0 이야?"
```
Context Recall이 1.0인 이유를 설명해드리겠습니다.

Context Recall은 ground truth에서 언급된 중요한 정보들이 생성된 답변에 얼마나 포함되어 있는지를 측정하는 지표입니다. 1.0이라는 점수는 ground truth의 핵심 정보들이 100% 생성된 답변에 포함되어 있다는 의미입니다.

이 사례에서 구체적으로 살펴보면:

1. Ground truth에서 언급된 주요 요리들이 대부분 생성된 답변에도 포함되어 있습니다:
   - 프랑스의 라타투이
   - 스페인의 가스파초
   - 그리스의 무사카
   - 모로코의 쿠스쿠스
   - 스페인의 빠에야

2. 각 요리의 주요 특징과 채식/육식 가능 여부에 대한 설명도 포함되어 있습니다.

3. 메뉴 구성의 장점(다양한 식단 선호도 고려, 유연성 등)도 답변에 반영되어 있습니다.

그러나 주목할 점은 Context Recall이 1.0임에도 불구하고 다른 지표들(AnswerRelevancy: 0.44, Faithfulness: 0.42, ContextPrecision: 0.5)은 상대적으로 낮습니다. 이는 생성된 답변이 ground truth의 정보는 모두 포함하고 있지만, 추가적인 정보를 더 포함하거나 다른 방식으로 표현하여 정확성과 관련성이 떨어졌을 수 있다는 것을 의미합니다.
```

# 4.4.워크샵 실행 방법
- 워크샵의 실행 방법은 여기 [워크샵 실행 방법](setup/README.md) 을 참조 해주세요.
<p>

# A. 참조 사항
[1] Okan Yenigün, RAGAS: A Framework for Accurate and Faithful RAG Model Evaluation, https://medium.com/ai-mind-labs/ragas-a-framework-for-accurate-and-faithful-rag-model-evaluation-5380b73d45b5 <br>
[2] ragas, List of available metrics, https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
[3] ragas, Testset Generation for RAG, https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/
[4] Kihyeon Myung, RAG Evaluation Using Bedrock, https://github.com/kevmyung/rag-evaluation-bedrock








This repository demonstrates how to evaluate the quality of a RAG (Retrieval-Augmented Generation) pipeline using the Bedrock API.

## 01-test-dataset-generation.ipynb

To evaluate RAG quality, we need to create an evaluation dataset. While it's possible to create datasets manually, it's time-consuming. Therefore, using LLMs to generate synthetic datasets is a popular approach. Here's how we generate the desired number of synthetic Q&A pairs:

1. Load a PDF document
2. Split the document into chunks of a specified length
3. Randomly select a chunk and generate a complex/simple question and its corresponding answer (ground truth) based on that chunk
4. Repeat the chunk selection and Q&A generation process N times

The completed test dataset is structured as follows:
```
{
    "quesiton": "{generated question}",
    "ground_truth": "{generated answer}",
    "question_type": "complex | simple",
    "contexts": "{randomly selected context used for Q&A generation}
}
```

## 02-ragas-evaluation.ipynb

This notebook uses the test dataset to evaluate RAG quality. It supports evaluation of RAGAS metrics including AnswerRelevancy, Faithfulness, ContextRecall, and ContextPrecision. 

While relying on existing RAGAS evaluation metrics and calculation methods, the following modifications have been made:

- Uses Bedrock's converse API instead of LangChain or LlamaIndex frameworks
- Employs tool use prompting techniques to ensure accurate JSON parsing of LLM outputs
- Changes asynchronous LLM calls to synchronous to avoid API throttling when dealing with large Q&A test sets
- Modifies segment separation for Faithfulness and Recall calculations from sentence-based (`.endswith(".")`) to paragraph-based (`'\n\n'`) for better context continuity
- Simplifies the Faithfulness calculation by removing the statement simplification and reason generation steps to reduce token usage and processing time
- The `libs/custom_ragas.py` file contains the custom implementation of the RAGAS evaluation metrics. It includes the modifications mentioned above.

Here's a brief overview of what each metric measures:

### AnswerRelevancy
Evaluates how well the generated answer fits the given prompt. It generates virtual questions based on the context and answer, calculates vector similarity between generated and user questions, and averages the scores. Higher scores indicate better performance.

### Faithfulness
Measures the factual consistency of the generated answer compared to the given context. It segments the answer and classifies each segment as either model-generated (0) or context-based (1). The final score is the average of these classifications. Higher scores indicate better performance.

### ContextRecall
Assesses how well the retrieved context matches the LLM-generated answer. It segments the ground truth and determines if each segment can be attributed to the retrieved context. The score is the average of these attributions. Higher scores indicate better performance.

### ContextPrecision
Evaluates how relevant documents are ranked in the retrieved context. It uses an LLM to judge the usefulness of each context in the list for answer generation, with higher weights given to useful contexts appearing earlier in the list.

These modifications aim to improve evaluation accuracy, reduce processing time, and optimize resource usage while maintaining the core principles of the RAGAS evaluation framework.

## 03-llm-as-a-judge.ipynb

This notebook demonstrates the concept of using an LLM as a judge for evaluating RAG quality. 
It imports and utilizes the custom implementation from `libs/custom_llm_as_a_judge.py`.

### LLM-as-a-Judge Approach

The LLM-as-a-Judge method allows for automated evaluation of RAG outputs based on predefined criteria. This approach can be particularly useful for:

1. Rapid iteration in LLM application development
2. Reducing the need for extensive human evaluation
3. Providing consistent evaluation across large datasets

### Evaluation Criteria

The notebook showcases how to evaluate the quality of Q&A based on criteria such as:

- Conciseness
- Relevance
- Correctness
- Coherence
- Helpfulness
...


## 📚 Reference

- [RAGAS] (https://github.com/explodinggradients/ragas)
- [langchain-ai](https://github.com/langchain-ai/langchain) 
- [langchain-kr](https://github.com/teddylee777/langchain-kr) 
