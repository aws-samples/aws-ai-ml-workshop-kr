<h1 align="left"><b>Retrieval-Augmented Generation (RAG) for Large Language Models on AWS</b></h1>
<p align="center">
    <a href="https://github.com/aws-samples">
            <img alt="Build" src="https://img.shields.io/badge/Contribution-Welcome-blue">
    </a>
    <a href="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
    <a href="https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/20_applications/02_qa_chatbot">
        <img alt="Build" src="https://img.shields.io/badge/AdvancedRAG-1.0-red">
    </a>
    <a href="https://huggingface.co/Dongjin-kr/ko-reranker">
        <img alt="Build" src="https://img.shields.io/badge/KoReranker-🤗-yellow">
    </a>
    <a href="https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/30_fine_tune/reranker-kr">
        <img alt="Build" src="https://img.shields.io/badge/KoReranker-1.0-red">
    </a>
</p>

- - -

## **Overview**

- ##### <span style="#FF69B4;"> Retrieval-Augmented Generation (RAG)는 LLM의 답변 생성에 외부 지식을 활용하는 것을 의미한다. </span>
- ##### <span style="#FF69B4;"> RAG는 특히 knowledge-intensive task에서 답변 정확도를 향상시키고 hallucination을 감소시키는 것으로 알려져 있다. </span>
- ##### <span style="#FF69B4;"> 하지만 semantic vector search 기반의 naive RAG의 경우 여전히 부족한 답변의 정확도가 문제가 되고 있고 이는 real-world production으로의 이동을 막는 장애물이 되고 있다.</span>
- ##### <span style="#FF69B4;"> 최근 들어 RAG의 한계를 극복하거나, 성능향상을 위한 기술적 발전이 계속 되고 있다.</span>
- ##### <span style="#FF69B4;"> 이러한 자료들은 public하게 공개되어 있어 누구나 접근이 가능하나, 쏟아지는 자료 속에서 양질의 컨텐츠를 찾는 수고로움과, 러닝커브(이해 및 구현)가 필요하기에 이를 자신의 워크로드에 빠르게 적용하기 힘든 상황이다. </span>
- ##### <span style="color:blue"> 따라서 이 Repositroy는 **양질의 기술 선별, 기술에 대한 설명 및 aws 기반 sample codes 제공을 통해, 고객의 RAG 기반 workload의 생산성 향상을 목적으로 한다.** </span>
- - -

## **Hands-On List**
- ##### <span style="#FF69B4;"> [Setting up the development environment](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/genai/aws-gen-ai-kr/00_setup/setup.ipynb) - 핸즈온 수행을 위한 환경설정</span>
- ##### <span style="#FF69B4;"> [Setting up the development environment](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/00_setup) - 핸즈온 수행을 위한 환경설정</span>

- - -

## **Usage**
- ##### <span style="color:red"> 반드시 해당 링크를 통해 환경세팅을 완료해 주세요 ==> [Env. setting](https://dongjin-notebook-bira.notebook.us-east-1.sagemaker.aws/lab/tree/aws-ai-ml-workshop-kr/genai/aws-gen-ai-kr/00_setup/setup.ipynb) </span>
- - -

## **Reading and Watching List**
- ##### <span style="#FF69B4;"> [READ] [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997v1)</span>
- ##### <span style="#FF69B4;"> [READ] [Practical Considerations in RAG Application Design](https://pub.towardsai.net/practical-considerations-in-rag-application-design-b5d5f0b2d19b)</span>
- ##### <span style="#FF69B4;"> [READ] [Why Your RAG Is Not Reliable in a Production Environment](https://towardsdatascience.com/why-your-rag-is-not-reliable-in-a-production-environment-9e6a73b3eddb)</span>
- ##### <span style="#FF69B4;"> [READ] [A Guide on 12 Tuning Strategies for Production-Ready RAG Applications](https://towardsdatascience.com/a-guide-on-12-tuning-strategies-for-production-ready-rag-applications-7ca646833439)</span>
- ##### <span style="#FF69B4;"> [READ] [5 Blog Posts To Become a RAG Master](https://levelup.gitconnected.com/5-blog-posts-to-become-a-rag-master-d6004c6150d0)</span>

- ##### <span style="#FF69B4;"> [WATCH] [Advanced RAG Webinar by AWS ](https://kr-resources.awscloud.com/kr-on-demand)</span>
- ##### <span style="#FF69B4;"> [WATCH] [Building Production-Ready RAG Apps](https://www.youtube.com/watch?v=TRjq7t2Ms5I)</span>
- ##### <span style="#FF69B4;"> [WATCH] [Use RAG to improve responses in generative AI applications - re:Invent session](https://www.youtube.com/watch?v=N0tlOXZwrSs) | [post](https://www.linkedin.com/posts/manikhanuja_aws-reinvent-2023-use-rag-to-improve-responses-activity-7137694254964903937-QCua/?utm_source=share&utm_medium=member_desktop) | [git](https://github.com/aws-samples/amazon-bedrock-samples/blob/main/knowledge-bases/1_managed-rag-kb-retrieve-generate-api.ipynb) | </span>
- - -

## **What Should We Know**
- ##### <span style="#FF69B4;"> **Lost in Middle** Phenomenon in RAG </span>
    - [paper] [Lost in the Middle: How Language Models Use Long Contexts](https://www-cs.stanford.edu/~nfliu/papers/lost-in-the-middle.arxiv2023.pdf)
    - [blog] [Lost in the Middle: How Language Models Use Long Contexts](https://medium.datadriveninvestor.com/lost-in-the-middle-how-language-models-use-long-contexts-9dd599d465be)
    - [blog] [Overcome Lost In Middle Phenomenon In RAG Using LongContextRetriver](https://medium.aiplanet.com/overcome-lost-in-middle-phenomenon-in-rag-using-longcontextretriver-2334dc022f0e)
    - [blog] [LLM의 Context Window Size가 크다고 좋은 것일까?](https://moon-walker.medium.com/llm%EC%9D%98-context-window-size%EA%B0%80-%ED%81%AC%EB%8B%A4%EA%B3%A0-%EC%A2%8B%EC%9D%80-%EA%B2%83%EC%9D%BC%EA%B9%8C-57870a3e315e)    
----

## **Building Production-Ready RAG Apps**
#### **1. Table stakers**
- ##### <span style="#FF69B4;"> **Better Parsers and Chunk size**</span>
    - [LLM based context splitter for large documents](https://medium.com/@ayhamboucher/llm-based-context-splitter-for-large-documents-445d3f02b01b)
    - [Accuracy by chunk sizes](https://pub.towardsai.net/practical-considerations-in-rag-application-design-b5d5f0b2d19b)
    - [**llmsherpa**](https://github.com/nlmatics/llmsherpa) - Mastering PDFs: Extracting Sections, Headings, Paragraphs, and Tables with Cutting-Edge Parser (PDF chunking) - | [blog](https://blog.llamaindex.ai/mastering-pdfs-extracting-sections-headings-paragraphs-and-tables-with-cutting-edge-parser-faea18870125) |
    - [**Stanza**](https://stanfordnlp.github.io/stanza/) – A Python NLP Package for Many Human Languages (Sentence based spliter) - | [git](https://github.com/nlmatics/llmsherpa) |

- ##### <span style="#FF69B4;"> **Hybrid Search** (Lexical + Semantic search)</span>
    - [vod] [Advanced RAG 03 - Hybrid Search BM25 & Ensembles](https://www.youtube.com/watch?v=lYxGYXjfrNI&list=PL8motc6AQftn-X1HkaGG9KjmKtWImCKJS&index=11)
    - [sample codes - aws] [Hybrid-Fusion](https://dongjin-notebook-bira.notebook.us-east-1.sagemaker.aws/lab/tree/aws-ai-ml-workshop-kr/genai/aws-gen-ai-kr/20_applications/02_qa_chatbot/02_rag/01_rag_hybrid_search.ipynb)
    - [blog] [Improving Retrieval Performance in RAG Pipelines with Hybrid Search](https://towardsdatascience.com/improving-retrieval-performance-in-rag-pipelines-with-hybrid-search-c75203c2f2f5)
    - [blog] [Amazon OpenSearch Service Hybrid Query를 통한 검색 기능 강화](https://aws.amazon.com/ko/blogs/tech/amazon-opensearch-service-hybrid-query-korean/)
    - Rank-Fusion: [RRF](https://velog.io/@acdongpgm/NLP.-Reciprocal-rank-fusion-RRF-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0) (Reciprocal Rank Fusion)
        - 서로 다른 관련성 지표(relevance indicators)를 가진 여러 개의 결과 집합을 하나의 결과 집합으로 결합하는 방법
        - 튜닝을 필요로 하지 않으며, 서로 다른 관련성 지표들이 상호 관련되지 않아도 고품질을 결과를 얻을 수 있음
        
- ##### <span style="#FF69B4;"> **Metadata Filters**</span>
    - Leverage your document metadata (self-query)
        - [vod] [Advanced RAG 01 - Self Querying Retrieval](https://www.youtube.com/watch?v=f4LeWlt3T8Y&list=PLJKSWzIAY6jCl7kY-Y8jEW6o0FW9Dtr9K&index=73&t=8s)
        - [sample codes] [selfQueryingRetriever_QAChains](https://github.com/insightbuilder/python_de_learners_data/blob/main/code_script_notebooks/projects/exploring_bard/selfQueryingRetriever_QAChains.ipynb?source=post_page-----cf12f3eed1f3--------------------------------)
        - [langchain] [Self-querying](https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/)
        - [blog] [Deep Dive Into Self-Query Retriever In Langchain : Exploring Pros of Building SQ Retriever with PaLM](https://medium.com/@kamaljp/deep-dive-into-self-query-retriever-in-langchain-exploring-pros-of-building-sq-retriever-with-cf12f3eed1f3)
        
- ##### <span style="#FF69B4;"> **Table extraction**</span>
    - [Table Transformer](https://www.linkedin.com/posts/smockbrandon_github-microsofttable-transformer-table-activity-7138940321568096256-Sn0q?utm_source=share&utm_medium=member_desktop)
        - Parsing tables in PDFs is a super important RAG use case.
        - The Table Transformer model extracts tables from PDFs using object detection 📊
    - [blog] [Extract custom table from PDF with LLMs](https://medium.com/@knowledgrator/extract-custom-table-from-pdf-with-llms-2ad678c26200)
    - [blog] [RAG Pipeline Pitfalls: The Untold Challenges of Embedding Table](https://medium.com/towards-artificial-intelligence/rag-pipeline-pitfalls-the-untold-challenges-of-embedding-table-5296b2d8230a)
    - [blog] [Working with Table Data in Documents: Tips and Tricks for LLM](https://medium.com/@easonlai888/working-with-table-data-in-documents-tips-and-tricks-for-llm-50f09d2c4e95)
     - [blog] [Revolutionizing RAG with Enhanced PDF Structure Recognition](https://medium.com/@chatdocai/revolutionizing-rag-with-enhanced-pdf-structure-recognition-22227af87442)

#### **2. Advanced Retrieval**

        
        
        











## 0. Features
- #### <span style="#FF69B4;"> Reranker는 임베딩 모델과 달리 질문과 문서를 입력으로 사용하며 임베딩 대신 유사도를 직접 출력합니다.</span>
- #### <span style="#FF69B4;"> Reranker에 질문과 구절을 입력하면 연관성 점수를 얻을 수 있습니다.</span>
- #### <span style="#FF69B4;"> Reranker는 CrossEntropy loss를 기반으로 최적화되므로 관련성 점수가 특정 범위에 국한되지 않습니다.</span>

## 1. Usage

- using Transformers
```
    def exp_normalize(x):
      b = x.max()
      y = np.exp(x - b)
      return y / y.sum()
    
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    pairs = [["나는 너를 싫어해", "나는 너를 사랑해"], \
             ["나는 너를 좋아해", "너에 대한 나의 감정은 사랑 일 수도 있어"]]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        scores = exp_normalize(scores.numpy())
        print (f'first: {scores[0]}, second: {scores[1]}')
```

- using SageMaker
```
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel
try:
	role = sagemaker.get_execution_role()
except ValueError:
	iam = boto3.client('iam')
	role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']
# Hub Model configuration. https://huggingface.co/models
hub = {
	'HF_MODEL_ID':'Dongjin-kr/ko-reranker',
	'HF_TASK':'text-classification'
}
# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
	transformers_version='4.28.1',
	pytorch_version='2.0.0',
	py_version='py310',
	env=hub,
	role=role, 
)
# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
	initial_instance_count=1, # number of instances
	instance_type='ml.g5.large' # ec2 instance type
)
runtime_client = boto3.Session().client('sagemaker-runtime')
payload = json.dumps(
    {
        "inputs": [
            {"text": "나는 너를 싫어해", "text_pair": "나는 너를 사랑해"},
            {"text": "나는 너를 좋아해", "text_pair": "너에 대한 나의 감정은 사랑 일 수도 있어"}
        ]
    }
)
response = runtime_client.invoke_endpoint(
    EndpointName="<endpoint-name>",
    ContentType="application/json",
    Accept="application/json",
    Body=payload
)
## deserialization
out = json.loads(response['Body'].read().decode()) ## for json
print (f'Response: {out}')
```
- - -

## 2. Backgound
- #### <span style="#FF69B4;"> **컨택스트 순서가 정확도에 영향 준다**([Lost in Middel, *Liu et al., 2023*](https://arxiv.org/pdf/2307.03172.pdf)) </span>

- #### <span style="#FF69B4;"> [Reranker 사용해야 하는 이유](https://www.pinecone.io/learn/series/rag/rerankers/)</span>
    - 현재 LLM은 context 많이 넣는다고 좋은거 아님, relevant한게 상위에 있어야 정답을 잘 말해준다
    - Semantic search에서 사용하는 similarity(relevant) score가 정교하지 않다. (즉, 상위 랭커면 하위 랭커보다 항상 더 질문에 유사한 정보가 맞아?) 
        * Embedding은 meaning behind document를 가지는 것에 특화되어 있다. 
        * 질문과 정답이 의미상 같은건 아니다. ([Hypothetical Document Embeddings](https://medium.com/prompt-engineering/hyde-revolutionising-search-with-hypothetical-document-embeddings-3474df795af8))
        * ANNs([Approximate Nearest Neighbors](https://towardsdatascience.com/comprehensive-guide-to-approximate-nearest-neighbors-algorithms-8b94f057d6b6)) 사용에 따른 패널티

- - -

## 3. Reranker models

- #### <span style="#FF69B4;"> [Cohere] [Reranker](https://txt.cohere.com/rerank/)</span>
- #### <span style="#FF69B4;"> [BAAI] [bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)</span>
- #### <span style="#FF69B4;"> [BAAI] [bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)</span>

- - -

## 4. Dataset

- #### <span style="#FF69B4;"> [msmarco-triplets](https://github.com/microsoft/MSMARCO-Passage-Ranking) </span>
    - (Question, Answer, Negative)-Triplets from MS MARCO Passages dataset, 499,184 samples
    - 해당 데이터 셋은 영문으로 구성되어 있습니다.
    - Amazon Translate 기반으로 번역하여 활용하였습니다.
    
#### <span style="#FF69B4;"> Format </span>
```
{"query": str, "pos": List[str], "neg": List[str]}
```
- Query는 질문이고, pos는 긍정 텍스트 목록, neg는 부정 텍스트 목록입니다. 쿼리에 대한 부정 텍스트가 없는 경우 전체 말뭉치에서 일부를 무작위로 추출하여 부정 텍스트로 사용할 수 있습니다.

#### <span style="#FF69B4;"> Example </span>
```
{"query": "대한민국의 수도는?", "pos": ["미국의 수도는 워싱턴이고, 일본은 도교이며 한국은 서울이다."], "neg": ["미국의 수도는 워싱턴이고, 일본은 도교이며 북한은 평양이다."]}
```
    
- - -

## 5. Performance
| Model                     | has-right-in-contexts | mrr (mean reciprocal rank) |
|:---------------------------|:-----------------:|:--------------------------:|
| without-reranker (default)| 0.93 | 0.80 |
| with-reranker (bge-reranker-large)| 0.95 | 0.84 |
| **with-reranker (fine-tuned using korean)** | **0.96** | **0.87** |

- **evaluation set**:
```code
./dataset/evaluation/eval_dataset.csv
```
- **training parameters**: 

```json
{
    "learning_rate": 5e-6,
    "fp16": True,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 32,
    "train_group_size": 3,
    "max_len": 512,
    "weight_decay": 0.01,
}
```

- - -

## 6. Acknowledgement
- <span style="#FF69B4;"> Part of the code is developed based on [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/tree/master?tab=readme-ov-file) and [KoSimCSE-SageMaker](https://github.com/daekeun-ml/KoSimCSE-SageMaker/tree/7de6eefef8f1a646c664d0888319d17480a3ebe5).</span>

- - -

## 7. Citation
- <span style="#FF69B4;"> If you find this repository useful, please consider giving a star ⭐ and citation</span>

- - -

## 8. Contributors:
- <span style="#FF69B4;"> **Dongjin Jang, Ph.D.** (AWS AI/ML Specislist Solutions Architect) | [Mail](mailto:dongjinj@amazon.com) | [Linkedin](https://www.linkedin.com/in/dongjin-jang-kr/) | [Git](https://github.com/dongjin-ml) | </span>

- - -

## 9. License
- <span style="#FF69B4;"> FlagEmbedding is licensed under the [MIT License](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/LICENSE). </span>
