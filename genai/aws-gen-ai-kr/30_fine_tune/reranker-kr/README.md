<h1 align="center">Korean Reranker on AWS</h1>
<p align="center">
    <a href="https://github.com/aws-samples">
            <img alt="Build" src="https://img.shields.io/badge/Contribution-Welcome-blue">
    </a>
    <a href="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
    <a href="https://huggingface.co/Dongjin-kr/ko-reranker">
        <img alt="Build" src="https://img.shields.io/badge/KoReranker-🤗-yellow">
    </a>
    <a href="https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/30_fine_tune/reranker-kr">
        <img alt="Build" src="https://img.shields.io/badge/KoReranker-1.0-red">
    </a>
</p>

### **한국어 Reranker** 개발을 위한 **[PyTorch distributed training on SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training.html)** 기반 파인튜닝 가이드를 제시합니다.
ko-reranker는 [BAAI/bge-reranker-larger](https://huggingface.co/BAAI/bge-reranker-large) 기반으로 한국어 데이터에 대해 fine-tuned된 model 입니다.

- - -

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
- <span style="#FF69B4;"> KoReranker is licensed under the [MIT License](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/LICENSE). </span>
