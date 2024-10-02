<h1 align="left"><b>SageMaker 에서 Llama3-8B 파인 튜닝, 모델 배포 및 추론 하기</b></h1>
<p align="center">
    <a href="https://github.com/aws-samples">
            <img alt="Build" src="https://img.shields.io/badge/Contribution-Welcome-blue">
    </a>
    <a href="https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Faws-samples%2Faws-ai-ml-workshop-kr%2Fblob%2Fmaster%2Fgenai%2Faws-gen-ai-kr%2F30_fine_tune%2F03-fine-tune-llama3%2Fllama3%2FREADME.md&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/>
    </a>
</p>


Updated: Oct. 1, 2024
 
---

SageMaker 에서 Llama3-8B 파인 튜닝 및 모델 배포 및 추론을 소개 합니다.<br>
- **FSDP (Fully Sharded Distributed Training)** 및 **QLoRA** 를 사용하여 파인 튜닝을 합니다.
    - 동일 훈련 코드로 SageMaker Training 을 최소 ml.g5.4xlarge 에서 동작 테스트가 되었고, ml.g5.12xlarge, ml.g5.24xlarge, ml.g5.48xlarge, ml.p4d.24xlarge 의 단일 머신 뿐만 아니라 2개의 머신에서도 동작 테스트 되었습니다.
- 파인 튜닝된 모델을 **SageMaker Endpoint에서 추론**을 하기 위해서, Amazon Large Model Inference (LMI) Containers 기반 [서빙 및 추론 실습](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/30_fine_tune/03-fine-tune-llama3/llama3/notebook/20-inference)도 준비되어 있습니다.
<br><br>
이 실습 과정은 제한된 GPU 리소스로 인해서, <u>모델의 품질 보다는 "코드가 동작" 되는 관점에서 준비 했습니다. </u><br>
- 충분한 GPU 리소스가 있으신 환경에서는, 코드 수정 없이 파라미터인 인스턴스 타입, 인스턴스 개수, 데이터 셋 사이즈 수정, Epoch 조정 등의 코드를 수정하여 모델을 최적화 할 수 있습니다. 
- 전체 과정의 실습의 소요 시간은 약 2~3 시간 정도 걸리고, 총 발생 비용은 $10 이하로 발생 됩니다.
 
---
## 1. 파인 튜닝 유스 케이스
- 한국 뉴스 기사 요약 작업을 합니다. 이를 위해 Instruction Fine-Tuning 을 위해 [naver-news-summarization-ko](https://huggingface.co/datasets/daekeun-ml/naver-news-summarization-ko) 데이터 셋을 사용 합니다.
- 이 데이타를 사용하여 다음과 같이 에시로서 파인 튜닝에 적합한 형태(Anthropic Claude Model 형식) 로 변형한 후에 파인 튜닝을 합니다.
```
You are an AI assistant specialized in news articles.Your role is to provide accurate summaries and insights in Korean. Please analyze the given text and provide concise, informative summaries that highlight the key goals and findings.

Human: Please summarize the goals for journalist in this text:

코트라 BMW 오픈이노베이션 IR 로드쇼 추진 스타트업 6개사 선발…현지서 기술 소개하고 상담 추진 기술실증 PoC 프로젝트까지 지원 이데일리 함정선 기자 한국의 스타트업들이 글로벌 기업인 BMW와 에어버스 Airbus 에 혁신 기술을 소개하고 협력 기회를 모색하는 자리가 마련됐다. KOTRA 코트라 는 한국벤처투자 BMW와 스타트업 개러지 코리아 등과 지난 6월 29일부터 7월1일까지 독일 뮌헨에서 ‘BMW 오픈이노베이션 IR 로드쇼’ 사업을 추진했다고 3일 밝혔다. 이번 BMW 오픈이노베이션 IR 로드쇼에서는 지난 4월 서울 컨벤션센터 SETEC 에서 개최된 ‘BMW 오픈이노베이션 피칭데이’를 통해 선발한 스타트업 6개사 15명이 참가했다. 참가 스타트업 6개사는 4박 5일간 독일 뮌헨을 방문해 BMW와 에어버스를 비롯한 독일의 글로벌 제조사에 자신들의 혁신 기술을 소개하고 협업을 제안했다. 6월 29일에는 BMW 그룹의 연구혁신센터에서 참가 스타트업 6개사의 피칭과 전시가 이뤄졌다. 피칭 행사에는 BMW 임직원 총 150명이 한국 스타트업의 피칭을 보기 위해 온·오프라인으로 모였고 하루 동안 총 350명이 전시 부스에 방문하는 등 BMW의 임직원들은 한국의 혁신 기술 스타트업과 협업하는 데에 큰 관심을 보였다. 이는 현장에서 일대일 상담으로 이어졌으며 코트라에 따르면 현재 참가 스타트업과 BMW그룹의 사업 부서 간 기술실증 PoC 프로젝트 추진 가능성이 논의되고 있다. 드라가나 코스틱 Dragana Kostic BMW 테크놀로지 오피스 Technology Office 아시아태평양본부 총괄은 “코로나 이후 한국의 스타트업이 본사를 직접 방문해 기술을 소개하는 것은 이번 행사가 처음”이라며 “참가한 스타트업들이 모두 우수해 앞으로 많은 협업 프로젝트가 이뤄질 것으로 기대한다”고 말했다. 6월 30일에는 뮌헨 외곽지역에 소재한 에어버스 그룹의 연구 혁신센터를 방문해 참가 스타트업 6개사의 피칭 행사를 진행했다. 에어버스 블루스카이 Blue Sky 오픈이노베이션 전담 조직 를 비롯한 그룹 내 핵심 인력을 대상으로 이루어진 이번 행사에서는 피칭 이후 질의응답까지 진행해 현장에서 기술 검증을 위한 여러 가지 논의가 이어졌다. 장 도미닉 코스트 Jean Dominique Coste 에어버스 블루스카이 그룹장은 “에어버스는 한국 스타트업과의 기술협력을 위해 모든 가능성을 열어두고 지속적으로 논의할 것”이라고 말했다. 이외에도 참가 스타트업 중 2개사는 이번 출장 기간 중 100년의 역사를 자랑하는 글로벌 자동차 부품기업인 베바스토 Webasto 자율주행·정밀엔지니어링 기업인 아큐론 Accuron 도 방문해 일대일 상담을 진행하고 협업 가능성을 논의했다. 참가 스타트업 6개사는 △AI 가상인간 D사 △인공신경망처리장치 NPU 기반 AI 반도체 D사 △친환경 가죽 시트 및 가죽실 제조사 A사 △클라우드 기반 VR 및 메타버스 솔루션 I사 △개발자용 AI 비서 L사 △디지털 ID 인증 및 보안솔루션 S사 등이며 한국에 돌아온 이후 BMW 및 에어버스와의 협업 프로젝트를 본격적으로 추진할 예정이다. 유정열 코트라 사장은 “글로벌기업과의 협업은 우리 스타트업이 해외 시장에서 스케일업할 수 있는 매우 중요한 기회”라며 “코트라의 128개 해외무역관이 우리 스타트업의 손과 발이 돼 현지의 글로벌기업과 협업할 기회를 지속적으로 만들어 나갈 것”이라고 밝혔다. BMW FIZ Center 연구혁신센터 에서 참가 스타트업이 BMW 직원을 대상으로 피칭을 진행하고 있다.<|end_of_text|>

Assistant: KOTRA 코트라 는 한국벤처투자 BMW와 스타트업 개러지 코리아 등과 지난 6월 29일부터 7월1일까지 독일 뮌헨에서 ‘BMW 오픈이노베이션 IR 로드쇼’ 사업을 추진했다고 3일 밝혔는데, 이번 BMW 오픈이노베이션 IR 로드쇼에서는 지난 4월 ‘BMW 오픈이노베이션 피칭데이’를 통해 선발한 스타트업 6개사 15명이 참가했으며 장 도미닉 코스트 Jean Dominique Coste 에어버스 블루스카이 그룹장은 “에어버스는 한국 스타트업과의 기술협력을 위해 모든 가능성을 열어두고 지속적으로 논의할 것”이라고 말했다.<|end_of_text|>
```

## 2. 선수 준비 내용
### 2.1 GPU Quota 확인 
현재 사용 계정에 아래의 Resource Quota 가 미리 준비 되어 있어야 합니다. 여기서 확인하시고, Quota 증가 하세요. --> [What is Service Quotas?](https://docs.aws.amazon.com/servicequotas/latest/userguide/intro.html)
- One ml.g5.4xlarge for notebook instance usage
- One ml.g5.4xlarge for training job usage 
- One ml.g5.12xlarge for endpoint usage
### 2.2 HF API Key 및 Llama-3-8B 모델 엑세스 권한
아래 사항을 먼저 준비 하셔야 합니다.
- huggingface Acess Key 준비 하기 : [User access tokens](https://huggingface.co/docs/hub/en/security-tokens)
- Llama-3-8B 모델 엑세스 권한 얻기: [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

## 3. 실습 환경
아래 설치 하기를 클릭하시고, 가이드를 따라가 주세요.
- [setup/README.md](setup/README.md)

## 4. 노트북 실행 순서
### 4.1. 기본 End-to-End (데이터 준비, 파인 튜닝, 모델 배포 및 추론)
아래와 같은 노트북을 차례로 실행 하시면 됩니다. notebook/01-naver-news-fsdp-QLoRA 하위의 노트북 입니다.
- 01-prepare-dataset-kr-news.ipynb  
    - 두가지 종류의 데이터 셋 준비 합니다.
        - Full dataset : 전체 데이터 셋 입니다. (Training: 22,194, Validation: 2466, Test: 27840)
        - Sample dataset for rapid experiment : 전체 데이터 셋 입니다. (Training: 10, Validation: 10, Test: 10)        
- 02-train-fsdp-qlora-local-kr-news.ipynb
    - 로컬 머신에서 훈련을 합니다. 
    - 사용하자 하는 로컬 머신이 GPU 1개이면 혹은 4개, 8 개 이면, 실행시 파라미터인 --nproc_per_node=4 만 수정하시면 됩니다.
- 03-train-fsdp-qlora-sm-kr-news.ipynb
    - SageMaker Local Mode 로  훈련이 가능합니다. 이렇게 테스트 후에 SageMaker Cloud 에서 훈련하시면 됩니다. USE_LOCAL_MODE = True 변수로 조절 하시면 됩니다.
    - run_debug_sample = True 에 따라서 전체 데이터셋으로 훈련을 할지, 일부 샘플 데이터로 빠르게 훈련 코드를 테스트 할 수 있습니다.

### 4.2. 모델 배포 및 추론 심화 ( Bring Your Own Model )
notebook/20-inference 하위의 노트북 입니다.
- sagemaker-infernece.ipynb
    - 파인 튜닝한 모델을 [SageMaker Inference Component](https://aws.amazon.com/blogs/machine-learning/reduce-model-deployment-costs-by-50-on-average-using-sagemakers-latest-features/) 로 직접 배포하고, 추론을 하는 예시 입니다.

## 5. 리소스 정리
- 실습을 완료 후에 반드시 사용하신 리소스를 정리 해야 합니다. 추가적인 요금이 불필요하게 발생할 수 있습니다. SageMaker Endpoint, SageMaker Notebook Instance 가 반드시 삭제가 되었는지 확인 해야 합니다.

## A. Reference
- [Fine-tune Llama 3 with PyTorch FSDP and Q-Lora on Amazon SageMaker](https://www.philschmid.de/sagemaker-train-deploy-llama3)
- [Efficiently fine-tune Llama 3 with PyTorch FSDP and Q-Lora](https://www.philschmid.de/fsdp-qlora-llama3)
- [Launching your 🤗 Accelerate scripts](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch)
- [LLM-based summarization: A case study of human, Llama 2 70b and GPT-4 summarization quality](https://www.anyscale.com/blog/llm-based-summarization-a-case-study-of-human-llama-2-70b-and-gpt-4-summarization-quality)
- [LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4, A Technical Report](https://arxiv.org/pdf/2405.00732)
- [Fine Tune LLM for Text Summary](https://www.kaggle.com/code/mitanshuchakrawarty/fine-tune-llm-for-text-summary)

## <div id="Contributors">**Contributors**</div>
- <span style="#FF69B4;"> **Gonsoo Moon** (AWS Sr. AI/ML Specislist Solutions Architect) | [Mail](mailto:moongons@amazon.com) | [Linkedin](https://www.linkedin.com/in/gonsoomoon/) | [Git](https://github.com/gonsoomoon-ml) | </span>
- <span style="#FF69B4;"> **Dongjin Jang, Ph.D.** (AWS AI/ML Specislist Solutions Architect) | [Mail](mailto:dongjinj@amazon.com) | [Linkedin](https://www.linkedin.com/in/dongjin-jang-kr/) | [Git](https://github.com/dongjin-ml) | [Hugging Face](https://huggingface.co/Dongjin-kr) | </span>
- <span style="#FF69B4;"> **Youngjin Kim** (AWS Sr. Solutions Architect) | [Mail](mailto:youngjik@amazon.com) | [Linkedin](https://www.linkedin.com/in/zerojin/) | [Git](https://github.com/comeddy) | </span>
- - -