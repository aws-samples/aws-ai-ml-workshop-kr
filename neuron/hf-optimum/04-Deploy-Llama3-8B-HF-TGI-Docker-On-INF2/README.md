#  AWS Inferentia 기반 위에 llama-2-13B 이용하여 챗봇 데모

Last Update: Feb 25, 2024

---

# 1. 기본 사전 단계
## Quota 준비 
- 먼저 AWS 계정에 아래에 해당되는 기본적인 [Quota](https://docs.aws.amazon.com/servicequotas/latest/userguide/intro.html) 가 필요 합니다. inf2.xlarge 는 vCPUS 4개, inf2.8xlarge 32 개 필요 합니다. Running On-Demand Inf Instances 가 36 개 이상 있어야 합니다. 여기를 통해서 [inf2 spec](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/inf2-arch.html) 확인 해보세요.
- ![quota.jpg](../../tutorial/inference-Llama-2-13b/img/quota.jpg)

# 2. Inf2 EC2 설치
여기서는 2개의 INF2 가 필요합니다. 컴파일을 위해서 inf2.8xlarge, 모델 서빙을 위해서 inf2.xlarge 필요 합니다.


##  2.1. Launch the Instance (inf2.2xlarge EC2 생성)
- Region: us-east-1 에서 진행 했음.
- AMI, Instance Type 지정. 
    - ![install_ec2_inf2-hf.png](img/inf2_xlarge.png)
- [중요] <u>Storage 는 512 GB 로 수정해주세요.</u>
- Trouble Shooting: Error
    - inf2.xlarge EC2 런칭시에 아래와 같은 에러:
        - Failed to start the instance i-04c1XXXXXXXX The requested configuration is currently not supported. Please check the documentation for supported configurations.
    - 솔루션
        - 위의 이유는 해당 Region 및 Availability Zone (AZ) 에 해당 EC2 가 부족하여 발생하는 에러 입니다. EC2 생성시에 Network 부분의 Subnet 을 바꾸어서 해보세요. 그래도 에러가 발생하면 AWS Account 팀에 문의 바랍니다. 

## 2.2 EC2 Connection
- 편하신 방법으로 EC2 에 SSH 로 연결하시면 됩니다. 저는 로컬에서 VS Code Remote Connection 으로 연결 하였습니다. 
- ![VSCode.png](img/EC2-Serving.png)

## 2.3 Launch the Instance (inf2.8xlarge EC2 생성)
위의 2.1, 2.2 과정을 반복해주세요. 단 2.1 에서 instance 를 inf2.8xlarge 로 선택 해주세요.


# 3. 모델 컴파일 하기
- EC2: inf2.8xlarge 에서 합니다.
## 3.1 환경 준비
- 가상 환경 진입 및 버전 확인
    - 아래와 같이 명령어를 통하여 가상 환경 및 버전 확인 합니다.
        - 현재 설치된 AMI 에는 neuronx-cc : 2.14.227 입니다.
            ```
            source  /opt/aws_neuronx_venv_pytorch_2_1/bin/activate
            dpkg -l | grep neuron
            pip list | grep -E "torch|neuronx"
            ```    
    - ![verify_version.png](img/verify_version.png)
- 필요 프로그램 설치
    - huggingface_hub 를 설치 합니다. 추후 모델 다운로드 및 업로시에 사용 합니다.
        ```
        pip install huggingface_hub
        ```
## 3.2 TGI 도커 이미지 다운로드 및 neuronx-cc 버전 확인 
- [neuronx-tgi](https://github.com/huggingface/optimum-neuron/pkgs/container/neuronx-tgi) 도커 이미지를 다운로드 합니다. 이 도커 이미지는 2024년 5월에 퍼블리싱 되었습니다.
    ```
    docker pull ghcr.io/huggingface/neuronx-tgi:0.0.23
    ```
    - 아래와 같이 Docker 를 pull 하였고, 저장된 이미지를 확인합니다.
        - ![pull_docker](img/pull_docker.png)

- 도커를 실행하여 도커에 진입하여 neuron-cc 버전을 확인 합니다.
    ```
    docker run -it --entrypoint /bin/bash \
    ghcr.io/huggingface/neuronx-tgi:latest 
    ```
    - [중요] neuronx-cc 버전이 2.13.66.0 입니다. 이 버전은 EC2 의 가상 환경에 설치된 neuronx-cc : 2.14.227 와 다릅니다. 우리는 TGI Docker image 에서 최종 모델 배포를 해서 서빙을 할 예정이기에, TGI Docker image 의 neuronx-cc 버전이 2.13.66.0 를 사용할 겁니다. 
    - ![docker_inside_neuronx_cc.png](img/docker_inside_neuronx_cc.png)

## 3.3 TGI Docker 의 optimum-cli 통한 파인 튜닝 모델 컴파일
여기서는 MLP-KTLim/llama-3-Korean-Bllossom-8B](https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B) 한국어 모델을 optimum-cli 를 통해서 컴파일 하겠습니다. 자세한 사항은 [Exporting neuron models using NeuronX TGI](https://huggingface.co/docs/optimum-neuron/guides/export_model) 을 참고하세요.

```
time docker run --entrypoint optimum-cli \
-v $(pwd)/data:/data --privileged \
ghcr.io/huggingface/neuronx-tgi:latest \
export neuron --model MLP-KTLim/llama-3-Korean-Bllossom-8B \
--batch_size 4 --sequence_length 4096 \
--auto_cast_type fp16 --num_cores 2 \
data/llama-3-Korean-Bllossom-8B
```
컴파일 실행 결과 화면 (약 5분 40초 걸림)
- ![compile_model.png](img/compile_model.png)


## 3.4 Neuron Model 을 Hugging Face Hub 로 업로드
- Hugging Face Hub 에 모델을 업로드하기 위해서는 "쓰기용" HF Writable Token 이 필요 합니다., 이후에 아래와 같이 명령어를 사용하여 로긴 하세요.
    - 토큰을 환경 변수에 저장
        ```
        export API_TOKEN=<HF Writable Token 입력>
        ```
    - HF 에 로그인
        ```
        huggingface-cli login --token $API_TOKEN
        ```
- 아래와 같이 Gonsoo/AWS-Neuron-llama-3-Korean-Bllossom-8B 에 업로드
    ```
    huggingface-cli upload  Gonsoo/AWS-Neuron-llama-3-Korean-Bllossom-8B \
    ./data/llama-3-Korean-Bllossom-8B --exclude "checkpoint/**"
    ```
    - 위의 명령어의 실행 화면 입니다. 
    - ![upload_model_hf.png](img/upload_model_hf.png)
- Hugging Face Hub 에 등록된 모델 화면 입니다.
    - ![AWS-Neuron-llama-3-Korean-Bllossom-8B.png](img/AWS-Neuron-llama-3-Korean-Bllossom-8B.png)


# 4. 모델 서빙하기
- EC2: inf2.2xlarge 에서 합니다.
## 4.1 환경 준비
- 가상 환경 진입 및 버전 확인
    - 아래와 같이 명령어를 통하여 가상 환경 및 버전 확인 합니다.
        - 현재 설치된 AMI 에는 neuronx-cc : 2.14.227 입니다.
            ```
            source  /opt/aws_neuronx_venv_pytorch_2_1/bin/activate
            dpkg -l | grep neuron
            pip list | grep -E "torch|neuronx"
            ```    
    - ![verify_version.png](img/verify_version.png)
- 필요 프로그램 설치
    - huggingface_hub 를 설치 합니다. 추후 모델 다운로드 및 업로시에 사용 합니다.
        ```
        pip install huggingface_hub
        ```
## 4.2 TGI 도커 이미지 다운로드 및 neuronx-cc 버전 확인 
- [neuronx-tgi](https://github.com/huggingface/optimum-neuron/pkgs/container/neuronx-tgi) 도커 이미지를 다운로드 합니다. 이 도커 이미지는 2024년 5월에 퍼블리싱 되었습니다.
    ```
    docker pull ghcr.io/huggingface/neuronx-tgi:0.0.23
    ```
    - 아래와 같이 Docker 를 pull 하였고, 저장된 이미지를 확인합니다.
        - ![pull_docker](img/pull_docker.png)

## 4.2. HF 에서 모델 다운로드 하기
아래와 같이 Gonsoo/AWS-Neuron-llama-3-Korean-Bllossom-8B 에서 모델을 다운로드 받습니다. 현재 폴더에서 data 폴더를 생성후에 실행 합니다.
```
huggingface-cli download Gonsoo/AWS-Neuron-llama-3-Korean-Bllossom-8B \
--local-dir ./data/AWS-Neuron-llama-3-Korean-Bllossom-8B
```
![download_model_from_hf.png](img/download_model_from_hf.png)

## 4.3. TGI 도커 실행하기
- 로컬에 Neuron 모델은 /data/AWS-Neuron-llama-3-Korean-Bllossom-8B 에 있습니다. 아래와 같이 docker run 을 통해서 TGI docker container 를 실행합니다.

    ```
    docker run \
    -p 8080:80 \
    -v $(pwd)/data:/data \
    --privileged \
    ghcr.io/huggingface/neuronx-tgi:latest \
    --model-id /data/AWS-Neuron-llama-3-Korean-Bllossom-8B
    ```
- 도커 실행 명령어를 실행한 후의 화면 입니다.
    - ![serve_model_tgi.png](img/serve_model_tgi.png)
- 도커 실행 명령어가 완료되어 TGI docker container 가 대기 중입니다.
    - 컴파일시에 --batch_size 4 --sequence_length 4096 로 했기에, 최대 max batch total tokens 이 16,384 ( 4 * 4096) 으로 세팅 되었습니다. 
    - ![ready_for_inference.png](img/ready_for_inference.png)
# 5.추론 테스트 하기 
## 5.1. Completion API 형태로 curl 실행
- 아래 curl 명령어를 실행 합니다.
    ```
    curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"딥러닝이 뭐야?","parameters":{"max_new_tokens":512}}' \
    -H 'Content-Type: application/json'
    ```
- 아래는 total_time="25.099483509s" validation_time="336.958µs" queue_time="24.691µs" inference_time="25.09912212s" time_per_token="49.021722ms" 의 걸린 시간을 보여 줍니다.    
    - ![inference_completion_api.png](img/inference_completion_api.png)

## 5.2. Message API 형태로 curl 실행
- 아래 명령어를 실행 합니다.
    ```
    curl localhost:8080/v1/chat/completions \
        -X POST \
        -d '{
    "model": "tgi",
    "messages": [
        {
        "role": "system",
        "content": "당신은 인공지능 전문가 입니다."
        },
        {
        "role": "user",
        "content": "딥러닝이 무엇입니까?"
        }
    ],
    "stream": false,
    "max_tokens": 512
    }' \
        -H 'Content-Type: application/json'  
    ```
- 아래는 total_time="24.338049835s" validation_time="394.38µs" queue_time="38.361µs" inference_time="24.337617254s" time_per_token="49.266431ms 시간을 보여주고, 위의 결과와 유사 합니다.
    - ![inference_message_api.png](img/inference_message_api.png)

## 5.3. Inferentia2 의 Neuron Core 사용량 확인
- 아래 내용은 위의 "추론 테스트 하기 (Message API)" 실행 중에, 스크린샷 화면 입니다. 두개의 코어가 거의 다 사용되고 있고, Memory 19.4 GB 사용되고 있습니다. 
    - ![neuron-top.png](img/neuron-top.png)         


# 6. Gradio 를 통한 웹으로 접속해서 테스트 하기
* Run Doc

---
        * UI: Gradio
            * create virtual env
            * create notebook and run
            * EC2 inbound 설정
            * Web brower Test
    * SageMake 배포
        * notebook creation








이 튜토리얼은 inf2.48xlarge 로 Llama-2-13b 모델을 로드하여 추론을 위한 가이드 입니다. 아래의 노트북을 실행 하기 위함 입니다. 
- [Create your own chatbot with llama-2-13B on AWS Inferentia
](https://github.com/huggingface/optimum-neuron/blob/main/notebooks/text-generation/llama2-13b-chatbot.ipynb)

실행을 위해서 위의 해당 노트북을 참조 하시고, 여기서는 중요 가이드 및 일부 코드 실행 결과를 보여 드립니다. 

참조:
- [Create your own chatbot with llama-2-13B on AWS Inferentia](https://huggingface.co/docs/optimum-neuron/tutorials/llama2-13b-chatbot)
<br>
<p> 



## 2.3 Start Jupyter Server and Select Kernel
- (1) Optimum-neuron Git Clone 합니다. 
    ```
    git clone https://github.com/huggingface/optimum-neuron.git
    ```
- (2) 아래와 같이 VS Code 에서 Jupyter 를 설치 합니다.
    - ![install_jupyter.png](img/install_jupyter.png)
- (3) 아래와 같이 jupyter server 실행 합니다.
    ```
    python -m notebook --allow-root --port=8080
    ```
    - ![run_jupyter_server.png](img/run_jupyter_server.png)
- (4) 아래의 화면 오른쪽 하단의 llama-2-13b-chat-neuron 노트북을 오픈합니다. 이후에 노트북 오른쪽 상단에 "Select Kernel" 읈 선택하고, Jupter Server 에서 제공한 경로(예: ```http://127.0.0.1:8080/tree?token=f607af8b9d2619a659bdbb5db0983d9f1e2ce50aeedab910```) 를 복사해서 --> "Existing Jupyter Server --> Enter Url of Jupter Serve --> 여기서 붙이기를 합니다. 이후에 아래와 같은 화면이 나옵니다.
    - ![load_notebook.png](img/load_notebook.png)

<p>

# 3. 노트북 실행
## 3.1. NeuronModelForCausalLM 통한 모델 컴파일 및 로딩
- 아래와 같이 Huggingface format 의 모델을 컴파일 하여 NEFF(Neuron Executable File Format) 파일로 변환 후에 모델 로딩 합니다.이 시점에 24개의 Neuron 을 사용하여 로딩 합니다.
    - ![load_model_NeuronModelForCausalLM.png](img/load_model_NeuronModelForCausalLM.png)
- 아래는 모델 추론을 20개를 해보고 있고, Neuron 24개 모두 사용되고 있고, 각각이 약 80% 정도 사용을 하고 있습니다. 또한 Neuron Accelerator (2개의 neuron 있음) 는 32 GB GPU 메모리에서 약 6.1 GB 사용됨을 보여 주고 있습니다. 
- ![inference_neuron.png](img/inference_neuron.png)

## 3.2. 모델 추론을 통한 챗팅
- 여러가지 질문을 해봅니다.
    - ![Llama-2-inference.png](img/Llama-2-inference.png)    

여기까지 오셨으면 성공 하셨습니다. 축하 드립니다. ^^

---
Contributor: 문곤수 (Gonsoo Moon)