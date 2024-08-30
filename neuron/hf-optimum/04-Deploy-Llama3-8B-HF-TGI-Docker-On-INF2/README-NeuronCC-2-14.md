#  Amazon ECR 의 도커 이미지 기반하에 Amazon EC2 Inferentia2 서빙하기
Last Update: Aug 30, 2024

---

이 문서는 AWS Inferentia2 EC2 기반에서 한국어 파인 튜닝 모델을 서빙하는 과정을 설명합니다. 
아래의 진행 전에 [Amazon EC2 Inferentia2 기반 위에 한국어 파인 튜닝 모델을 서빙하기](README.md) 를 먼저 확인하시고 보기를 권장 합니다.
---


# 1. INF2.8xlarge EC2 에 권한 부여 하기
- 다음과 같이 "역할" 이름을 찾습니다.
    * EC2 콘솔에서 역할을 찾는 방법은 다음과 같습니다:
    * AWS Management Console에 로그인합니다.
    * EC2 서비스로 이동합니다.
    * 왼쪽 탐색 창에서 "인스턴스"를 선택합니다.
    * 역할을 확인하거나 변경하려는 EC2 인스턴스를 선택합니다.
    * 인스턴스 세부 정보 페이지에서 "보안" 탭을 클릭합니다.
    * "IAM 역할" 섹션에서 현재 인스턴스에 연결된 IAM 역할을 확인할 수 있습니다
- 아래와 같이 ECR 권한을 추가 합니다.
    - ![ec2_role.png](img/ec2_role.png)
    

# 2. Neuron-CC 2.14 로 컴파일
- 준비된 INF2.8xlarge 머신에서 터미널을 오픈 합니다.
- ECR 에 로그인하고, 해당 이미지를 다운로드 합니다.
```
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.2-optimum0.0.24-neuronx-py310-ubuntu22.04-v1.0
```
- 컴파일을 합니다.
```
time docker run --entrypoint optimum-cli \
-v $(pwd)/data:/data --privileged \
763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.2-optimum0.0.24-neuronx-py310-ubuntu22.04-v1.0 \
export neuron --model MLP-KTLim/llama-3-Korean-Bllossom-8B \
--batch_size 4 --sequence_length 4096 \
--auto_cast_type fp16 --num_cores 2 \
data/llama-3-Korean-Bllossom-8B-NeuronCC-2.14
```
- 위의 명령어에 대한 스냅샵 입니다.
    - ![pull_docker_image.jpg](img/pull_docker_image.jpg)
- 위의 작업을 진행하기 전에 Neuro-cc version 도 아래와 같이 확인을 권장 합니다.
```
docker run -it --entrypoint /bin/bash 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.2-optimum0.0.24-neuronx-py310-ubuntu22.04-v1.0

neuronx-cc --version
```
- ![neuron-cc-version.png](img/neuron-cc-version.png)
   
# 3. 컴파일된 모델을 HF 에 업로드 하기
- HF 키 저장
```
export API_TOKEN=<키 입력>
```
- 로그인
```
huggingface-cli login --token $API_TOKEN
```
- 업로드 ( Gonsoo/AWS-NeuronCC-2-14-llama-3-Korean-Bllossom-8B 는 본인의 모델 이름으로 수정 해주세요 ) 
```
huggingface-cli upload  Gonsoo/AWS-NeuronCC-2-14-llama-3-Korean-Bllossom-8B \
            ./data/llama-3-Korean-Bllossom-8B-NeuronCC-2.14 --exclude "checkpoint/**"
```

# 4. 모델 서빙하기
- 아래의 노트북을 실행하시면, SageMaker Endpoint 에서 추론을 할 수 있습니다. 특히 이 버전은 HF 의 모델을 다운로드 받고, S3에 업로드 합니다. SageMaker Endpoint 는 S3 에서 모델 파이라미터를 사용하여 모델 서빙을 합니다.
- [03-deploy-llama-3-neuron-moel-inferentia2-from-S3.ipynb](notebook/03-deploy-llama-3-neuron-moel-inferentia2-from-S3.ipynb)


**여기까지 오셨으면 성공 하셨습니다. 축하 드립니다. ^^**

---
Contributor: 문곤수 (Gonsoo Moon)