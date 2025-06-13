# AWS Neuron 가이드

AWS Neuron (Trainium, Trainium1, Inferentia, Inferentia2 ) 에 관련 링크, 튜토리얼, 가이드를 제공 합니다.

Last updated: Mar 31, 2024

---


# 1. Quick Links
## AWS Neuron
- AWS Neuron 공식 문서: [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/)
- AWS Neuron 공식 Git Repo: [aws-neuron-samples](https://github.com/aws-neuron/aws-neuron-samples)
- Trainium 에서 지원 하는 모델 확인: [Training Samples/Tutorials](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/models/training-trn1-samples.html#model-samples-training-trn1)
- Inferentia2/Trainium 에서 지원 하는 모델 확인: [Inference Samples/Tutorials (Inf2/Trn1)
](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/models/inference-inf2-trn1-samples.html#model-samples-inference-inf2-trn1)
- Inferentia 에서 지원 하는 모델 확인: [Inference Samples/Tutorials (Inf1)
](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/models/inference-inf1-samples.html#model-samples-inference-inf1)

## Hugging Face Optimum Neuron
- Hugging Face 로 쉽게 AWS Neuron 활용: [Hugging Face Optimum Neuron](https://huggingface.co/docs/optimum-neuron/index)
-  Hugging Face Optimum Neuron 지원 아키텍처:  [지원 아키텍처](https://huggingface.co/docs/optimum-neuron/package_reference/supported_models)
- Hugging Face Optimum Neuron Git Repo: [Optimum-neuron git](https://github.com/huggingface/optimum-neuron.git)

## vLLM
- [Installation with Neuron](https://docs.vllm.ai/en/latest/getting_started/neuron-installation.html)
        
<p>

# 2. 튜토리얼 및 코드 예시
여기는 AWS Neuron 을 사용한  튜토리얼, 코드, 지직 및 Tip 을 제공합니다.

## 2.1 AWS Neuron 
- (Feb 2024) [Run Hugging Face `meta-llama/Llama-2-13b autoregressive sampling on Inf2 & Trn1](tutorial/inference-Llama-2-13b/README.md)

## 2.2. Hugging Face Optimum Neuron
- (Sep 2024) [한국어 파인 튜닝 모델의 INF2 및 SageMaker 배포](hf-optimum/04-Deploy-Llama3-8B-HF-TGI-Docker-On-INF2/README.md)   
- (Feb 2024) [AWS Inferentia 기반 위에 llama-2-13B 이용하여 챗봇 데모](hf-optimum/01-Chatbot-Llama-2-13B-Inf2/README.md)
- (Feb 2024) [AWS Trainium 기반 위에 llama-2-7B 및 Dolly Dataset 으로 파인 튜닝](hf-optimum/02-Fine-tune-Llama-7B-Trn1/README.md)

## 2.3. vLLM on Inferentia/Trainium 
- (Mar 2024) SOLAR-10.7B-instruct, yanolja-KoSOLAR-10.7B, 04-yanolja-EEVE-Korean-Instruct-10.8B 배치 추론 함: [vLLM 으로 Inferentia2 (inf2.48xlarge)에서 배치성 추론 하기](vLLM/01-offline-inference_neuron/Readme.md)

# 3. 관련 블로그
- [주요 블로그 보기](blog/Readme.md)
---

## License
This library is licensed under the Apache 2.0 License. For more details, please take a look at the LICENSE file.

---

## Contributing
Although we're extremely excited to receive contributions from the community, we're still working on the best mechanism to take in examples from external sources. Please bear with us in the short-term if pull requests take longer than expected or are closed. Please read our contributing guidelines if you'd like to open an issue or submit a pull request.

---
