# Run Llama-2-13b on inf2.32xlarge 로 

이 튜토리얼은 inf2.32xlarge 로 Llama-2-13b 모델을 로드하여 추론을 위한 가이드 입니다. 아래의 노트북을 실행 하기 위함 입니다. 
- [Run Hugging Face meta-llama/Llama-2-13b autoregressive sampling on Inf2 & Trn1](https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/transformers-neuronx/inference/meta-llama-2-13b-sampling.ipynb)

이를 위해서 단계별로 가이드 제공을 합니다. 또한 AWS Neuron Documentation 의 내용을 기본으로 했으며, <u>추후 업데이트로 인한 아래의 가이드가 에러 발생시에는 AWS Neuron Documentation 를 참조 해주시기 바랍니다.</u>

- [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/)

# 기본 사전 단계
- 