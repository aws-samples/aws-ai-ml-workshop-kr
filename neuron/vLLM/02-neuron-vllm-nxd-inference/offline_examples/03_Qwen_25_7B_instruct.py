import os
from vllm import LLM, SamplingParams

# 환경변수 설정 (필요에 따라 추가/삭제)
os.environ['NEURON_CONTEXT_LENGTH_BUCKETS'] = "512,1024,2048"
os.environ['NEURON_TOKEN_GEN_BUCKETS'] = "256,512,1024"
os.environ['NEURON_COMPILED_ARTIFACTS_PATH'] = "/workspace/local-models/Qwen/Qwen2.5-7B-Instruct/"

# 샘플 프롬프트
prompts = [
    "안녕하세요, 제 이름은",
    "인공지능의 미래는", 
    "서울의 날씨는",
    "대한민국의 수도는"
]

# 샘플링 파라미터 설정
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    repetition_penalty=1.05,
    max_tokens=512
)

def main():
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        device="neuron",
        tensor_parallel_size=2,
        max_num_seqs=4,
        max_model_len=2048,
        block_size=1024,
        download_dir="/workspace/local-models",  # ← 모델 경로 지정
        override_neuron_config={
            "enable_bucketing": True,
            "sequence_parallel_enabled": False
        }
    )
    
    print("추론을 시작합니다...")
    outputs = llm.generate(prompts, sampling_params)
    
    print("=" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"프롬프트: {prompt}")
        print(f"생성된 텍스트: {generated_text}")
        print("-" * 60)

if __name__ == "__main__":
    main()