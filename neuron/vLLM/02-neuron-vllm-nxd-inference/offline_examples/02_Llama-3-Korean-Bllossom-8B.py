import os
from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download

# === 설정 ===
MODEL_NAME = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
MODEL_DIR = "/workspace/local-models"
ARTIFACTS_PATH = f"{MODEL_DIR}/MLP-KTLim/llama-3-Korean-Bllossom-8B/"

# 환경변수 설정
os.environ['NEURON_CONTEXT_LENGTH_BUCKETS'] = "512,1024,2048"
os.environ['NEURON_TOKEN_GEN_BUCKETS'] = "256,512,1024"
os.environ['NEURON_COMPILED_ARTIFACTS_PATH'] = ARTIFACTS_PATH

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

def check_and_download_model():
    config_path = os.path.join(ARTIFACTS_PATH, "config.json")
    if not os.path.exists(config_path):
        print(f"[안내] 모델 파일이 {ARTIFACTS_PATH}에 없습니다. HuggingFace Hub에서 자동 다운로드합니다...")
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=ARTIFACTS_PATH,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        if not os.path.exists(config_path):
            print(f"[오류] 모델 다운로드에 실패했습니다. HuggingFace 토큰 또는 네트워크를 확인하세요.")
            return False
    return True

def main():
    if not check_and_download_model():
        return

    llm = LLM(
        model=MODEL_NAME,
        device="neuron",
        tensor_parallel_size=2,
        max_num_seqs=4,
        max_model_len=2048,
        block_size=1024,
        download_dir=MODEL_DIR,
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