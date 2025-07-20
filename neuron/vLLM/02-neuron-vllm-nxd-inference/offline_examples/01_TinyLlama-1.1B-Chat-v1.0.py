import os
from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download

# === 설정 ===
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_DIR = os.path.abspath("local-models")
ARTIFACTS_PATH = f"{MODEL_DIR}/TinyLlama/TinyLlama-1.1B-Chat-v1.0/"

# 캐시 디렉토리 설정
model_cache_name = MODEL_NAME.replace("/", "_")
cache_dir = f"/workspace/my-neuron-cache/{model_cache_name}"

# 캐시 디렉토리 생성
if not os.path.exists(cache_dir):
    print(f"[안내] 캐시 디렉토리 {cache_dir}를 생성합니다...")
    os.makedirs(cache_dir, exist_ok=True)

# 기본 설정
tensor_parallel_size = 2
max_model_len = 256
block_size = 256

# 환경 변수 설정
os.environ['NEURON_COMPILE_CACHE_URL'] = cache_dir
os.environ['NEURON_COMPILED_ARTIFACTS_PATH'] = ARTIFACTS_PATH
os.environ['NEURON_CONTEXT_LENGTH_BUCKETS'] = "256"
os.environ['NEURON_TOKEN_GEN_BUCKETS'] = "256"
os.environ['NEURON_CC_FLAGS'] = f"--cache_dir={cache_dir}"

# 샘플 프롬프트 (한글)
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
        os.makedirs(os.path.dirname(ARTIFACTS_PATH), exist_ok=True)
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

    print(f"[안내] 캐시 디렉토리: {cache_dir}")
    
    llm = LLM(
        model=ARTIFACTS_PATH,
        device="neuron",
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=4,
        max_model_len=max_model_len,
        block_size=block_size,
        override_neuron_config={
            "enable_bucketing": False,
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
