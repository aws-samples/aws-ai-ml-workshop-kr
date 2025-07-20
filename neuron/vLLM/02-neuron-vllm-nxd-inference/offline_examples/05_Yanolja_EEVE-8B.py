import os
from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download, login

# === 설정 ===
MODEL_NAME = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
MODEL_DIR = os.path.abspath("local-models")
ARTIFACTS_PATH = f"{MODEL_DIR}/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B/"

# 캐시 디렉토리 설정
model_cache_name = MODEL_NAME.replace("/", "_")
cache_dir = f"/workspace/my-neuron-cache/{model_cache_name}"

# 캐시 디렉토리 생성
if not os.path.exists(cache_dir):
    print(f"[안내] 캐시 디렉토리 {cache_dir}를 생성합니다...")
    os.makedirs(cache_dir, exist_ok=True)

# HyperCLOVAX 모델에 최적화된 설정
tensor_parallel_size = 2
max_model_len = 2048  # HyperCLOVAX는 더 긴 컨텍스트 지원
block_size = 2048

# 환경 변수 설정
os.environ['NEURON_COMPILE_CACHE_URL'] = cache_dir
os.environ['NEURON_COMPILED_ARTIFACTS_PATH'] = ARTIFACTS_PATH
os.environ['NEURON_CONTEXT_LENGTH_BUCKETS'] = "2048"
os.environ['NEURON_TOKEN_GEN_BUCKETS'] = "2048"
os.environ['NEURON_CC_FLAGS'] = f"--cache_dir={cache_dir}"

# HyperCLOVAX 모델용 프롬프트 템플릿
def format_prompt(instruction, input_text=""):
    if input_text:
        return f"### 지시사항:\n{instruction}\n\n### 입력:\n{input_text}\n\n### 응답:\n"
    else:
        return f"### 지시사항:\n{instruction}\n\n### 응답:\n"

# 샘플 프롬프트 (HyperCLOVAX 모델에 최적화된 한국어 프롬프트)
prompts = [
    format_prompt("안녕하세요! 간단한 자기소개를 해주세요."),
    format_prompt("인공지능의 미래 발전 방향에 대해 설명해주세요."),
    format_prompt("서울의 주요 관광지에 대해 알려주세요."),
    format_prompt("한국의 전통 문화에 대해 설명해주세요.")
]

# HyperCLOVAX 모델에 최적화된 샘플링 파라미터
sampling_params = SamplingParams(
    temperature=0.8,  # 창의성 증가
    top_p=0.9,        # 더 다양한 응답
    repetition_penalty=1.1,  # 반복 방지
    max_tokens=1024   # 더 긴 응답 생성
)

def check_hf_token():
    """HuggingFace 토큰 확인 및 설정"""
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if not token:
        print("[경고] HuggingFace 토큰이 설정되지 않았습니다.")
        print("[안내] 다음 중 하나의 방법으로 토큰을 설정하세요:")
        print("1. 환경 변수 설정: export HF_TOKEN=your_token_here")
        print("2. 코드에서 직접 설정: os.environ['HF_TOKEN'] = 'your_token_here'")
        print("3. HuggingFace CLI 로그인: huggingface-cli login")
        return False
    
    try:
        login(token=token)
        print("[안내] HuggingFace 토큰 인증이 완료되었습니다.")
        return True
    except Exception as e:
        print(f"[오류] HuggingFace 토큰 인증 실패: {e}")
        return False

def check_and_download_model():
    config_path = os.path.join(ARTIFACTS_PATH, "config.json")
    if not os.path.exists(config_path):
        print(f"[안내] HyperCLOVAX 모델 파일이 {ARTIFACTS_PATH}에 없습니다.")
        
        # 토큰 확인
        if not check_hf_token():
            print("[오류] HyperCLOVAX 모델에 접근하려면 HuggingFace 토큰이 필요합니다.")
            return False
        
        print("HuggingFace Hub에서 HyperCLOVAX 모델을 다운로드합니다...")
        os.makedirs(os.path.dirname(ARTIFACTS_PATH), exist_ok=True)
        
        try:
            snapshot_download(
                repo_id=MODEL_NAME,
                local_dir=ARTIFACTS_PATH,
                local_dir_use_symlinks=False
            )
            if not os.path.exists(config_path):
                print(f"[오류] HyperCLOVAX 모델 다운로드에 실패했습니다.")
                return False
        except Exception as e:
            print(f"[오류] HyperCLOVAX 모델 다운로드 실패: {e}")
            return False
    
    return True

def main():
    if not check_and_download_model():
        print("[오류] HyperCLOVAX 모델 다운로드에 실패했습니다. 프로그램을 종료합니다.")
        return

    print(f"[안내] HyperCLOVAX-SEED-Text-Instruct-1.5B 모델을 로드합니다...")
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
    
    print("HyperCLOVAX 모델 추론을 시작합니다...")
    outputs = llm.generate(prompts, sampling_params)
    
    print("=" * 80)
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[응답 {i+1}]")
        print(f"프롬프트: {prompt}")
        print(f"생성된 텍스트: {generated_text}")
        print("-" * 80)

if __name__ == "__main__":
    main()
