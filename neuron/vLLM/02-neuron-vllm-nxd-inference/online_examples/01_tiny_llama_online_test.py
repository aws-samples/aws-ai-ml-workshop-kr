import requests
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
import json

def test_single_request(prompt, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """단일 요청 테스트"""
    start_time = time.time()
    
    try:
        # 한글 프롬프트를 위한 헤더 설정
        headers = {
            "Content-Type": "application/json; charset=utf-8"
        }
        
        response = requests.post(
            "http://localhost:8080/v1/chat/completions",
            headers=headers,
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,  # 50에서 200으로 증가
                "temperature": 0.7
            },
            timeout=30
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            tokens_used = result.get("usage", {}).get("total_tokens", 0)
            return {
                "success": True,
                "response_time": response_time,
                "tokens_used": tokens_used,
                "response": result["choices"][0]["message"]["content"]
            }
        else:
            # 에러 응답 내용도 포함
            error_detail = ""
            try:
                error_detail = response.json()
            except:
                error_detail = response.text
            
            return {
                "success": False,
                "response_time": response_time,
                "error": f"HTTP {response.status_code}",
                "error_detail": error_detail
            }
            
    except Exception as e:
        return {
            "success": False,
            "response_time": time.time() - start_time,
            "error": str(e)
        }

def check_available_models():
    """사용 가능한 모델 확인"""
    try:
        response = requests.get("http://localhost:8080/v1/models")
        if response.status_code == 200:
            models = response.json()
            print("📋 사용 가능한 모델:")
            for model in models.get("data", []):
                print(f"  - {model['id']}")
            return models.get("data", [])
        else:
            print(f"❌ 모델 목록 조회 실패: HTTP {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ 모델 목록 조회 중 오류: {e}")
        return []

def run_benchmark(num_requests=10, concurrent_requests=1):
    """벤치마크 실행"""
    print(f"🚀 벤치마크 시작: {num_requests}개 요청, {concurrent_requests}개 동시 실행")
    print("=" * 50)
    
    # 먼저 사용 가능한 모델 확인
    available_models = check_available_models()
    if not available_models:
        print("⚠️  사용 가능한 모델을 찾을 수 없습니다.")
        return
    
    # 첫 번째 모델 사용
    model_name = available_models[0]["id"]
    print(f"🎯 사용할 모델: {model_name}")
    
    # 한글 테스트 프롬프트들 (원래대로 복원)
    test_prompts = [
        "안녕하세요! 간단한 인사말을 해주세요.",
        "파리는 어느 나라의 수도인가요?",
        "1+1은 무엇인가요?",
        "재미있는 농담을 하나 해주세요.",
        "파이썬이란 무엇인가요?"
    ]
    
    results = []
    
    if concurrent_requests == 1:
        # 순차 실행
        for i in range(num_requests):
            prompt = test_prompts[i % len(test_prompts)]
            print(f"요청 {i+1}/{num_requests}: {prompt[:30]}...")
            
            result = test_single_request(prompt, model_name)
            results.append(result)
            
            if result["success"]:
                print(f"  ✅ 성공 - {result['response_time']:.2f}초, {result['tokens_used']} 토큰")
                print(f"     응답: {result['response'][:50]}...")
            else:
                print(f"  ❌ 실패 - {result['error']}")
                if "error_detail" in result:
                    print(f"     상세: {result['error_detail']}")
    
    else:
        # 동시 실행
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = []
            for i in range(num_requests):
                prompt = test_prompts[i % len(test_prompts)]
                futures.append(executor.submit(test_single_request, prompt, model_name))
            
            for i, future in enumerate(futures):
                result = future.result()
                results.append(result)
                print(f"요청 {i+1}/{num_requests} 완료 - {result['response_time']:.2f}초")
    
    # 결과 분석
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    print("\n" + "=" * 50)
    print("📊 벤치마크 결과")
    print("=" * 50)
    
    print(f"총 요청 수: {len(results)}")
    print(f"성공: {len(successful_results)}")
    print(f"실패: {len(failed_results)}")
    print(f"성공률: {len(successful_results)/len(results)*100:.1f}%")
    
    if successful_results:
        response_times = [r["response_time"] for r in successful_results]
        tokens_used = [r["tokens_used"] for r in successful_results]
        
        print(f"\n⏱️  응답 시간:")
        print(f"  평균: {statistics.mean(response_times):.2f}초")
        print(f"  중간값: {statistics.median(response_times):.2f}초")
        print(f"  최소: {min(response_times):.2f}초")
        print(f"  최대: {max(response_times):.2f}초")
        
        print(f"\n📝 토큰 사용량:")
        print(f"  평균: {statistics.mean(tokens_used):.1f} 토큰")
        print(f"  총합: {sum(tokens_used)} 토큰")
        
        # 초당 처리량 계산
        total_time = sum(response_times)
        if total_time > 0:
            requests_per_second = len(successful_results) / total_time
            print(f"\n 처리량: {requests_per_second:.2f} 요청/초")
    
    if failed_results:
        print(f"\n❌ 실패한 요청들:")
        for i, result in enumerate(failed_results):
            print(f"  {i+1}. {result['error']}")
            if "error_detail" in result:
                print(f"     상세: {result['error_detail']}")

if __name__ == "__main__":
    # 한글 프롬프트 벤치마크 (10개 요청, 순차 실행)
    run_benchmark(num_requests=10, concurrent_requests=1)
    
    print("\n" + "=" * 50)
    
    # 동시 실행 벤치마크 (5개 요청, 2개 동시 실행)
    run_benchmark(num_requests=5, concurrent_requests=2)