import requests
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
import json

def test_single_request(prompt, model_name="/workspace/local-models/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"):
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
                "max_tokens": 500,  # 50-500 토큰 범위를 위해 500으로 유지
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
    """사용 가능한 모델 목록을 확인합니다."""
    try:
        response = requests.get("http://localhost:8080/v1/models")
        response.raise_for_status()
        models = response.json()
        
        # 모델 목록 출력
        print("📋 사용 가능한 모델:")
        if "data" in models and models["data"]:
            for model in models["data"]:
                print(f"  - {model['id']}")
            return models["data"]  # data 배열만 반환
        else:
            print("  사용 가능한 모델이 없습니다.")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 모델 확인 중 오류 발생: {e}")
        return []
    except Exception as e:
        print(f"❌ 모델 목록 파싱 중 오류: {e}")
        return []

def run_benchmark(num_requests=64, concurrent_requests=1):
    """벤치마크 실행"""
    print(f"🚀 벤치마크 시작: {num_requests}개 요청, {concurrent_requests}개 동시 실행")
    print("=" * 50)
    
    # 벤치마크 전체 시작 시간 기록
    benchmark_start_time = time.time()
    
    # 먼저 사용 가능한 모델 확인
    available_models = check_available_models()
    if not available_models:
        print("⚠️  사용 가능한 모델을 찾을 수 없습니다.")
        return
    
    # 첫 번째 모델 사용
    model_name = available_models[0]["id"]
    print(f"🎯 사용할 모델: {model_name}")
    
    # 64개의 물건 거래 상황 관련 한글 테스트 프롬프트들 (50-500 토큰 범위의 답변이 나오도록 설계)
    test_prompts = [
        # 구매 상황 (8개)
        "온라인에서 신발을 살 때 주의사항을 알려주세요.",
        "중고 물건을 구매할 때 확인해야 할 사항들을 나열해주세요.",
        "전자제품 구매 시 체크리스트를 만들어주세요.",
        "의류 쇼핑할 때 사이즈 선택 방법을 알려주세요.",
        "가구 구매 시 고려사항을 설명해주세요.",
        "화장품 구매 시 성분 확인 방법을 알려주세요.",
        "식품 온라인 구매 시 주의사항을 제시해주세요.",
        "도서 구매 시 저자와 출판사 확인 방법을 설명해주세요.",
        
        # 판매 상황 (8개)
        "중고 물건을 팔 때 가격 책정 방법을 설명해주세요.",
        "온라인에서 물건을 팔 때 사진 촬영 팁을 알려주세요.",
        "상품 설명을 작성할 때 주의사항을 제시해주세요.",
        "거래 시 안전한 방법을 알려주세요.",
        "경매 사이트에서 물건을 팔 때 팁을 알려주세요.",
        "상품 키워드 설정 방법을 설명해주세요.",
        "판매자 평점 관리 방법을 제시해주세요.",
        "상품 배송 준비 방법을 알려주세요.",
        
        # 가격 흥정 (8개)
        "가격 흥정할 때 효과적인 방법을 설명해주세요.",
        "할인을 요청할 때 예의 바른 방법을 알려주세요.",
        "가격 비교하는 방법을 제시해주세요.",
        "가격 협상 시 주의사항을 설명해주세요.",
        "묶음 구매 시 할인 요청 방법을 알려주세요.",
        "정가 대비 할인율 계산 방법을 설명해주세요.",
        "가격 흥정 시 적정선 판단 방법을 제시해주세요.",
        "할인 쿠폰 활용 방법을 알려주세요.",
        
        # 품질 확인 (8개)
        "물건의 품질을 확인하는 방법을 알려주세요.",
        "가짜 상품을 구별하는 방법을 설명해주세요.",
        "제품 리뷰를 읽을 때 주의사항을 제시해주세요.",
        "상품 보증서 확인 방법을 알려주세요.",
        "제품 인증 마크 확인 방법을 설명해주세요.",
        "상품 제조일자와 유통기한 확인 방법을 알려주세요.",
        "제품 A/S 정책 확인 방법을 제시해주세요.",
        "상품 품질 등급 확인 방법을 설명해주세요.",
        
        # 배송/수령 (8개)
        "택배로 받은 물건을 확인하는 방법을 설명해주세요.",
        "직거래 시 만남 장소 선택 방법을 알려주세요.",
        "해외 배송 상품 구매 시 주의사항을 제시해주세요.",
        "부재 시 택배 수령 방법을 설명해주세요.",
        "배송 추적 방법을 알려주세요.",
        "배송 지연 시 대처 방법을 설명해주세요.",
        "택배 보관함 이용 방법을 제시해주세요.",
        "배송비 계산 방법을 알려주세요.",
        
        # 문제 해결 (8개)
        "받은 물건이 파손되었을 때 대처 방법을 알려주세요.",
        "사이즈가 맞지 않을 때 교환 방법을 설명해주세요.",
        "판매자가 연락이 안 될 때 해결 방법을 제시해주세요.",
        "환불 요청 시 필요한 절차를 알려주세요.",
        "상품 하자가 있을 때 대처 방법을 설명해주세요.",
        "배송 오류 발생 시 해결 방법을 제시해주세요.",
        "결제 오류 발생 시 대처 방법을 알려주세요.",
        "상품 설명과 다른 경우 해결 방법을 설명해주세요.",
        
        # 안전 거래 (8개)
        "온라인 거래 시 사기 방지 방법을 설명해주세요.",
        "직거래 시 안전한 만남 방법을 알려주세요.",
        "개인정보 보호 방법을 제시해주세요.",
        "거래 기록 보관의 중요성을 설명해주세요.",
        "안전한 결제 방법을 알려주세요.",
        "거래 시 계약서 작성 방법을 설명해주세요.",
        "사기 판매자 신고 방법을 제시해주세요.",
        "거래 분쟁 해결 방법을 알려주세요.",
        
        # 특수 상황 (8개)
        "경매에서 물건을 살 때 주의사항을 알려주세요.",
        "대량 구매 시 할인 요청 방법을 설명해주세요.",
        "기프트 상품 구매 시 주의사항을 제시해주세요.",
        "시즌 세일에서 물건을 살 때 팁을 알려주세요.",
        "프리오더 상품 구매 시 주의사항을 설명해주세요.",
        "한정판 상품 구매 방법을 알려주세요.",
        "공동구매 참여 시 주의사항을 제시해주세요.",
        "리셀 상품 구매 시 주의사항을 설명해주세요."
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
    
    # 벤치마크 전체 종료 시간 기록
    benchmark_end_time = time.time()
    total_benchmark_time = benchmark_end_time - benchmark_start_time
    
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
        
        # 기존 처리량 계산 (개별 응답 시간 기반)
        total_response_time = sum(response_times)
        if total_response_time > 0:
            requests_per_second_individual = len(successful_results) / total_response_time
            print(f"\n🚀 개별 응답 시간 기반 처리량: {requests_per_second_individual:.2f} 요청/초")
        
        # 새로운 총 처리량 계산 (전체 벤치마크 시간 기반)
        print(f"\n🚀 총 처리량 지표:")
        print(f"  전체 실행 시간: {total_benchmark_time:.2f}초")
        print(f"  총 처리량: {len(successful_results)/total_benchmark_time:.2f} 요청/초")
        print(f"  토큰 처리량: {sum(tokens_used)/total_benchmark_time:.1f} 토큰/초")
        
        # 효율성 비교
        if total_response_time > 0:
            efficiency_ratio = total_benchmark_time / total_response_time
            print(f"  효율성 비율: {efficiency_ratio:.2f} (1.0에 가까울수록 효율적)")
    
    if failed_results:
        print(f"\n❌ 실패한 요청들:")
        for i, result in enumerate(failed_results):
            print(f"  {i+1}. {result['error']}")
            if "error_detail" in result:
                print(f"     상세: {result['error_detail']}") 

if __name__ == "__main__":
    # 물건 거래 상황 벤치마크 (64개 요청, 순차 실행)
    run_benchmark(num_requests=64, concurrent_requests=1)
    
    print("\n" + "=" * 50)
    
    # 동시 실행 벤치마크 (64개 요청, 4개 동시 실행)
    run_benchmark(num_requests=64, concurrent_requests=4)
    
    print("\n" + "=" * 50)
    
    # 동시 실행 벤치마크 (64개 요청, 16개 동시 실행)
    run_benchmark(num_requests=64, concurrent_requests=16)
    
    print("\n" + "=" * 50)
    
    # 동시 실행 벤치마크 (64개 요청, 32개 동시 실행)
    run_benchmark(num_requests=64, concurrent_requests=32)
    
    print("\n" + "=" * 50)
    
    # 동시 실행 벤치마크 (64개 요청, 32개 동시 실행)
    run_benchmark(num_requests=64, concurrent_requests=64)
    
    print("\n" + "=" * 50)
