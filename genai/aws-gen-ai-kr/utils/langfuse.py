from functools import wraps
from langfuse.decorators import observe

def filtered_observe(fields_to_track, **observe_kwargs):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_result = func(*args, **kwargs)
            
            # Langfuse에 기록할 필터링된 결과 생성
            if isinstance(original_result, dict) and fields_to_track:
                filtered_result = {k: original_result[k] for k in fields_to_track if k in original_result}
                # Langfuse 트래킹 코드
                langfuse_span = observe(**observe_kwargs)(lambda: filtered_result)()
            else:
                # 딕셔너리가 아니거나 필터링할 필드가 없는 경우
                langfuse_span = observe(**observe_kwargs)(lambda: original_result)()
            
            # 원래 함수의 결과 그대로 반환
            return original_result
        
        return wrapper
    return decorator