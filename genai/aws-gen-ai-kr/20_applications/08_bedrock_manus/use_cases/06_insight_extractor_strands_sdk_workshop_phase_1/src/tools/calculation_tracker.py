import os
import json
from datetime import datetime
from functools import wraps
from typing import Any, List, Union
import logging

logger = logging.getLogger(__name__)

class CalculationTracker:
    """계산 메타데이터를 자동으로 추적하는 클래스"""
    
    def __init__(self, metadata_file: str = './artifacts/calculation_metadata.json'):
        self.metadata_file = metadata_file
        self.calculations = []
        self.load_existing_metadata()
    
    def load_existing_metadata(self):
        """기존 메타데이터 파일 로드"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.calculations = data.get('calculations', [])
        except Exception as e:
            logger.warning(f"기존 메타데이터 로드 실패: {e}")
            self.calculations = []
    
    def track_calculation(
        self, 
        calc_id: str, 
        value: Union[float, int], 
        description: str, 
        formula: str,
        source_file: str = None,
        source_columns: List[str] = None,
        source_rows: str = "all rows",
        importance: str = "medium",
        notes: str = ""
    ):
        """계산 결과를 메타데이터에 추가"""
        calculation = {
            "id": calc_id,
            "value": float(value),
            "description": description,
            "formula": formula,
            "source_file": source_file,
            "source_columns": source_columns or [],
            "source_rows": source_rows,
            "importance": importance,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "verification_notes": notes
        }
        
        # 중복 제거 (같은 ID가 있으면 업데이트)
        existing_idx = None
        for i, calc in enumerate(self.calculations):
            if calc.get('id') == calc_id:
                existing_idx = i
                break
        
        if existing_idx is not None:
            self.calculations[existing_idx] = calculation
        else:
            self.calculations.append(calculation)
        
        self.save_metadata()
        return calculation
    
    def save_metadata(self):
        """메타데이터를 파일에 저장"""
        try:
            os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
            metadata = {"calculations": self.calculations}
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"계산 메타데이터가 {self.metadata_file}에 저장됨")
        except Exception as e:
            logger.error(f"메타데이터 저장 실패: {e}")

# 전역 트래커 인스턴스
_global_tracker = CalculationTracker()

def track_calculation(
    calc_id: str, 
    description: str, 
    formula: str,
    source_file: str = None,
    source_columns: List[str] = None,
    importance: str = "medium",
    notes: str = ""
):
    """계산 추적 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # 결과가 숫자인 경우에만 추적
            if isinstance(result, (int, float)):
                _global_tracker.track_calculation(
                    calc_id=calc_id,
                    value=result,
                    description=description,
                    formula=formula,
                    source_file=source_file,
                    source_columns=source_columns,
                    importance=importance,
                    notes=notes
                )
            
            return result
        return wrapper
    return decorator

def manual_track(
    calc_id: str,
    value: Union[float, int],
    description: str,
    formula: str,
    source_file: str = None,
    source_columns: List[str] = None,
    importance: str = "medium",
    notes: str = ""
):
    """수동으로 계산 결과 추적"""
    return _global_tracker.track_calculation(
        calc_id=calc_id,
        value=value,
        description=description,
        formula=formula,
        source_file=source_file,
        source_columns=source_columns,
        importance=importance,
        notes=notes
    )

def get_all_calculations():
    """모든 계산 메타데이터 반환"""
    return _global_tracker.calculations