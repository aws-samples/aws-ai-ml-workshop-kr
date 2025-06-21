"""
Test script for SCM workflow with Chicago port strike scenario
"""

from src.workflow import run_scm_workflow

def test_chicago_port_strike():
    """Test SCM workflow with Chicago port strike scenario"""
    
    chicago_scenario = """
    2025년 5월, 미국 시카고 지역의 대형 항만 파업으로 인해 기존의 주요 물류 루트가 갑작스럽게 마비되었다. 
    
    상황 배경:
    - 2024년 10월 미국 동부/걸프만 36개 항만에서 파업 발생
    - 미국 전체 컨테이너 화물의 40% 이상 처리 중단
    - 홍해 위기로 아시아-유럽 운송비 5배, 중국-미국 2배 상승
    - 선박들이 아프리카 남단 우회, 운송시간 4,000마일/10-14일 추가
    - LG ES는 시카고를 경유하던 화물의 운송 경로를 남부 루트(휴스턴 경유)로 긴급 변경해야 함
    
    이것이 우리 회사에 미치는 영향력을 분석해주세요.
    """
    
    print("🔗 Starting Chicago Port Strike SCM Analysis...")
    print("="*60)
    
    try:
        result = run_scm_workflow(chicago_scenario, debug=True)
        
        print("\n" + "="*60)
        print("✅ SCM Analysis Complete!")
        print("="*60)
        
        # Show final results
        if "history" in result:
            print(f"\n📊 Analysis completed with {len(result['history'])} agent interactions")
            for i, history in enumerate(result["history"], 1):
                print(f"{i}. {history['agent']}: {len(history['message'])} characters of analysis")
        
        # Show artifacts summary
        from src.utils.scm_file_utils import get_artifacts_summary
        summary = get_artifacts_summary()
        print(f"\n📁 Generated {summary['total_files']} analysis files:")
        for file_info in summary['files']:
            print(f"   - {file_info['filename']} ({file_info['size_bytes']} bytes)")
            
        return result
        
    except Exception as e:
        print(f"❌ Error during SCM analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_chicago_port_strike()