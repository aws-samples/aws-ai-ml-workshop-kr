import os
import sys
import boto3
import pandas as pd

# 모듈 경로 추가
module_path = ".."
sys.path.append(os.path.abspath(module_path))

from utils.ssm import parameter_store
from utils.opensearch import opensearch_utils

"""
공급망 데이터 완전 인덱싱 스크립트 (AWS OpenSearch용)
ChainScope AI 프로젝트용

사용법:
    python complete_supply_chain_indexing.py

필요한 파일:
    - ./data/shipment_tracking_data.txt
    - ./data/order_fulfillment_data.txt  
    - ./data/inventory_levels_data.txt
    - ./data/supplier_performance_data.txt
    - ./data/ira_compliance_data.txt

AWS 설정:
    - AWS OpenSearch 도메인 엔드포인트
    - 인증 정보 (username, password 또는 IAM 역할)
"""

def index_all_supply_chain_data(region="us-east-1", host=None, username=None, password=None):
    """
    모든 공급망 데이터를 AWS OpenSearch에 인덱싱
    
    Args:
        region: AWS 리전 (예: "us-east-1", "ap-northeast-2")
        host: AWS OpenSearch 도메인 엔드포인트 (https://your-domain.region.es.amazonaws.com)
        username: OpenSearch 사용자명
        password: OpenSearch 비밀번호
    """
    
    print("🚀 공급망 데이터 인덱싱 시작 (AWS OpenSearch)")
    print("="*60)
    
    # AWS OpenSearch 연결 정보 확인
    if not host:
        print("❌ AWS OpenSearch 호스트가 지정되지 않았습니다.")
        print("   예: https://your-domain.ap-northeast-2.es.amazonaws.com")
        return False
        
    if not username or not password:
        print("❌ AWS OpenSearch 인증 정보가 필요합니다.")
        print("   username과 password를 지정해주세요.")
        return False
    
    # 1. AWS OpenSearch 클라이언트 생성
    try:
        client = opensearch_utils.create_aws_opensearch_client(
            region=region,
            host=host,
            http_auth=(username, password)
        )
        print(f"✅ AWS OpenSearch 연결 성공")
        print(f"   리전: {region}")
        print(f"   호스트: {host}")
    except Exception as e:
        print(f"❌ AWS OpenSearch 연결 실패: {e}")
        return False
    
    # 2. 매핑 및 파일 설정
    #mappings = get_supply_chain_mappings()

    mappings = {
        "shipment_tracking": {
            "properties": {
            "date": {"type": "date", "format": "yyyy-MM-dd"},
            "shipment_id": {"type": "keyword"},
            "origin_port": {"type": "keyword"},
            "destination_port": {"type": "keyword"},
            "route_type": {"type": "keyword"},
            "lead_time_days": {"type": "integer"},
            "transport_cost_usd": {"type": "float"},
            "cargo_type": {"type": "keyword"},
            "volume_containers": {"type": "integer"},
            "timestamp": {"type": "date"}
            }
        },
        "order_fulfillment": {
            "properties": {
            "date": {"type": "date"},
            "customer_id": {"type": "keyword"},
            "order_id": {"type": "keyword"},
            "requested_delivery": {"type": "date"},
            "actual_delivery": {"type": "date"},
            "status": {"type": "keyword"},
            "order_value_usd": {"type": "float"},
            "penalty_applied": {"type": "boolean"},
            "delay_days": {"type": "integer"},
            "fulfillment_rate": {"type": "float"}
            }
        },
        "inventory_levels": {
            "properties": {
            "date": {"type": "date"},
            "material_type": {"type": "keyword"},
            "location": {"type": "keyword"},
            "quantity_units": {"type": "float"},
            "safety_stock_days": {"type": "integer"},
            "current_days_supply": {"type": "float"},
            "reorder_triggered": {"type": "boolean"},
            "inventory_turnover": {"type": "float"}
            }
        },
        "supplier_performance": {
            "properties": {
            "date": {"type": "date"},
            "supplier_id": {"type": "keyword"},
            "region": {"type": "keyword"},
            "on_time_delivery_rate": {"type": "float"},
            "quality_score": {"type": "float"},
            "lead_time_variance_days": {"type": "float"},
            "communication_score": {"type": "float"},
            "overall_performance": {"type": "float"}
            }
        },
        "ira_compliance": {
            "properties": {
            "date": {"type": "date"},
            "shipment_id": {"type": "keyword"},
            "material_origin": {"type": "keyword"},
            "processing_location": {"type": "keyword"},
            "fta_compliant": {"type": "boolean"},
            "china_content_ratio": {"type": "float"},
            "compliance_status": {"type": "keyword"},
            "compliance_score": {"type": "float"}
            }
        }
    }
    
    csv_files = {
        "shipment_tracking": "./data/shipment_tracking_data.txt",
        "order_fulfillment": "./data/order_fulfillment_data.txt", 
        "inventory_levels": "./data/inventory_levels_data.txt",
        "supplier_performance": "./data/supplier_performance_data.txt",
        "ira_compliance": "./data/ira_compliance_data.txt"
    }
    
    # 3. 각 파일 인덱싱
    results = {}
    total_documents = 0
    
    for index_name, csv_path in csv_files.items():
        print(f"\n📁 인덱싱 중: {index_name}")
        print("-" * 40)
        
        try:
            # CSV 파일 로드
            df = pd.read_csv(csv_path)
            documents = opensearch_utils.prepare_documents_from_dataframe(df)
            
            # 벌크 인덱싱
            result = opensearch_utils.bulk_index_documents(
                os_client=client,
                documents=documents,
                index_name=index_name,
                mapping=mappings[index_name],
                batch_size=500
            )
            
            results[index_name] = result
            if result['success']:
                total_documents += result['indexed_count']
                print(f"   📊 {result['indexed_count']}개 문서 인덱싱 완료")
            else:
                print(f"   ❌ 인덱싱 실패: {result['error']}")
                
        except FileNotFoundError:
            print(f"   ❌ 파일을 찾을 수 없습니다: {csv_path}")
            results[index_name] = {"success": False, "error": "File not found"}
        except Exception as e:
            print(f"   ❌ 예외 발생: {e}")
            results[index_name] = {"success": False, "error": str(e)}
    
    # 4. 인덱싱 결과 요약
    print(f"\n📊 인덱싱 결과 요약")
    print("="*60)
    
    successful_indices = []
    failed_indices = []
    
    for index_name, result in results.items():
        if result['success']:
            successful_indices.append(index_name)
            stats = opensearch_utils.get_index_stats(client, index_name)
            print(f"✅ {index_name:20} : {stats.get('document_count', 0):>6}개 문서 ({stats.get('size_mb', 0):>6.1f}MB)")
        else:
            failed_indices.append(index_name)
            print(f"❌ {index_name:20} : 실패")
    
    print(f"\n🎯 총 결과:")
    print(f"   성공: {len(successful_indices)}/{len(csv_files)}개 인덱스")
    print(f"   총 문서: {total_documents:,}개")
    
    if failed_indices:
        print(f"   실패한 인덱스: {', '.join(failed_indices)}")
    
    return len(failed_indices) == 0

def verify_supply_chain_data(client=None, region="us-east-1", host=None, username=None, password=None):
    """
    인덱싱된 데이터 검증
    """
    if client is None:
        if not all([host, username, password]):
            print("❌ AWS OpenSearch 연결 정보가 필요합니다.")
            return
            
        client = opensearch_utils.create_aws_opensearch_client(
            region=region, host=host, http_auth=(username, password)
        )
    
    print("\n🔍 데이터 검증")
    print("="*60)
    
    indices = ["shipment_tracking", "order_fulfillment", "inventory_levels", 
               "supplier_performance", "ira_compliance"]
    
    for index_name in indices:
        try:
            # 기본 통계
            count = opensearch_utils.get_count(client, index_name)['count']
            
            # 샘플 검색
            sample_query = {
                "query": {"match_all": {}},
                "size": 1,
                "sort": [{"date": {"order": "desc"}}]
            }
            
            sample_response = opensearch_utils.search_document(client, sample_query, index_name)
            
            if sample_response['hits']['total']['value'] > 0:
                latest_doc = sample_response['hits']['hits'][0]['_source']
                latest_date = latest_doc.get('date', 'N/A')
                print(f"✅ {index_name:20} : {count:>6}개 문서, 최근 데이터: {latest_date}")
            else:
                print(f"⚠️  {index_name:20} : 문서 없음")
                
        except Exception as e:
            print(f"❌ {index_name:20} : 오류 - {e}")

def run_sample_analysis_queries(client=None, region="us-east-1", host=None, username=None, password=None):
    """
    샘플 분석 쿼리 실행
    """
    if client is None:
        if not all([host, username, password]):
            print("❌ AWS OpenSearch 연결 정보가 필요합니다.")
            return
            
        client = opensearch_utils.create_aws_opensearch_client(
            region=region, host=host, http_auth=(username, password)
        )
    
    print("\n📈 샘플 분석 쿼리")
    print("="*60)
    
    # 1. 10월 이후 운송비 평균
    transport_query = {
        "query": {
            "range": {
                "date": {"gte": "2024-10-01"}
            }
        },
        "aggs": {
            "avg_cost": {
                "avg": {"field": "transport_cost_usd"}
            }
        },
        "size": 0
    }
    
    try:
        response = opensearch_utils.search_document(client, transport_query, "shipment_tracking")
        avg_cost = response['aggregations']['avg_cost']['value']
        print(f"📊 10월 이후 평균 운송비: ${avg_cost:,.2f}")
    except Exception as e:
        print(f"❌ 운송비 분석 실패: {e}")
    
    # 2. 고객별 지연 주문 수
    delay_query = {
        "query": {
            "term": {"status": "Delayed"}
        },
        "aggs": {
            "by_customer": {
                "terms": {"field": "customer_id", "size": 5}
            }
        },
        "size": 0
    }
    
    try:
        response = opensearch_utils.search_document(client, delay_query, "order_fulfillment")
        customers = response['aggregations']['by_customer']['buckets']
        print(f"📊 지연 주문이 많은 고객 TOP 3:")
        for i, customer in enumerate(customers[:3], 1):
            print(f"   {i}. {customer['key']}: {customer['doc_count']}건")
    except Exception as e:
        print(f"❌ 지연 주문 분석 실패: {e}")
    
    # 3. IRA 준수율
    compliance_query = {
        "aggs": {
            "compliance_rate": {
                "avg": {
                    "script": {
                        "source": "doc['fta_compliant'].value ? 1 : 0"
                    }
                }
            }
        },
        "size": 0
    }
    
    try:
        response = opensearch_utils.search_document(client, compliance_query, "ira_compliance")
        compliance_rate = response['aggregations']['compliance_rate']['value'] * 100
        print(f"📊 전체 IRA 준수율: {compliance_rate:.1f}%")
    except Exception as e:
        print(f"❌ IRA 준수율 분석 실패: {e}")

def main():
    """
    메인 실행 함수
    """
    print("🎯 ChainScope AI - 공급망 데이터 인덱싱 (AWS OpenSearch)")
    print("="*60)
    
    # AWS OpenSearch 설정 (여기서 실제 값으로 변경하세요)
    region = boto3.Session().region_name
    pm = parameter_store(region)

    AWS_CONFIG = {
        "region": region,
        "host": pm.get_params(key="opensearch_domain_endpoint", enc=False),
        "username": pm.get_params(key="opensearch_user_id", enc=False),
        "password": pm.get_params(key="opensearch_user_password", enc=True)
    }
    
    print("📋 AWS OpenSearch 설정:")
    print(f"   리전: {AWS_CONFIG['region']}")
    print(f"   호스트: {AWS_CONFIG['host']}")
    print(f"   사용자: {AWS_CONFIG['username']}")
    print()
    
    # 1. 데이터 인덱싱
    success = index_all_supply_chain_data(**AWS_CONFIG)
    
    if not success:
        print("\n❌ 인덱싱에 실패했습니다. 로그를 확인해주세요.")
        return
    
    # 2. 데이터 검증
    verify_supply_chain_data(**AWS_CONFIG)
    
    # 3. 샘플 분석
    run_sample_analysis_queries(**AWS_CONFIG)
    
    print(f"\n🎉 모든 작업이 완료되었습니다!")
    print(f"   이제 AWS OpenSearch에서 이 데이터를 분석할 수 있습니다.")

if __name__ == "__main__":
    main()