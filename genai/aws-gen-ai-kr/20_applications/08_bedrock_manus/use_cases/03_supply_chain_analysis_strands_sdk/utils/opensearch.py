import copy
import pandas as pd
from typing import List, Tuple, Dict, Any
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk

class opensearch_utils():
    
    @classmethod
    def create_aws_opensearch_client(cls, region: str, host: str, http_auth: Tuple[str, str]) -> OpenSearch:

        client = OpenSearch(
            hosts=[
                {'host': host.replace("https://", ""),
                 'port': 443
                }
            ],
            http_auth=http_auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )

        return client

    @classmethod
    def create_index(cls, os_client, index_name, index_body):
        '''
        인덱스 생성
        '''
        response = os_client.indices.create(
            index=index_name,  # 키워드 인자로 변경
            body=index_body
        )
        print('\nCreating index:')
        print(response)

    @classmethod
    def check_if_index_exists(cls, os_client, index_name):
        '''
        인덱스가 존재하는지 확인
        '''
        exists = os_client.indices.exists(index=index_name)  # 키워드 인자로 변경
        print(f"index_name={index_name}, exists={exists}")

        return exists

    @classmethod
    def delete_index_if_exists(cls, os_client, index_name):
        '''
        인덱스가 존재하면 삭제
        '''
        if cls.check_if_index_exists(os_client, index_name):
            cls.delete_index(os_client, index_name)
            return True
        return False

    @classmethod
    def add_doc(cls, os_client, index_name, document, id):
        '''
        # Add a document to the index.
        '''
        response = os_client.index(
            index = index_name,
            body = document,
            id = id,
            refresh = True
        )

        print('\nAdding document:')
        print(response)

    @classmethod
    def bulk_index_documents(cls, os_client, documents: List[Dict], index_name: str, 
                           mapping: Dict = None, id_field: str = None, 
                           batch_size: int = 500, create_index: bool = True) -> Dict:
        '''
        문서들을 벌크로 인덱싱
        
        Args:
            os_client: OpenSearch 클라이언트
            documents: 인덱싱할 문서 리스트 (딕셔너리 형태)
            index_name: 대상 인덱스명
            mapping: 인덱스 매핑 (None이면 기본 매핑 사용)
            id_field: ID로 사용할 필드명 (None이면 자동 생성)
            batch_size: 배치 크기
            create_index: 인덱스 생성 여부

            ex) documents = [
                    {"name": "Product A", "price": 100.0, "date": "2024-01-01"},
                    {"name": "Product B", "price": 200.0, "date": "2024-01-02"},
                    {"name": "Product C", "price": 300.0, "date": "2024-01-03"}
                ]
            
        Returns:
            인덱싱 결과 딕셔너리
        '''
        try:
            # 인덱스 생성 (필요한 경우)
            if create_index:
                if mapping:
                    cls.delete_index_if_exists(os_client, index_name)
                    # 매핑을 올바른 형태로 변환
                    index_body = {"mappings": mapping}
                    cls.create_index(os_client, index_name, index_body)
                else:
                    # 인덱스가 없으면 자동 생성됨 (dynamic mapping)
                    if not cls.check_if_index_exists(os_client, index_name):
                        print(f'ℹ️  인덱스가 존재하지 않음. 자동 매핑으로 생성됩니다: {index_name}')
            
            def doc_generator():
                for i, doc in enumerate(documents):
                    doc_id = doc.get(id_field) if id_field else i + 1
                    yield {
                        "_index": index_name,
                        "_id": doc_id,
                        "_source": doc
                    }
            
            # 벌크 인덱싱 실행
            success_count, errors = bulk(
                os_client,
                doc_generator(),
                chunk_size=batch_size,
                request_timeout=60
            )
            
            result = {
                "success": True,
                "indexed_count": success_count,
                "errors": errors,
                "total_documents": len(documents)
            }
            
            print(f'✅ 벌크 인덱싱 완료: {index_name} ({success_count}/{len(documents)}개 문서)')
            
            if errors:
                print(f'⚠️  {len(errors)}개 문서 인덱싱 실패')
                
            return result
            
        except Exception as e:
            print(f'❌ 벌크 인덱싱 실패 {index_name}: {e}')
            return {
                "success": False,
                "error": str(e),
                "indexed_count": 0,
                "total_documents": len(documents)
            }

    @classmethod
    def bulk_index_from_csv(cls, os_client, csv_path: str, index_name: str, 
                          mapping: Dict = None, id_field: str = None, 
                          batch_size: int = 500) -> Dict:
        '''
        CSV 파일에서 직접 벌크 인덱싱
        
        Args:
            os_client: OpenSearch 클라이언트
            csv_path: CSV 파일 경로
            index_name: 대상 인덱스명
            mapping: 인덱스 매핑 (None이면 인덱스 생성 안함)
            id_field: ID로 사용할 필드명
            batch_size: 배치 크기
            
        Returns:
            인덱싱 결과 딕셔너리
        '''
        try:
            # CSV 파일 읽기
            df = pd.read_csv(csv_path)
            print(f'📄 CSV 로드: {csv_path} ({len(df)}행)')
            
            # 인덱스 생성 (매핑이 제공된 경우)
            if mapping:
                cls.delete_index_if_exists(os_client, index_name)
                index_body = {"mappings": mapping}
                cls.create_index(os_client, index_name, index_body)
            
            # DataFrame을 딕셔너리 리스트로 변환
            documents = cls.prepare_documents_from_dataframe(df)
            
            # 벌크 인덱싱 실행
            result = cls.bulk_index_documents(
                os_client, documents, index_name, None, id_field, batch_size, False
            )
            
            return result
            
        except Exception as e:
            print(f'❌ CSV 벌크 인덱싱 실패 {csv_path}: {e}')
            return {
                "success": False,
                "error": str(e),
                "indexed_count": 0
            }

    @classmethod
    def prepare_documents_from_dataframe(cls, df: pd.DataFrame) -> List[Dict]:
        '''
        DataFrame을 OpenSearch 문서 형태로 변환
        '''
        documents = df.to_dict('records')
        
        # NaN 값을 None으로 변환
        for doc in documents:
            for key, value in doc.items():
                if pd.isna(value):
                    doc[key] = None
        
        return documents

    @classmethod
    def bulk_update_documents(cls, os_client, updates: List[Dict], index_name: str, 
                            batch_size: int = 500) -> Dict:
        '''
        문서들을 벌크로 업데이트
        
        Args:
            updates: [{"_id": "doc_id", "doc": {"field": "new_value"}}] 형태
        '''
        def update_generator():
            for update in updates:
                yield {
                    "_op_type": "update",
                    "_index": index_name,
                    "_id": update["_id"],
                    "doc": update["doc"]
                }
        
        try:
            success_count, errors = bulk(
                os_client,
                update_generator(),
                chunk_size=batch_size,
                request_timeout=60
            )
            
            result = {
                "success": True,
                "updated_count": success_count,
                "errors": errors,
                "total_updates": len(updates)
            }
            
            print(f'✅ 벌크 업데이트 완료: {index_name} ({success_count}/{len(updates)}개 문서)')
            return result
            
        except Exception as e:
            print(f'❌ 벌크 업데이트 실패 {index_name}: {e}')
            return {"success": False, "error": str(e)}

    @classmethod
    def bulk_delete_documents(cls, os_client, doc_ids: List[str], index_name: str, 
                            batch_size: int = 500) -> Dict:
        '''
        문서들을 벌크로 삭제
        '''
        def delete_generator():
            for doc_id in doc_ids:
                yield {
                    "_op_type": "delete",
                    "_index": index_name,
                    "_id": doc_id
                }
        
        try:
            success_count, errors = bulk(
                os_client,
                delete_generator(),
                chunk_size=batch_size,
                request_timeout=60
            )
            
            result = {
                "success": True,
                "deleted_count": success_count,
                "errors": errors,
                "total_deletes": len(doc_ids)
            }
            
            print(f'✅ 벌크 삭제 완료: {index_name} ({success_count}/{len(doc_ids)}개 문서)')
            return result
            
        except Exception as e:
            print(f'❌ 벌크 삭제 실패 {index_name}: {e}')
            return {"success": False, "error": str(e)}

    @classmethod
    def search_document(cls, os_client, query, index_name):
        response = os_client.search(
            body=query,
            index=index_name
        )
        #print('\nKeyword Search results:')
        return response

    @classmethod
    def delete_index(cls, os_client, index_name):
        response = os_client.indices.delete(
            index=index_name  # 키워드 인자로 변경
        )

        print('\nDeleting index:')
        print(response)

    @classmethod
    def get_index_stats(cls, os_client, index_name: str) -> Dict:
        '''
        인덱스 통계 정보 조회
        '''
        try:
            # 문서 수 조회
            count_response = os_client.count(index=index_name)
            doc_count = count_response['count']
            
            # 인덱스 사이즈 조회
            stats_response = os_client.indices.stats(index=index_name)
            index_size = stats_response['indices'][index_name]['total']['store']['size_in_bytes']
            
            # 매핑 정보 조회
            mapping_response = os_client.indices.get_mapping(index=index_name)
            
            return {
                "document_count": doc_count,
                "size_bytes": index_size,
                "size_mb": round(index_size / 1024 / 1024, 2),
                "mappings": mapping_response[index_name]['mappings']
            }
            
        except Exception as e:
            print(f'❌ 인덱스 통계 조회 실패 {index_name}: {e}')
            return {}

    @classmethod
    def parse_keyword_response(cls, response, show_size=3):
        '''
        키워드 검색 결과를 보여 줌.
        '''
        length = len(response['hits']['hits'])
        if length >= 1:
            print("# of searched docs: ", length)
            print(f"# of display: {show_size}")        
            print("---------------------")        
            for idx, doc in enumerate(response['hits']['hits']):
                print("_id in index: " , doc['_id'])
                print(doc['_score'])
                print(doc['_source']['text'])
                print(doc['_source']['metadata'])
                print("---------------------")
                if idx == show_size-1:
                    break
        else:
            print("There is no response")

    @classmethod
    def opensearch_pretty_print_documents(cls, response):
        '''
        OpenSearch 결과인 LIST 를 파싱하는 함수
        '''
        for doc, score in response:
            print(f'\nScore: {score}')
            print(f'Document Number: {doc.metadata["row"]}')

            # Split the page content into lines
            lines = doc.page_content.split("\n")

            # Extract and print each piece of information if it exists
            for line in lines:
                split_line = line.split(": ")
                if len(split_line) > 1:
                    print(f'{split_line[0]}: {split_line[1]}')

            print("Metadata:")
            print(f'Type: {doc.metadata["type"]}')
            print(f'Source: {doc.metadata["source"]}')        

            print('-' * 50)

    @classmethod
    def get_document(cls, os_client, doc_id, index_name):
        response = os_client.get(
            id= doc_id,
            index=index_name
        )

        return response

    @classmethod
    def get_count(cls, os_client, index_name):
        response = os_client.count(
            index=index_name
        )

        return response

    @classmethod
    def get_query(cls, **kwargs):

        # Reference:
        # OpenSearcj boolean query:
        #  - https://opensearch.org/docs/latest/query-dsl/compound/bool/
        # OpenSearch match qeury:
        #  - https://opensearch.org/docs/latest/query-dsl/full-text/index/#match-boolean-prefix
        # OpenSearch Query Description (한글)
        #  - https://esbook.kimjmin.net/05-search)

        search_type = kwargs.get("search_type", "lexical")

        if search_type == "lexical":

            min_shoud_match = 0
            if "minimum_should_match" in kwargs:
                min_shoud_match = kwargs["minimum_should_match"]

            QUERY_TEMPLATE = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "text": {
                                        "query": f'{kwargs["query"]}',
                                        "minimum_should_match": f'{min_shoud_match}%',
                                        "operator":  "or",
                                        # "fuzziness": "AUTO",
                                        # "fuzzy_transpositions": True,
                                        # "zero_terms_query": "none",
                                        # "lenient": False,
                                        # "prefix_length": 0,
                                        # "max_expansions": 50,
                                        # "boost": 1
                                    }
                                }
                            },
                        ],
                        "filter": [
                        ]
                    }
                }
            }

            if "filter" in kwargs:
                QUERY_TEMPLATE["query"]["bool"]["filter"].extend(kwargs["filter"])

        elif search_type == "semantic":

            QUERY_TEMPLATE = {
            "query": {
                    "bool": {
                        "must": [
                            {
                                "knn": {
                                    kwargs["vector_field"]: {
                                        "vector": kwargs["vector"],
                                        "k": kwargs["k"],
                                    }
                                }
                            },
                        ],
                        "filter": [
                        ]
                    }
                }
            }

            if "filter" in kwargs:
                QUERY_TEMPLATE["query"]["bool"]["filter"].extend(kwargs["filter"])

        return QUERY_TEMPLATE

    @classmethod
    def get_filter(cls, **kwargs):

        BOOL_FILTER_TEMPLATE = {
            "bool": {
                "filter": [
                ]
            }
        }

        if "filter" in kwargs:
            BOOL_FILTER_TEMPLATE["bool"]["filter"].extend(kwargs["filter"])

        return BOOL_FILTER_TEMPLATE

    @staticmethod
    def get_documents_by_ids(os_client, ids, index_name):

        response = os_client.mget(
            body={"ids": ids},
            index=index_name
        )

        return response

    @staticmethod
    def opensearch_pretty_print_documents_with_score(response):
        '''
        OpenSearch 결과인 LIST 를 파싱하는 함수
        '''
        responses = copy.deepcopy(response)
        for doc, score in responses:
            print(f'\nScore: {score}')
            # print(f'Document Number: {doc.metadata["row"]}')
            # Split the page content into lines
            lines = doc.page_content.split("\n")
            metadata = doc.metadata
            if "image_base64" in metadata: metadata["image_base64"] = ""
            if "orig_elements" in metadata: metadata["orig_elements"] = ""
            
            print(lines)
            print(metadata)


# ==================== 편의 함수들 ====================

def create_simple_mapping(field_types: Dict[str, str]) -> Dict:
    """
    간단한 필드 타입 딕셔너리로 매핑 생성
    
    Args:
        field_types: {"field_name": "field_type"} 형태
    """
    properties = {}
    for field_name, field_type in field_types.items():
        properties[field_name] = {"type": field_type}
    
    return {
        "mappings": {
            "properties": properties
        }
    }

def quick_csv_to_opensearch(csv_path: str, index_name: str, field_types: Dict[str, str], 
                          host: str = "localhost", port: int = 9200, 
                          username: str = None, password: str = None) -> Dict:
    """
    CSV 파일을 OpenSearch에 빠르게 인덱싱
    
    Args:
        csv_path: CSV 파일 경로
        index_name: 인덱스명
        field_types: 필드 타입 딕셔너리
        host, port: OpenSearch 서버 정보
        username, password: 인증 정보 (선택사항)
    
    Returns:
        인덱싱 결과
    """
    # 클라이언트 생성
    client = opensearch_utils.create_local_opensearch_client(host, port, username, password)
    
    # 매핑 생성
    mapping = create_simple_mapping(field_types)
    
    # CSV 인덱싱
    result = opensearch_utils.bulk_index_from_csv(
        client, csv_path, index_name, mapping
    )
    
    # 결과 확인
    if result['success']:
        stats = opensearch_utils.get_index_stats(client, index_name)
        print(f"📊 인덱스 통계: {stats['document_count']}개 문서, {stats['size_mb']}MB")
    
    return result