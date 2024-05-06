from typing import List, Tuple
from opensearchpy import OpenSearch, RequestsHttpConnection
import concurrent.futures
import tqdm.notebook as tq

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
            index_name,
            body=index_body
        )
        print('\nCreating index:')
        print(response)

    @classmethod
    def check_if_index_exists(cls, os_client, index_name):
        '''
        인덱스가 존재하는지 확인
        '''
        exists = os_client.indices.exists(index_name)
        print(f"index_name={index_name}, exists={exists}")

        return exists

    @classmethod
    def add_doc(cls, os_client, index_name, document, verbose = False, id=None):
        '''
        # Add a document to the index.
        '''
        if id:
            response = os_client.index(
                index = index_name,
                body = document,
                id = id,
                refresh = True
            )
        else:
            response = os_client.index(
                index = index_name,
                body = document,
            )            

        if verbose:
            print('\nAdding document:')
            print(response)
        
        return response

    @classmethod
    def search_document(cls, os_client, query, index_name):
        response = os_client.search(
            body=query,
            index=index_name
        )
        #print('\nKeyword Search results:')
        return response

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
    def delete_index(cls, os_client, index_name):
        response = os_client.indices.delete(
            index=index_name
        )

        print('\nDeleting index:')
        print(response)

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
        for doc, score in response:
            print(f'\nScore: {score}')
            # print(f'Document Number: {doc.metadata["row"]}')

            # Split the page content into lines
            lines = doc.page_content.split("\n")
            metadata = doc.metadata
            print(lines)
            print(metadata)

    @classmethod
    def create_opensearch_doc(cls, image_vector,description, item_id, image_url):
        document = {
                    'image_vector': image_vector,
                    "description": description,
                    "item_id" : item_id,
                    "image_url": image_url,  
                    }
        return document


    @classmethod
    def _add_doc_to_opensearch(cls, os_client, index_name, dataset):
        '''
        데이터 세트를 오픈 서치에 넣는 함수
        '''
        for idx, record in tq.tqdm(dataset.iterrows(), total=len(dataset)):
            image_vector = dataset['embedding_img'][idx]
            description = dataset['item_name_in_en_us'][idx]
            item_id = dataset['item_id'][idx]
            image_url = dataset['img_full_path'][idx]

            document = opensearch_utils.create_opensearch_doc(image_vector,description, item_id, image_url)   
            response = opensearch_utils.add_doc(os_client, index_name,document, verbose=False, id = None)               

        total_count_docs  = total_count_docs = opensearch_utils.get_count(os_client, index_name)

        return total_count_docs

    @classmethod
    def parallel_add_doc_to_opensearch(cls, os_client, num_dataset, num_worker, index_name, dataset):        
        '''
        Add dataset in parallel to opensearch
        '''

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_worker) as executor:
            futures = []
            for i in range(num_dataset):
                # futures.append(executor.submit(_inference_with_latency_calculation, i))
                futures.append(executor.submit(cls._add_doc_to_opensearch, os_client, index_name, dataset))
                # print("i; ", i)
            for future in concurrent.futures.as_completed(futures):
                get_result = future.result()
                print("get_result: ", get_result)


    
    
