from typing import List, Dict
import boto3
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection

import logging

logger = logging.getLogger(__name__)

class OpensearchService:
    def __init__(self, aws_region: str, aws_profile: str, prefix: str, domain_name: str = None, document_name: str = None, user: None = None, password: None = None):
        self.aws_region = aws_region
        self.aws_profile = aws_profile
        self.prefix = prefix
        self.domain_name = domain_name
        self.document_name = document_name
        self.user = user
        self.password = password

        self.opensearch_client = self._init_opensearch_client(self.aws_region, self.aws_profile, self.domain_name, self.user, self.password)

    def search_by_knn(self, vector: List[float], index_name: str, top_n: int = 80) -> List[Dict]:
        query = {
            "size": top_n,
            "_source": ["content", "metadata"],
            "query": {
                "knn": {
                    "content_embedding": {
                        "vector": vector,
                        "k": top_n
                    }
                }
            }
        }

        try:
            response = self.opensearch_client.search(index=f"{index_name}", body=query)
            return [self._format_search_result(hit, 'knn') 
                   for hit in response['hits']['hits']]
        except Exception as e:
            logger.error(f"KNN search error: {e}")
            return []

    def search_by_bm25(self, query_text: str, index_name: str, top_n: int = 80) -> List[Dict]:
        query = {
            "size": top_n,
            "_source": ["content", "metadata"],
            "query": {
                "match": {
                    "content": {
                        "query": query_text,
                        "operator": "or"
                    }
                }
            }
        }

        try:
            response = self.opensearch_client.search(index=f"{index_name}", body=query)
            return [self._format_search_result(hit, 'bm25') 
                   for hit in response['hits']['hits']]
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return []

    def _init_opensearch_client(self, aws_region: str, aws_profile: str, domain_name: str = None, user: None = None, password: None = None) -> OpenSearch:
        session = boto3.Session(region_name=aws_region, profile_name=aws_profile)
        os_client = session.client('opensearch')
        
        # get domain name and host
        if not domain_name:
            domain_name = [domain['DomainName'] for domain in os_client.list_domain_names().get('DomainNames')][0]
        host = os_client.describe_domain(DomainName=domain_name)['DomainStatus']['Endpoint']
        
        # get credentials
        if not user and not password:
            credentials = session.get_credentials()
            auth = AWS4Auth(
                credentials.access_key,
                credentials.secret_key,
                aws_region,
                "es",
                session_token=credentials.token
            )
        else:
            auth = (user, password)
        
        return OpenSearch(
            hosts=[{'host': host, 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300
        )
    
    @staticmethod
    def _format_search_result(hit: Dict, search_method: str) -> Dict:
        return {
            "content": hit['_source']["content"],
            "score": hit['_score'],
            "metadata": hit['_source']['metadata'],
            "search_method": search_method
        }
