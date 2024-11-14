import os
import json
import yaml
from copy import deepcopy
from typing import List, Optional, Dict, Tuple
import streamlit as st
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain.schema import BaseRetriever, Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.vectorstores import OpenSearchVectorSearch
from .common_utils import sample_query_indexing, schema_desc_indexing
from .ssm import parameter_store

class OpenSearchClient:
    def __init__(self, emb, index_name, mapping_name, vector, text, output):
        pm = parameter_store('us-west-2')
        config = self.load_opensearch_config()
        self.index_name = index_name
        self.emb = emb
        self.config = config
        self.endpoint = pm.get_params(key="chatbot-opensearch_domain_endpoint", enc=False)
        #self.endpoint = f"{domain_endpoint}"
        self.http_auth = (pm.get_params(key="chatbot-opensearch_user_id", enc=False), pm.get_params(key="chatbot-opensearch_user_password", enc=True))
        self.vector = vector
        self.text = text
        self.output = output
        self.mapping = {"settings": config['settings'], "mappings": config[mapping_name]}
        self.conn = OpenSearch(
            hosts=[{'host': self.endpoint.replace("https://", ""), 'port': 443}],
            http_auth=self.http_auth, 
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        ) 
        self.vector_store = OpenSearchVectorSearch(
            index_name=self.index_name,
            opensearch_url=self.endpoint,
            embedding_function=self.emb,
            http_auth=self.http_auth,
        )
        
    def load_opensearch_config(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(file_dir, "opensearch.yml")

        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def is_index_present(self):
        return self.conn.indices.exists(self.index_name)
    
    def create_index(self):
        self.conn.indices.create(self.index_name, body=self.mapping)

    def delete_index(self):
        if self.is_index_present():
            self.conn.indices.delete(self.index_name)

class OpenSearchHybridRetriever(BaseRetriever):
    os_client: OpenSearchClient
    k: int = 5
    verbose: bool = True
    filter: List[dict] = []

    def __init__(self, os_client: OpenSearchClient, k):
        super().__init__(os_client=os_client)
        self.os_client = os_client
        self.k = k
    
    def _get_relevant_documents(self, query: str, *, ensemble: List, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        os_client = self.os_client
        search_result = retriever_utils.search_hybrid(
            query=query,
            k=self.k,
            filter=self.filter,
            index_name=os_client.index_name,
            os_conn=os_client.conn,
            emb=os_client.emb,
            ensemble_weights=ensemble,
            vector_field=os_client.vector,
            text_field=os_client.text,
            output_field=os_client.output
        )
        return search_result

class retriever_utils():

    @classmethod 
    def normalize_search_results(cls, search_results):
        hits = (search_results["hits"]["hits"])
        max_score = float(search_results["hits"]["max_score"])
        for hit in hits:
            hit["_score"] = float(hit["_score"]) / max_score
        search_results["hits"]["max_score"] = hits[0]["_score"]
        search_results["hits"]["hits"] = hits
        return search_results
    
    @classmethod 
    def search_semantic(cls, **kwargs):
        semantic_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                kwargs["vector_field"]: {
                                    "vector": kwargs["emb"].embed_query(kwargs["query"]),
                                    "k": kwargs["k"],
                                }
                            }
                        },
                    ],
                    "filter": kwargs.get("boolean_filter", []),
                }
            },
            "size": kwargs["k"],
            #"min_score": 0.3
        }
        # get semantic search results
        search_results = lookup_opensearch_document(
            index_name=kwargs["index_name"],
            os_conn=kwargs["os_conn"],
            query=semantic_query,
        )

        results = []
        if search_results.get("hits", {}).get("hits", []):
            # normalize the scores
            search_results = cls.normalize_search_results(search_results)
            for res in search_results["hits"]["hits"]:
                if "metadata" in res["_source"]:
                    metadata = res["_source"]["metadata"]
                else:
                    metadata = {}
                metadata["id"] = res["_id"]

                # extract the text contents
                page_content = json.dumps({field: res["_source"].get(field, "") for field in kwargs["output_field"]})
                doc = Document(
                    page_content=page_content,
                    metadata=metadata
                )
                results.append((doc, res["_score"]))
        return results

    @classmethod
    def search_lexical(cls, **kwargs):
        lexical_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                kwargs["text_field"]: {
                                    "query": kwargs["query"],
                                    "operator": "or",
                                }
                            }
                        },
                    ],
                    "filter": kwargs["filter"]
                }
            },
            "size": kwargs["k"] 
        }

        # get lexical search results
        search_results = lookup_opensearch_document(
            index_name=kwargs["index_name"],
            os_conn=kwargs["os_conn"],
            query=lexical_query,
        )

        results = []
        if search_results.get("hits", {}).get("hits", []):
            # normalize the scores
            search_results = cls.normalize_search_results(search_results)
            for res in search_results["hits"]["hits"]:
                if "metadata" in res["_source"]:
                    metadata = res["_source"]["metadata"]
                else:
                    metadata = {}
                metadata["id"] = res["_id"]

                # extract the text contents
                page_content = json.dumps({field: res["_source"].get(field, "") for field in kwargs["output_field"]})
                doc = Document(
                    page_content=page_content,
                    metadata=metadata
                )
                results.append((doc, res["_score"]))
        return results

    @classmethod
    def search_hybrid(cls, **kwargs):

        assert "query" in kwargs, "Check your query"
        assert "emb" in kwargs, "Check your emb"
        assert "index_name" in kwargs, "Check your index_name"
        assert "os_conn" in kwargs, "Check your OpenSearch Connection"

        search_filter = deepcopy(kwargs.get("filter", []))
        similar_docs_semantic = cls.search_semantic(
                index_name=kwargs["index_name"],
                os_conn=kwargs["os_conn"],
                emb=kwargs["emb"],
                query=kwargs["query"],
                k=kwargs.get("k", 5),
                vector_field=kwargs["vector_field"],
                output_field=kwargs["output_field"],
                boolean_filter=search_filter,
            )
        # print("semantic_docs:", similar_docs_semantic)

        similar_docs_lexical = cls.search_lexical(
                index_name=kwargs["index_name"],
                os_conn=kwargs["os_conn"],
                query=kwargs["query"],
                k=kwargs.get("k", 5),
                text_field=kwargs["text_field"],
                output_field=kwargs["output_field"],
                minimum_should_match=kwargs.get("minimum_should_match", 1),
                filter=search_filter,
            )
        # print("lexical_docs:", similar_docs_lexical)

        similar_docs = retriever_utils.get_ensemble_results(
            doc_lists=[similar_docs_semantic, similar_docs_lexical],
            weights=kwargs.get("ensemble_weights", [.51, .49]),
            k=kwargs.get("k", 5),
        )
        
        return similar_docs        


    @classmethod
    def get_ensemble_results(cls, doc_lists: List[List[Tuple[Document, float]]], weights: List[float], k: int = 5) -> List[Document]:
        hybrid_score_dic: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        
        # Weight-based adjustment
        for doc_list, weight in zip(doc_lists, weights):
            for doc, score in doc_list:
                doc_id = doc.metadata.get("id", doc.page_content)
                if doc_id not in hybrid_score_dic:
                    hybrid_score_dic[doc_id] = 0.0
                hybrid_score_dic[doc_id] += score * weight
                doc_map[doc_id] = doc

        sorted_docs = sorted(hybrid_score_dic.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in sorted_docs[:k]]



def lookup_opensearch_document(index_name, os_conn, query):
    response = os_conn.search(
        index=index_name,
        body=query
    )
    return response

def initialize_os_client(enable_flag: bool, client_params: Dict, indexing_function, lang_config: Dict):
    if enable_flag:
        client = OpenSearchClient(**client_params)
        indexing_function(client, lang_config)
    else:
        client = ""
    return client

def init_opensearch(emb_model, lang_config):
    with st.sidebar:
        enable_rag_query = st.sidebar.checkbox(lang_config['rag_query'], value=True, disabled=True)
        sql_os_client = initialize_os_client(
            enable_rag_query,
            {
                "emb": emb_model,
                "index_name": 'example_queries',
                "mapping_name": 'mappings-sql',
                "vector": "input_v",
                "text": "input",
                "output": ["input", "query"]
            },
            sample_query_indexing,
            lang_config
        )

        enable_schema_desc = st.sidebar.checkbox(lang_config['schema_desc'], value=True, disabled=True)
        schema_os_client = initialize_os_client(
            enable_schema_desc,
            {
                "emb": emb_model,
                "index_name": 'schema_descriptions',
                "mapping_name": 'mappings-detailed-schema',
                "vector": "table_summary_v",
                "text": "table_summary",
                "output": ["table_name", "table_summary"]
            },
            schema_desc_indexing,
            lang_config
        )

    return sql_os_client, schema_os_client