############################################################    
############################################################    
# RAG 관련 함수들
############################################################    
############################################################    

import json
import copy
import numpy as np
import pandas as pd
from pprint import pprint
from operator import itemgetter
from itertools import chain as ch
from typing import Any, Dict, List, Optional, List, Tuple

from opensearchpy import OpenSearch, RequestsHttpConnection

from local_utils import print_ww
from local_utils.opensearch import opensearch_utils

from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from langchain.retrievers import AmazonKendraRetriever
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from local_utils.bedrock import bedrock_info

############################################################
################## Added Lib ############################
############################################################


############################################################
# get_embedding_model 함수
############################################################

def get_embedding_model(is_bedrock_embeddings, is_KoSimCSERobert, boto3_bedrock, aws_region, endpont_name=None):

    if is_bedrock_embeddings:
        # We will be using the Titan Embeddings Model to generate our Embeddings.
        from langchain.embeddings import BedrockEmbeddings

        llm_emb = BedrockEmbeddings(
            client=boto3_bedrock,
            model_id=bedrock_info.get_model_id(
                model_name="Titan-Embeddings-G1"
            )
        )
        print("Bedrock Embeddings Model Loaded")
        
    elif is_KoSimCSERobert:
        LLMEmbHandler = KoSimCSERobertaContentHandler()
        endpoint_name_emb = endpont_name
        llm_emb = SagemakerEndpointEmbeddingsJumpStart(
            endpoint_name=endpoint_name_emb,
            region_name=aws_region,
            content_handler=LLMEmbHandler,
        )        
        print("KoSimCSERobert Embeddings Model Loaded")
    else:
        llm_emb = None
        print("No Embedding Model Selected")
    
    return llm_emb   

from multiprocessing.pool import ThreadPool
from functools import partial
import json
import boto3
from local_utils.reranker import RunReranker

def evaluate_search_methods(**kwargs):
    '''
    검증 데이터 셋에 대해서 검색방법을 평가 합니다.
    '''
    assert "output_file_path" in kwargs, "Check your output file path"
    assert "parallel" in kwargs, "Check your parallel way"    
    assert "search_method" in kwargs, "Check your search method"
    assert "eval_df" in kwargs, "Check your evaluation pandas dataframe"
    assert "vector_db" in kwargs, "Check langchain opensearch connector"
    assert "single_num_result" in kwargs, "Check # of results for lexical and semanitc"
    assert "hybrid_num_result" in kwargs, "Check # of results for hybrid"
    assert "index_name" in kwargs, "Check opensearch index name"    
    assert "os_client" in kwargs, "Check opensearch python client"        
    assert "minimum_should_match" in kwargs, "Check minimum should match parameter"        
    assert "ensemble_weights" in kwargs, "Check ensemble weights"            


    output_file_path = kwargs.get("output_file_path", None)
    parallel = kwargs.get("parallel", True)
    num_process = kwargs.get("num_process", 10)
    num_eval = kwargs.get("num_eval", 1)    
    search_method = kwargs.get("search_method", None)
    eval_df = kwargs.get("eval_df", None)
    local_eval_df = eval_df.copy()
    verbose = kwargs.get("verbose", False)
    vector_db = kwargs.get("vector_db", None)
    single_num_result = kwargs.get("single_num_result", 3)
    hybrid_num_result = kwargs.get("hybrid_num_result", 1)    
    index_name = kwargs.get("index_name", None)        
    os_client = kwargs.get("os_client", None)        
    minimum_should_match = kwargs.get("minimum_should_match", 50)            
    ensemble_weights = kwargs.get("ensemble_weights", [.5, .5])                
    reranker_endpoint_name = kwargs.get("reranker_endpoint_name", None)                    

    if search_method == "lexical":
        is_lexical = True
        is_semantic = False
        is_hybrid = False
    elif search_method == "semantic":
        is_lexical = False
        is_semantic = True
        is_hybrid = False
    elif search_method == "hybrid":
        is_lexical = False
        is_semantic = False
        is_hybrid = True
    elif search_method == "reranker":
        assert "reranker_endpoint_name" in kwargs, "Check reranker_endpoint_name"            
        is_lexical = False
        is_semantic = False
        is_hybrid = True
    else:
        print("#### pass ###")
        pass

    runtime_client = boto3.Session().client('sagemaker-runtime')

    print("##############################")
    print("parallel way: ", parallel)
    print("search method: ", search_method)
    print("##############################")    

    def do_sync():
        '''
        싱글 쓰레드로 시행
        '''
        for idx,qry in enumerate (local_eval_df['input']):

            search_hybrid_result = multiple_search(
                query=qry,
                vector_db= vector_db,
                k = single_num_result,
                h_k = hybrid_num_result, 
                index_name= index_name,
                os_client=os_client,
                filter=[
                    {"term": {"metadata.type": "[]"}},
                    {"term": {"metadata.source": "[]"}},
                ],
                Semantic_Search = is_semantic,    
                Lexical_Search = is_lexical,    
                Hybrid_Search = is_hybrid,     
                minimum_should_match = minimum_should_match,   
                fusion_algorithm="RRF", # ["RRF", "simple_weighted"]
                ensemble_weights= ensemble_weights, # 시멘트 서치에 가중치 0.5 , 키워드 서치 가중치 0.5 부여.
                verbose=False
            )    

            if verbose:
                print("query: ", qry)
                print("raw search result: \n", search_hybrid_result)

            if search_method == "lexical":
                response = search_hybrid_result[0][0].metadata['intent']
            elif search_method == "semantic":
                response = search_hybrid_result[0][0].metadata['intent']
            elif search_method == "hybrid":
                response = search_hybrid_result[0][0].metadata['intent']
            elif search_method == "reranker":
                # response = search_hybrid_result[0][0].metadata['intent']
                intent = RunReranker(query=qry, query_pair=search_hybrid_result, 
                        boto3_client=runtime_client, 
                        endpoint_name=reranker_endpoint_name, 
                        verbose= verbose)    
                response = intent.highest_intent      

            if verbose:
                print("intent: ", response)

            try:
                local_eval_df.loc[idx,'result'] = response
            except:
                local_eval_df.loc[idx,'result'] = "None"

            if idx == (num_eval-1):
                break

        local_eval_df.to_csv(output_file_path)
        print(f"{output_file_path} is saved")


    def get_multiple_search(idx, qry):
        search_hybrid_result = multiple_search(
            query=qry,
            vector_db= vector_db,
            k = single_num_result,
            h_k = hybrid_num_result, 
            index_name= index_name,
            os_client=os_client,
            filter=[
                {"term": {"metadata.type": "[]"}},
                {"term": {"metadata.source": "[]"}},
            ],
            Semantic_Search = is_semantic,    
            Lexical_Search = is_lexical,    
            Hybrid_Search = is_hybrid,     
            minimum_should_match = minimum_should_match,   
            fusion_algorithm="RRF", # ["RRF", "simple_weighted"]
            ensemble_weights= ensemble_weights, # 시멘트 서치에 가중치 0.5 , 키워드 서치 가중치 0.5 부여.
            verbose= False
        )

        if verbose:
            print("query: ", qry)
            print("raw search result: \n", search_hybrid_result)

        try:
            if search_method == "lexical":
                response = search_hybrid_result[0][0].metadata['intent']
            elif search_method == "semantic":
                response = search_hybrid_result[0][0].metadata['intent']
            elif search_method == "hybrid":
                #response = search_hybrid_result[0][0].metadata['intent']
                response = search_hybrid_result[0].metadata['intent']
            elif search_method == "reranker":
                #response = search_hybrid_result[0][0].metadata['intent']
                intent = RunReranker(query=qry, query_pair=search_hybrid_result, 
                        boto3_client=runtime_client, 
                        endpoint_name=reranker_endpoint_name, 
                        verbose= False)    
                response = intent.highest_intent      
                
        except:
            response = "No Result"
            

        if verbose:
            print("intent: ", response)
        return response
            


    def do_async(num_process):
        '''
        멀티 쓰레드로 시행
        '''
        ThreadPool()
        pool = ThreadPool(processes= num_process)
        partial_get_multiple_search = partial(get_multiple_search)

        results = []
        for idx,qry in enumerate (local_eval_df['input']):

            async_result = pool.apply_async(partial_get_multiple_search, args=(idx, qry))
            results.append(async_result)

            if idx == (num_eval-1):
                break

        pool.close()
        pool.join()

        intent_list = []
        for result in results:
            # print("result_get: ", result.get())
            intent_list.append(result.get())
        
        return intent_list

    if parallel:
        intent_list = do_async(num_process)
        num_rows = local_eval_df.shape[0]
        if len(intent_list) < num_rows: # 주어진 데이터 프레임보다 개수가 작으면 None 로 채운다
            intent_list.extend([None] * (num_rows - len(intent_list)) )
        local_eval_df.insert(len(eval_df.columns), "result", intent_list)
        local_eval_df.to_csv(output_file_path)
        print(f"{output_file_path} is saved")
    else:
        do_sync()

def multiple_search(**kwargs):
    
    assert "query" in kwargs, "Check your query"
    assert "vector_db" in kwargs, "Check your vector_db"
    assert "index_name" in kwargs, "Check your index_name"
    assert "os_client" in kwargs, "Check your os_client"
    assert "Semantic_Search" in kwargs, "Check your Semantic_Search"    
    assert "Lexical_Search" in kwargs, "Check your Lexical_Search"        
    assert "Hybrid_Search" in kwargs, "Check your Hybrid_Search"            
    assert "minimum_should_match" in kwargs, "Check your minimum_should_match"                
    
    query = kwargs["query"]
    index_name = kwargs["index_name"]
    os_client = kwargs["os_client"]
    vector_db = kwargs["vector_db"]
    k = kwargs.get("k", False)
    h_k = kwargs.get("h_k", 1)
    minimum_should_match = kwargs.get("minimum_should_match", 50)
    ensemble_weights = kwargs.get("ensemble_weights", [.49, .51])  
    fusion_algorithm = kwargs.get("fusion_algorithm", "RRF") # ["RRF", "simple_weighted"]    
    verbose = kwargs.get("verbose", False)
    
    if verbose:
        print("Lexical_Search: ", kwargs["Lexical_Search"])        
        print("Semantic_Search: ", kwargs["Semantic_Search"])
        print("Hybrid_Search: ", kwargs["Hybrid_Search"])    
        print("Query: \n", kwargs["query"])        
    
    if (kwargs["Lexical_Search"] == True):
        similar_docs_keyword = get_lexical_similar_docs(
            query= query,
            minimum_should_match= minimum_should_match,
            filter= [],
            index_name= index_name,
            os_client= os_client,
            k= k, 
            hybrid=True,
            verbose = verbose,
        )
        if verbose:
            print("##############################")    
            print("Lexcial_Search")    
            print("##############################")    
            # print(similar_docs_keyword)        
            opensearch_pretty_print_documents_with_score(similar_docs_keyword)            

        
        return similar_docs_keyword

    
    if (kwargs["Semantic_Search"] == True):
        similar_docs_semantic = get_semantic_similar_docs(
            vector_db= vector_db,
            query= query,
            k= k, 
            hybrid=True
        )
        if verbose:
            print("##############################")    
            print("Semantic_Search")    
            print("##############################")    
            # print(similar_docs_keyword)        
            opensearch_pretty_print_documents_with_score(similar_docs_semantic)            
        
        return similar_docs_semantic
        

    if kwargs["Hybrid_Search"] == True:
        similar_docs_keyword = get_lexical_similar_docs(
            query= query,
            minimum_should_match= minimum_should_match,
            filter= [],
            index_name= index_name,
            os_client= os_client,
            k= k, 
            hybrid=True
        )
        if verbose:
            print("##############################")    
            print("Lexcial_Search")    
            print("##############################")    
            # print(similar_docs_keyword)        
            opensearch_pretty_print_documents_with_score(similar_docs_keyword)            


        similar_docs_semantic = get_semantic_similar_docs(
            vector_db= vector_db,
            query= query,
            k= k, 
            hybrid=True
        )
        if verbose:
            print("##############################")    
            print("Semantic_Search")    
            print("##############################")    
            # print(similar_docs_keyword)        
            opensearch_pretty_print_documents_with_score(similar_docs_semantic)            

        similar_docs_ensemble = get_ensemble_results(
            doc_lists = [similar_docs_semantic, similar_docs_keyword],
            weights = ensemble_weights,
            algorithm= fusion_algorithm,
            c=60,
            k= h_k
        )
        if verbose:
            print("##############################")    
            print("Ensemble_Search")    
            print("##############################")    
            # print(similar_docs_keyword)        
            opensearch_pretty_print_documents_with_score(similar_docs_ensemble)            

    
        similar_docs_ensemble = list(map(lambda x:x[0], similar_docs_ensemble))
    
        return similar_docs_ensemble

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


############################################################
################## Existing Lib ############################
############################################################

############################################################
# RetrievalQA (Langchain)
############################################################

def run_RetrievalQA(**kwargs):

    chain_types = ["stuff", "map_reduce", "refine"]

    assert "llm" in kwargs, "Check your llm"
    assert "query" in kwargs, "Check your query"
    assert "prompt" in kwargs, "Check your prompt"
    assert "vector_db" in kwargs, "Check your vector_db"
    assert kwargs.get("chain_type", "stuff") in chain_types, f'Check your chain_type, {chain_types}'

    qa = RetrievalQA.from_chain_type(
        llm=kwargs["llm"],
        chain_type=kwargs.get("chain_type", "stuff"),
        retriever=kwargs["vector_db"].as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": kwargs.get("k", 5),
                "boolean_filter": kwargs.get("boolean_filter", {})
            }
        ),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": kwargs["prompt"],
            "verbose": kwargs.get("verbose", False),
        },
        verbose=kwargs.get("verbose", False)
    )

    return qa(kwargs["query"])

def run_RetrievalQA_kendra(query, llm_text, PROMPT, kendra_index_id, k, aws_region, verbose):
    qa = RetrievalQA.from_chain_type(
        llm=llm_text,
        chain_type="stuff",
        retriever=AmazonKendraRetriever(
            index_id=kendra_index_id,
            region_name=aws_region,
            top_k=k,
            attribute_filter = {
                "EqualsTo": {      
                    "Key": "_language_code",
                    "Value": {
                        "StringValue": "ko"
                    }
                },
            }
        ),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PROMPT,
            "verbose": verbose,
        },
        verbose=verbose
    )

    result = qa(query)

    return result

#################################################################
# Document Retriever with custom function: return List(documents)
#################################################################

# semantic search based
def get_semantic_similar_docs(**kwargs):

    search_types = ["approximate_search", "script_scoring", "painless_scripting"]
    space_types = ["l2", "l1", "linf", "cosinesimil", "innerproduct", "hammingbit"]

    assert "vector_db" in kwargs, "Check your vector_db"
    assert "query" in kwargs, "Check your query"
    assert kwargs.get("search_type", "approximate_search") in search_types, f'Check your search_type: {search_types}'
    assert kwargs.get("space_type", "l2") in space_types, f'Check your space_type: {space_types}'

    results = kwargs["vector_db"].similarity_search_with_score(
            query=kwargs["query"],
            k=kwargs.get("k", 5),
            search_type=kwargs.get("search_type", "approximate_search"),
            space_type=kwargs.get("space_type", "l2"),
            boolean_filter=kwargs.get("boolean_filter", {}),
            # fetch_k=3,
        )

    if kwargs.get("hybrid", False):
        max_score = results[0][1]
        new_results = []
        for doc in results:
            nomalized_score = float(doc[1]/max_score)
            new_results.append((doc[0], nomalized_score))
        results = copy.deepcopy(new_results)

    return results

# lexical(keyword) search based (using Amazon OpenSearch)
def get_lexical_similar_docs(**kwargs):

    assert "query" in kwargs, "Check your query"
    assert "k" in kwargs, "Check your k"
    assert "os_client" in kwargs, "Check your os_client"
    assert "index_name" in kwargs, "Check your index_name"

    verbose = kwargs.get("verbose", False)

    def normalize_search_results(search_results):

        hits = (search_results["hits"]["hits"])
        max_score = float(search_results["hits"]["max_score"])
        for hit in hits:
            hit["_score"] = float(hit["_score"]) / max_score
        search_results["hits"]["max_score"] = hits[0]["_score"]
        search_results["hits"]["hits"] = hits
        return search_results

    query = opensearch_utils.get_query(
        query=kwargs["query"],
        minimum_should_match=kwargs.get("minimum_should_match", 0),
        filter=kwargs.get("filter", [])
    )
    query["size"] = kwargs["k"]

    if verbose:
        print ("lexical search query: ")
        pprint (query)

    search_results = opensearch_utils.search_document(
        os_client=kwargs["os_client"],
        query=query,
        index_name=kwargs["index_name"]
    )

    results = []
    if search_results["hits"]["hits"]:
        search_results = normalize_search_results(search_results)
        for res in search_results["hits"]["hits"]:

            metadata = res["_source"]["metadata"]
            #metadata["score"] = res["_score"]
            metadata["id"] = res["_id"]

            doc = Document(
                page_content=res["_source"]["text"],
                metadata=metadata
            )
            if kwargs.get("hybrid", False):
                results.append((doc, res["_score"]))
            else:
                results.append((doc))

    return results

# hybrid (lexical + semantic) search based
# def search_hybrid(**kwargs):

#     assert "query" in kwargs, "Check your query"
#     assert "vector_db" in kwargs, "Check your vector_db"
#     assert "index_name" in kwargs, "Check your index_name"
#     assert "os_client" in kwargs, "Check your os_client"

#     verbose = kwargs.get("verbose", False)

#     similar_docs_semantic = get_semantic_similar_docs(
#         vector_db=kwargs["vector_db"],
#         query=kwargs["query"],
#         k=kwargs.get("k", 5),
#         hybrid=True
#     )
#     similar_docs_keyword = get_lexical_similar_docs(
#         query=kwargs["query"],
#         minimum_should_match=kwargs.get("minimum_should_match", 0),
#         filter=kwargs.get("filter", []),
#         index_name=kwargs["index_name"],
#         os_client=kwargs["os_client"],
#         k=kwargs.get("k", 5),
#         hybrid=True
#     )

#     similar_docs_ensemble = get_ensemble_results(
#         doc_lists=[similar_docs_semantic, similar_docs_keyword],
#         weights=kwargs.get("ensemble_weights", [.5, .5]),
#         algorithm=kwargs.get("fusion_algorithm", "RRF"), # ["RRF", "simple_weighted"]
#         c=60,
#         k=kwargs.get("k", 5)
#     )

#     if verbose:
#         print("##############################")
#         print("similar_docs_semantic")
#         print("##############################")
#         print(similar_docs_semantic)

#         print("##############################")
#         print("similar_docs_keyword")
#         print("##############################")
#         print(similar_docs_keyword)

#         print("##############################")
#         print("similar_docs_ensemble")
#         print("##############################")
#         print(similar_docs_ensemble)

#     similar_docs_ensemble = list(map(lambda x:x[0], similar_docs_ensemble))

#     return similar_docs_ensemble

# Score fusion and re-rank (lexical + semantic)
def get_ensemble_results(doc_lists: List[List[Document]], weights, algorithm="RRF", c=60, k=5) -> List[Document]:

    assert algorithm in ["RRF", "simple_weighted"]

    # print("########### get_ensemble_results ################")
    # print("########### algorithm ################", algorithm)    
    # Create a union of all unique documents in the input doc_lists
    all_documents = set()

    for doc_list in doc_lists:
        for (doc, _) in doc_list:
            all_documents.add(doc.page_content)

    # Initialize the score dictionary for each document
    hybrid_score_dic = {doc: 0.0 for doc in all_documents}

    # Calculate RRF scores for each document
    for doc_list, weight in zip(doc_lists, weights):
        for rank, (doc, score) in enumerate(doc_list, start=1):
            if algorithm == "RRF": # RRF (Reciprocal Rank Fusion)
                score = weight * (1 / (rank + c))
                # print("rank, doc, score: ", rank, doc, score )                
                # print("weight: ", weight )
                # print("score: ", score )                
            elif algorithm == "simple_weighted":
                score *= weight
            hybrid_score_dic[doc.page_content] += score

    # Sort documents by their scores in descending order
    sorted_documents = sorted(
        hybrid_score_dic.items(), key=lambda x: x[1], reverse=True
    )

    # print("sorted_documents: \n", sorted_documents)
    # Map the sorted page_content back to the original document objects
    page_content_to_doc_map = {
        doc.page_content: doc for doc_list in doc_lists for (doc, orig_score) in doc_list
    }

    # print("page_content_to_doc_map: \n", page_content_to_doc_map)
    sorted_docs = [
        (page_content_to_doc_map[page_content], hybrid_score) for (page_content, hybrid_score) in sorted_documents
    ]

    return sorted_docs[:k]

#################################################################
# Document Retriever with Langchain(BaseRetriever): return List(documents)
#################################################################

# lexical(keyword) search based (using Amazon OpenSearch)
class OpenSearchLexicalSearchRetriever(BaseRetriever):

    os_client: Any
    index_name: str
    k = 3
    minimum_should_match = 0
    filter = []

    def normalize_search_results(self, search_results):

        hits = (search_results["hits"]["hits"])
        max_score = float(search_results["hits"]["max_score"])
        for hit in hits:
            hit["_score"] = float(hit["_score"]) / max_score
        search_results["hits"]["max_score"] = hits[0]["_score"]
        search_results["hits"]["hits"] = hits
        return search_results

    def update_search_params(self, **kwargs):

        self.k = kwargs.get("k", 3)
        self.minimum_should_match = kwargs.get("minimum_should_match", 0)
        self.filter = kwargs.get("filter", [])
        self.index_name = kwargs.get("index_name", self.index_name)

    def _reset_search_params(self, ):

        self.k = 3
        self.minimum_should_match = 0
        self.filter = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:

        query = opensearch_utils.get_query(
            query=query,
            minimum_should_match=self.minimum_should_match,
            filter=self.filter
        )
        query["size"] = self.k

        print ("lexical search query: ")
        pprint(query)

        search_results = opensearch_utils.search_document(
            os_client=self.os_client,
            query=query,
            index_name=self.index_name
        )

        results = []
        if search_results["hits"]["hits"]:
            search_results = self.normalize_search_results(search_results)
            for res in search_results["hits"]["hits"]:

                metadata = res["_source"]["metadata"]
                metadata["id"] = res["_id"]

                doc = Document(
                    page_content=res["_source"]["text"],
                    metadata=metadata
                )
                results.append((doc))

        self._reset_search_params()

        return results[:self.k]

# hybrid (lexical + semantic) search based
class OpenSearchHybridSearchRetriever(BaseRetriever):

    os_client: Any
    vector_db: Any
    index_name: str
    k = 3
    minimum_should_match = 0
    filter = []
    fusion_algorithm: str
    ensemble_weights: List
    verbose = False

    def update_search_params(self, **kwargs):

        self.k = kwargs.get("k", 3)
        self.minimum_should_match = kwargs.get("minimum_should_match", 0)
        self.filter = kwargs.get("filter", [])
        self.index_name = kwargs.get("index_name", self.index_name)
        self.fusion_algorithm = kwargs.get("fusion_algorithm", self.fusion_algorithm)
        self.ensemble_weights = kwargs.get("ensemble_weights", self.ensemble_weights)
        self.verbose = kwargs.get("verbose", self.verbose)

    def _reset_search_params(self, ):

        self.k = 3
        self.minimum_should_match = 0
        self.filter = []

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:

        search_hybrid_result = search_hybrid(
            query=query,
            vector_db=self.vector_db,
            k=self.k,
            index_name=self.index_name,
            os_client=self.os_client,
            filter=self.filter,
            fusion_algorithm=self.fusion_algorithm, # ["RRF", "simple_weighted"]
            ensemble_weights=self.ensemble_weights, # 시멘트 서치에 가중치 0.5 , 키워드 서치 가중치 0.5 부여.
            verbose=self.verbose
        )

        #self._reset_search_params()

        return search_hybrid_result

#################################################################
# Document visualization
#################################################################

def show_context_used(context_list, limit=10):

    for idx, context in enumerate(context_list):
        if idx < limit:
            print("-----------------------------------------------")
            print(f"{idx+1}. Chunk: {len(context.page_content)} Characters")
            print("-----------------------------------------------")
            print_ww(context.page_content)
            print_ww("metadata: \n", context.metadata)
        else:
            break

def show_chunk_stat(documents):

    doc_len_list = [len(doc.page_content) for doc in documents]
    print(pd.DataFrame(doc_len_list).describe())
    avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents])//len(documents)
    avg_char_count_pre = avg_doc_length(documents)
    print(f'Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters.')

    max_idx = doc_len_list.index(max(doc_len_list))
    print("\nShow document at maximum size")
    print(documents[max_idx].page_content)




    
    
    
    
    
    
    
    

    
    
    
    



###########################################    
### 1.2 [한국어 임베딩벡터 모델] SageMaker 임베딩 벡터 모델 KoSimCSE-roberta endpiont handler
###########################################    

### SagemakerEndpointEmbeddingsJumpStart클래스는 SagemakerEndpointEmbeddings를 상속받아서 작성

# 매개변수 (Parameters):
# * texts: 임베딩을 생성할 텍스트의 리스트입니다.
# * chunk_size: 한 번의 요청에 그룹화될 입력 텍스트의 수를 정의합니다. 만약 None이면, 클래스에 지정된 청크 크기를 사용합니다.

# Returns:
# * 각 텍스트에 대한 임베딩의 리스트를 반환
###########################################    



class SagemakerEndpointEmbeddingsJumpStart(SagemakerEndpointEmbeddings):
    def embed_documents(self, texts: List[str], chunk_size: int=1) -> List[List[float]]:
        """Compute doc embeddings using a SageMaker Inference Endpoint.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size defines how many input texts will
                be grouped together as request. If None, will use the
                chunk size specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        results = []
        _chunk_size = len(texts) if chunk_size > len(texts) else chunk_size
        
        print("text size: ", len(texts))
        print("_chunk_size: ", _chunk_size)

        for i in range(0, len(texts), _chunk_size):
            
            #print (i, texts[i : i + _chunk_size])
            response = self._embedding_func(texts[i : i + _chunk_size])
            #print (i, response, len(response[0].shape))
            
            results.extend(response)
        return results    
    
class KoSimCSERobertaContentHandler(EmbeddingsContentHandler):
    
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
        
        input_str = json.dumps({"inputs": prompt, **model_kwargs})
        
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        
        response_json = json.loads(output.read().decode("utf-8"))
        ndim = np.array(response_json).ndim    
        
        if ndim == 4:
            # Original shape (1, 1, n, 768)
            emb = response_json[0][0][0]
            emb = np.expand_dims(emb, axis=0).tolist()
        elif ndim == 2:
            # Original shape (n, 1)
            emb = []
            for ele in response_json:
                e = ele[0][0]
                emb.append(e)
        else:
            print(f"Other # of dimension: {ndim}")
            emb = None
        return emb    
    
    
def opensearch_pretty_print_documents(response):
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
    
def opensearch_pretty_print_documents_wo_filter(response):
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
                
        print('-' * 50)

    
############################################################    
# OpenSearch Client
############################################################    
    


def create_aws_opensearch_client(region: str, host: str, http_auth: Tuple[str, str]) -> OpenSearch:
    '''
    오픈서치 클라이언트를 제공함.
    '''
    aws_client = OpenSearch(
        hosts = [{'host': host.replace("https://", ""), 'port': 443}],
        http_auth = http_auth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )
    
    return aws_client

def create_index(aws_client, index_name, index_body):    
    '''
    인덱스 생성
    '''
    response = aws_client.indices.create(
        index_name,
        body=index_body
    )
    print('\nCreating index:')
    print(response)

    
def check_if_index_exists(aws_client, index_name):
    '''
    인덱스가 존재하는지 확인
    '''
    exists = aws_client.indices.exists(index_name)
    print(f"index_name={index_name}, exists={exists}")
    return exists


def add_doc(aws_client, index_name, document, id):
    '''
    # Add a document to the index.
    '''
    response = aws_client.index(
        index = index_name,
        body = document,
        id = id,
        refresh = True
    )

    print('\nAdding document:')
    print(response)

def search_document(aws_client, query, index_name):
    response = aws_client.search(
        body=query,
        index=index_name
    )
    print('\nSearch results:')
    # print(response)
    return response
    

def delete_index(aws_client, index_name):
    response = aws_client.indices.delete(
        index = index_name
    )

    print('\nDeleting index:')
    print(response)



def generate_opensearch_AndQuery(question):
    '''
    주어진 앱력을 키워드로 분리하고 AND 조건으로 바꾸어 주는 쿼리 생성
    '''
    keywords = question.split(' ')
    query = {
        "query": {
            "bool": {
                "must": []
            }
        }
    }
    
    for keyword in keywords:
        query["query"]["bool"]["must"].append({
            "match": {
                "text": keyword
            }
        })
    
    # return query
    return json.dumps(query, indent=2, ensure_ascii=False)


# def parse_keyword_response(response, show_size=3):
#     '''
#     키워드 검색 결과를 보여 줌.
#     '''
#     length = len(response['hits']['hits'])
#     if length >= 1:
#         print("# of searched docs: ", length)
#         print(f"# of display: {show_size}")        
#         print("---------------------")        
#         for idx, doc in enumerate(response['hits']['hits']):
#             print("_id in index: " , doc['_id'])            
#             print(doc['_score'])            
#             print(doc['_source']['text'])
#             print("---------------------")
#             if idx == show_size-1:
#                 break
#     else:
#         print("There is no response")



        
###########################################    
### Chatbot Functions
###########################################    

# turn verbose to true to see the full logs and documents
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import BaseMessage


# We are also providing a different chat history retriever which outputs the history as a Claude chat (ie including the \n\n)
_ROLE_MAP = {"human": "\n\nHuman: ", "ai": "\n\nAssistant: "}
def _get_chat_history(chat_history):
    buffer = ""
    for dialogue_turn in chat_history:
        if isinstance(dialogue_turn, BaseMessage):
            role_prefix = _ROLE_MAP.get(dialogue_turn.type, f"{dialogue_turn.type}: ")
            buffer += f"\n{role_prefix}{dialogue_turn.content}"
        elif isinstance(dialogue_turn, tuple):
            human = "\n\nHuman: " + dialogue_turn[0]
            ai = "\n\nAssistant: " + dialogue_turn[1]
            buffer += "\n" + "\n".join([human, ai])
        else:
            raise ValueError(
                f"Unsupported chat history format: {type(dialogue_turn)}."
                f" Full chat history: {chat_history} "
            )
    return buffer


import ipywidgets as ipw
from IPython.display import display, clear_output

class ChatUX:
    """ A chat UX using IPWidgets
    """
    def __init__(self, qa, retrievalChain = False):
        self.qa = qa
        self.name = None
        self.b=None
        self.retrievalChain = retrievalChain
        self.out = ipw.Output()

        if "ConversationChain" in str(type(self.qa)):
            self.streaming = self.qa.llm.streaming
        elif "ConversationalRetrievalChain" in str(type(self.qa)):
            self.streaming = self.qa.combine_docs_chain.llm_chain.llm.streaming

    def start_chat(self):
        print("Starting chat bot")
        display(self.out)
        self.chat(None)


    def chat(self, _):
        if self.name is None:
            prompt = ""
        else: 
            prompt = self.name.value
        if 'q' == prompt or 'quit' == prompt or 'Q' == prompt:
            print("Thank you , that was a nice chat !!")
            return
        elif len(prompt) > 0:
            with self.out:
                thinking = ipw.Label(value="Thinking...")
                display(thinking)
                try:
                    if self.retrievalChain:
                        result = self.qa.run({'question': prompt })
                    else:
                        result = self.qa.run({'input': prompt }) #, 'history':chat_history})
                except:
                    result = "No answer because some errors occurredr"
                thinking.value=""
                if self.streaming:
                    response = f"AI:{result}"
                else:
                    print_ww(f"AI:{result}")
                self.name.disabled = True
                self.b.disabled = True
                self.name = None

        if self.name is None:
            with self.out:
                self.name = ipw.Text(description="You:", placeholder='q to quit')
                self.b = ipw.Button(description="Send")
                self.b.on_click(self.chat)
                display(ipw.Box(children=(self.name, self.b)))



