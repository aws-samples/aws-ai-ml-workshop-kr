############################################################    
############################################################    
# RAG 관련 함수들
############################################################    
############################################################    

import json
import copy
import boto3
import numpy as np
import pandas as pd
from pprint import pprint
from operator import itemgetter
from itertools import chain as ch
from typing import Any, Dict, List, Optional, List, Tuple
from opensearchpy import OpenSearch, RequestsHttpConnection

from utils import print_ww
from utils.opensearch import opensearch_utils

from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from langchain.prompts import PromptTemplate
from langchain.retrievers import AmazonKendraRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler

import threading
from functools import partial
from multiprocessing.pool import ThreadPool
#pool = ThreadPool(processes=2)
#rag_fusion_pool = ThreadPool(processes=5)

############################################################
# Prompt repo
############################################################
class prompt_repo():
    
    template_types = ["web_search", "sci_fact", "fiqa", "trec_news"]

    @staticmethod
    def get_rag_fusion():

        prompt = """
        \n\nHuman:
        You are a helpful assistant that generates multiple search queries based on a single input query.
        Generate multiple search queries related to: {query}
        OUTPUT ({query_augmentation_size} queries):
        \n\nAssistant:"""

        prompt_template = PromptTemplate(
            template=prompt, input_variables=["query", "query_augmentation_size"]
        )

        return prompt_template

    @classmethod
    def get_hyde(cls, template_type):

        assert template_type in cls.template_types, "Check your template_type"
        
        # There are a few different templates to choose from
        # These are just different ways to generate hypothetical documents
        hyde_template = {
            "web_search": """\n\nHuman:\nPlease write a concise passage to answer the question\nQuestion: {query}\nPassage:\n\nAssistant:""",
            "sci_fact": """\n\nHuman:\nPlease write a concise scientific paper passage to support/refute the claim\nClaim: {query}\nPassage:\n\nAssistant:""",
            "fiqa": """\n\nHuman:\nPlease write a concise financial article passage to answer the question\nQuestion: {query}\nPassage:\n\nAssistant:""",
            "trec_news": """\n\nHuman:\nPlease write a concise news passage about the topic\nTopic: {query}\nPassage:\n\nAssistant:"""
        }

        return PromptTemplate(template=hyde_template[template_type], input_variables=["query"])

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
                "boolean_filter": opensearch_utils.get_filter(
                    filter=kwargs.get("boolean_filter", [])
                ),
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

class retriever_utils():
    
    runtime_client = boto3.Session().client('sagemaker-runtime')
    pool = ThreadPool(processes=2)
    rag_fusion_pool = ThreadPool(processes=5)
    hyde_pool = ThreadPool(processes=4)
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=512,
        chunk_overlap=0,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
    token_limit = 510

    #@classmethod
    #def get_num_tokns(cls, **kwargs):


    @classmethod
    # semantic search based
    def get_semantic_similar_docs(cls, **kwargs):

        #print(f"Thread={threading.get_ident()}, Process={os.getpid()}")
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
            boolean_filter=opensearch_utils.get_filter(
                filter=kwargs.get("boolean_filter", [])
            ),
        )

        # print ("\nsemantic search args: ")
        # pprint ({
        #     "k": kwargs.get("k", 5),
        #     "search_type": kwargs.get("search_type", "approximate_search"),
        #     "space_type": kwargs.get("space_type", "l2"),
        #     "boolean_filter": opensearch_utils.get_filter(filter=kwargs.get("boolean_filter", []))
        # })

        if kwargs.get("hybrid", False) and results:
            max_score = results[0][1]
            new_results = []
            for doc in results:
                nomalized_score = float(doc[1]/max_score)
                new_results.append((doc[0], nomalized_score))
            results = copy.deepcopy(new_results)

        return results

    @classmethod
    # lexical(keyword) search based (using Amazon OpenSearch)
    def get_lexical_similar_docs(cls, **kwargs):

        assert "query" in kwargs, "Check your query"
        assert "k" in kwargs, "Check your k"
        assert "os_client" in kwargs, "Check your os_client"
        assert "index_name" in kwargs, "Check your index_name"

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

        # print ("\nlexical search query: ")
        # pprint (query)

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

    @classmethod
    # rag-fusion based
    def get_rag_fusion_similar_docs(cls, **kwargs):

        search_types = ["approximate_search", "script_scoring", "painless_scripting"]
        space_types = ["l2", "l1", "linf", "cosinesimil", "innerproduct", "hammingbit"]

        assert "vector_db" in kwargs, "Check your vector_db"
        assert "query" in kwargs, "Check your query"
        assert "query_transformation_prompt" in kwargs, "Check your query_transformation_prompt"
        assert kwargs.get("search_type", "approximate_search") in search_types, f'Check your search_type: {search_types}'
        assert kwargs.get("space_type", "l2") in space_types, f'Check your space_type: {space_types}'
        assert kwargs.get("llm_text", None) != None, "Check your llm_text"

        llm_text = kwargs["llm_text"]
        query_augmentation_size = kwargs["query_augmentation_size"]
        query_transformation_prompt = kwargs["query_transformation_prompt"]

        generate_queries = (
            {
                "query": itemgetter("query"),
                "query_augmentation_size": itemgetter("query_augmentation_size")
            }
            | query_transformation_prompt
            | llm_text
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )
        rag_fusion_query = generate_queries.invoke(
            {
                "query": kwargs["query"],
                "query_augmentation_size": kwargs["query_augmentation_size"]
            }
        )
        rag_fusion_query = [query for query in rag_fusion_query if query != ""]
        if len(rag_fusion_query) > query_augmentation_size: rag_fusion_query = rag_fusion_query[-query_augmentation_size:]
        rag_fusion_query.insert(0, kwargs["query"])

        if kwargs["verbose"]:
            print("===== RAG-Fusion Queries =====")
            print(rag_fusion_query)

        tasks = []
        for query in rag_fusion_query:
            semantic_search = partial(
                cls.get_semantic_similar_docs,
                vector_db=kwargs["vector_db"],
                query=query,
                k=kwargs["k"],
                boolean_filter=kwargs.get("filter", []),
                hybrid=True
            )
            tasks.append(cls.rag_fusion_pool.apply_async(semantic_search,))
        rag_fusion_docs = [task.get() for task in tasks]

        similar_docs = cls.get_ensemble_results(
            doc_lists=rag_fusion_docs,
            weights=[1/(query_augmentation_size+1)]*(query_augmentation_size+1), #query_augmentation_size + original query
            algorithm=kwargs.get("fusion_algorithm", "RRF"), # ["RRF", "simple_weighted"]
            c=60,
            k=kwargs["k"],
        )

        return similar_docs

    @classmethod
    # HyDE based
    def get_hyde_similar_docs(cls, **kwargs):

        def get_hyde_response(query, prompt, llm_text):

            chain = (
                {
                    "query": itemgetter("query")
                }
                | prompt
                | llm_text
                | StrOutputParser()
            )
            return chain.invoke({"query": query})

        search_types = ["approximate_search", "script_scoring", "painless_scripting"]
        space_types = ["l2", "l1", "linf", "cosinesimil", "innerproduct", "hammingbit"]

        assert "vector_db" in kwargs, "Check your vector_db"
        assert "query" in kwargs, "Check your query"
        assert "hyde_query" in kwargs, "Check your hyde_query"
        assert kwargs.get("search_type", "approximate_search") in search_types, f'Check your search_type: {search_types}'
        assert kwargs.get("space_type", "l2") in space_types, f'Check your space_type: {space_types}'
        assert kwargs.get("llm_text", None) != None, "Check your llm_text"

        query = kwargs["query"]
        llm_text = kwargs["llm_text"]
        hyde_query = kwargs["hyde_query"]

        tasks = []
        for template_type in hyde_query:
            hyde_response = partial(
                get_hyde_response,
                query=query,
                prompt=prompt_repo.get_hyde(template_type),
                llm_text=llm_text
            )
            tasks.append(cls.hyde_pool.apply_async(hyde_response,))
        hyde_answers = [task.get() for task in tasks]
        hyde_answers.insert(0, query)

        tasks = []
        for hyde_answer in hyde_answers:
            semantic_search = partial(
                cls.get_semantic_similar_docs,
                vector_db=kwargs["vector_db"],
                query=hyde_answer,
                k=kwargs["k"],
                boolean_filter=kwargs.get("filter", []),
                hybrid=True
            )
            tasks.append(cls.hyde_pool.apply_async(semantic_search,))
        hyde_docs = [task.get() for task in tasks]
        hyde_doc_size = len(hyde_docs)

        similar_docs = cls.get_ensemble_results(
            doc_lists=hyde_docs,
            weights=[1/(hyde_doc_size)]*(hyde_doc_size), #query_augmentation_size + original query
            algorithm=kwargs.get("fusion_algorithm", "RRF"), # ["RRF", "simple_weighted"]
            c=60,
            k=kwargs["k"],
        )
        if kwargs["verbose"]:
            print("===== HyDE Answers =====")
            print(hyde_answers)

        return similar_docs

    @classmethod
    def get_rerank_docs(cls, **kwargs):

        assert "reranker_endpoint_name" in kwargs, "Check your reranker_endpoint_name"
        assert "k" in kwargs, "Check your k"

        contexts, query, llm_text, rerank_queries = kwargs["context"], kwargs["query"], kwargs["llm_text"], {"inputs":[]}

        exceed_info = []
        for idx, (context, score) in enumerate(contexts):
            page_content = context.page_content
            token_size = llm_text.get_num_tokens(query+page_content)
            exceed_flag = False

            if token_size > cls.token_limit:
                exceed_flag = True
                splited_docs = cls.text_splitter.split_documents([context])
                print(f"\nNumber of chunk_docs after split and chunking= {len(splited_docs)}\n")

                partial_set, length = [], []
                for splited_doc in splited_docs:
                    rerank_queries["inputs"].append({"text": query, "text_pair": splited_doc.page_content})
                    length.append(llm_text.get_num_tokens(splited_doc.page_content))
                    partial_set.append(len(rerank_queries["inputs"])-1)
            else:
                rerank_queries["inputs"].append({"text": query, "text_pair": page_content})

            if exceed_flag:
                exceed_info.append([idx, exceed_flag, partial_set, length])
            else:
                exceed_info.append([idx, exceed_flag, len(rerank_queries["inputs"])-1, None])

        rerank_queries = json.dumps(rerank_queries)

        response = cls.runtime_client.invoke_endpoint(
            EndpointName=kwargs["reranker_endpoint_name"],
            ContentType="application/json",
            Accept="application/json",
            Body=rerank_queries
        )
        outs = json.loads(response['Body'].read().decode()) ## for json

        rerank_contexts = []
        for idx, exceed_flag, partial_set, length in exceed_info:
            if not exceed_flag:
                rerank_contexts.append((contexts[idx][0], outs[partial_set]["score"]))
            else:
                partial_scores = [outs[partial_idx]["score"] for partial_idx in partial_set]
                partial_scores = np.average(partial_scores, axis=0, weights=length)
                rerank_contexts.append((contexts[idx][0], partial_scores))

        #rerank_contexts = [(contexts[idx][0], out["score"]) for idx, out in enumerate(outs)]
        rerank_contexts = sorted(
            rerank_contexts,
            key=lambda x: x[1],
            reverse=True
        )

        return rerank_contexts[:kwargs["k"]]

    @classmethod
    # hybrid (lexical + semantic) search based
    def search_hybrid(cls, **kwargs):

        assert "query" in kwargs, "Check your query"
        assert "vector_db" in kwargs, "Check your vector_db"
        assert "index_name" in kwargs, "Check your index_name"
        assert "os_client" in kwargs, "Check your os_client"

        verbose = kwargs.get("verbose", False)
        async_mode = kwargs.get("async_mode", True)
        reranker = kwargs.get("reranker", False)
        rag_fusion = kwargs.get("rag_fusion", False)
        hyde = kwargs.get("hyde", False)

        def do_sync():

            if rag_fusion:
                similar_docs_semantic = cls.get_rag_fusion_similar_docs(
                    vector_db=kwargs["vector_db"],
                    query=kwargs["query"],
                    k=kwargs.get("k", 5) if not reranker else int(kwargs["k"]*1.5),
                    boolean_filter=kwargs.get("filter", []),
                    hybrid=True,
                    llm_text=kwargs.get("llm_text", None),
                    query_augmentation_size=kwargs["query_augmentation_size"],
                    query_transformation_prompt=kwargs.get("query_transformation_prompt", None),
                    fusion_algorithm=kwargs.get("fusion_algorithm", "RRF"), # ["RRF", "simple_weighted"]
                    verbose=kwargs.get("verbose", False),
                )
            elif hyde:
                similar_docs_semantic = cls.get_hyde_similar_docs(
                    vector_db=kwargs["vector_db"],
                    query=kwargs["query"],
                    k=kwargs.get("k", 5) if not reranker else int(kwargs["k"]*1.5),
                    boolean_filter=kwargs.get("filter", []),
                    hybrid=True,
                    llm_text=kwargs.get("llm_text", None),
                    hyde_query=kwargs["hyde_query"],
                    fusion_algorithm=kwargs.get("fusion_algorithm", "RRF"), # ["RRF", "simple_weighted"]
                    verbose=kwargs.get("verbose", False),
                )

            else:
                similar_docs_semantic = cls.get_semantic_similar_docs(
                    vector_db=kwargs["vector_db"],
                    query=kwargs["query"],
                    k=kwargs.get("k", 5) if not reranker else int(kwargs["k"]*1.5),
                    boolean_filter=kwargs.get("filter", []),
                    hybrid=True
                )

            similar_docs_keyword = cls.get_lexical_similar_docs(
                query=kwargs["query"],
                minimum_should_match=kwargs.get("minimum_should_match", 0),
                filter=kwargs.get("filter", []),
                index_name=kwargs["index_name"],
                os_client=kwargs["os_client"],
                k=kwargs.get("k", 5) if not reranker else int(kwargs["k"]*1.5),
                hybrid=True
            )

            return similar_docs_semantic, similar_docs_keyword

        def do_async():

            if rag_fusion:
                semantic_search = partial(
                    cls.get_rag_fusion_similar_docs,
                    vector_db=kwargs["vector_db"],
                    query=kwargs["query"],
                    k=kwargs.get("k", 5) if not reranker else int(kwargs["k"]*1.5),
                    boolean_filter=kwargs.get("filter", []),
                    hybrid=True,
                    llm_text=kwargs.get("llm_text", None),
                    query_augmentation_size=kwargs["query_augmentation_size"],
                    query_transformation_prompt=kwargs.get("query_transformation_prompt", None),
                    fusion_algorithm=kwargs.get("fusion_algorithm", "RRF"), # ["RRF", "simple_weighted"]
                    verbose=kwargs.get("verbose", False),
                )
            elif hyde:
                semantic_search = partial(
                    cls.get_hyde_similar_docs,
                    vector_db=kwargs["vector_db"],
                    query=kwargs["query"],
                    k=kwargs.get("k", 5) if not reranker else int(kwargs["k"]*1.5),
                    boolean_filter=kwargs.get("filter", []),
                    hybrid=True,
                    llm_text=kwargs.get("llm_text", None),
                    hyde_query=kwargs["hyde_query"],
                    fusion_algorithm=kwargs.get("fusion_algorithm", "RRF"), # ["RRF", "simple_weighted"]
                    verbose=kwargs.get("verbose", False),
                )
            else:
                semantic_search = partial(
                    cls.get_semantic_similar_docs,
                    vector_db=kwargs["vector_db"],
                    query=kwargs["query"],
                    k=kwargs.get("k", 5) if not reranker else int(kwargs["k"]*1.5),
                    boolean_filter=kwargs.get("filter", []),
                    hybrid=True
                )

            lexical_search = partial(
                cls.get_lexical_similar_docs,
                query=kwargs["query"],
                minimum_should_match=kwargs.get("minimum_should_match", 0),
                filter=kwargs.get("filter", []),
                index_name=kwargs["index_name"],
                os_client=kwargs["os_client"],
                k=kwargs.get("k", 5) if not reranker else int(kwargs["k"]*1.5),
                hybrid=True
            )
            semantic_pool = cls.pool.apply_async(semantic_search,)
            lexical_pool = cls.pool.apply_async(lexical_search,)
            similar_docs_semantic, similar_docs_keyword = semantic_pool.get(), lexical_pool.get()

            return similar_docs_semantic, similar_docs_keyword

        if async_mode:
            similar_docs_semantic, similar_docs_keyword = do_async()
        else:
            similar_docs_semantic, similar_docs_keyword = do_sync()

        similar_docs = cls.get_ensemble_results(
            doc_lists=[similar_docs_semantic, similar_docs_keyword],
            weights=kwargs.get("ensemble_weights", [.5, .5]),
            algorithm=kwargs.get("fusion_algorithm", "RRF"), # ["RRF", "simple_weighted"]
            c=60,
            k=kwargs.get("k", 5) if not reranker else int(kwargs["k"]*1.5),
        )
        #print (len(similar_docs_keyword), len(similar_docs_semantic), len(similar_docs))

        if reranker:
            reranker_endpoint_name = kwargs["reranker_endpoint_name"]
            similar_docs = cls.get_rerank_docs(
                llm_text=kwargs["llm_text"],
                query=kwargs["query"],
                context=similar_docs,
                k=kwargs.get("k", 5),
                reranker_endpoint_name=reranker_endpoint_name,
            )

        if verbose:

            print("##############################")
            print("async_mode")
            print("##############################")
            print(async_mode)

            print("##############################")
            print("reranker")
            print("##############################")
            print(reranker)

            print("##############################")
            print("rag_fusion")
            print("##############################")
            print(rag_fusion)
            
            print("##############################")
            print("HyDE")
            print("##############################")
            print(hyde)

            print("##############################")
            print("similar_docs_semantic")
            print("##############################")
            print(similar_docs_semantic)

            print("##############################")
            print("similar_docs_keyword")
            print("##############################")
            print(similar_docs_keyword)

            print("##############################")
            print("similar_docs")
            print("##############################")
            print(similar_docs)

        similar_docs = list(map(lambda x:x[0], similar_docs))

        return similar_docs

    @classmethod
    # Score fusion and re-rank (lexical + semantic)
    def get_ensemble_results(cls, doc_lists: List[List[Document]], weights, algorithm="RRF", c=60, k=5) -> List[Document]:

        assert algorithm in ["RRF", "simple_weighted"]

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
                elif algorithm == "simple_weighted":
                    score *= weight
                hybrid_score_dic[doc.page_content] += score

        # Sort documents by their scores in descending order
        sorted_documents = sorted(
            hybrid_score_dic.items(), key=lambda x: x[1], reverse=True
        )

        # Map the sorted page_content back to the original document objects
        page_content_to_doc_map = {
            doc.page_content: doc for doc_list in doc_lists for (doc, orig_score) in doc_list
        }

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
    async_mode = True
    reranker = False
    reranker_endpoint_name = ""
    rag_fusion = False
    query_augmentation_size: Any
    rag_fusion_prompt = prompt_repo.get_rag_fusion()
    llm_text: Any
    hyde = False
    hyde_query: Any

    def update_search_params(self, **kwargs):

        self.k = kwargs.get("k", 3)
        self.minimum_should_match = kwargs.get("minimum_should_match", 0)
        self.filter = kwargs.get("filter", [])
        self.index_name = kwargs.get("index_name", self.index_name)
        self.fusion_algorithm = kwargs.get("fusion_algorithm", self.fusion_algorithm)
        self.ensemble_weights = kwargs.get("ensemble_weights", self.ensemble_weights)
        self.verbose = kwargs.get("verbose", self.verbose)
        self.async_mode = kwargs.get("async_mode", True)
        self.reranker = kwargs.get("reranker", False)
        self.reranker_endpoint_name = kwargs.get("reranker_endpoint_name", self.reranker_endpoint_name)
        self.rag_fusion = kwargs.get("rag_fusion", False)
        self.query_augmentation_size = kwargs.get("query_augmentation_size", 3)
        self.llm_text = kwargs.get("llm_text", None)
        self.hyde = kwargs.get("hyde", False)
        self.hyde_query = kwargs.get("hyde_query", ["web_search"])

    def _reset_search_params(self, ):

        self.k = 3
        self.minimum_should_match = 0
        self.filter = []

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:

        search_hybrid_result = retriever_utils.search_hybrid(
            query=query,
            vector_db=self.vector_db,
            k=self.k,
            index_name=self.index_name,
            os_client=self.os_client,
            filter=self.filter,
            minimum_should_match=self.minimum_should_match,
            fusion_algorithm=self.fusion_algorithm, # ["RRF", "simple_weighted"]
            ensemble_weights=self.ensemble_weights, # 시멘트 서치에 가중치 0.5 , 키워드 서치 가중치 0.5 부여.
            async_mode=self.async_mode,
            reranker=self.reranker,
            reranker_endpoint_name=self.reranker_endpoint_name,
            rag_fusion=self.rag_fusion,
            query_augmentation_size=self.query_augmentation_size,
            query_transformation_prompt=self.rag_fusion_prompt if self.rag_fusion else "",
            hyde=self.hyde,
            hyde_query=self.hyde_query if self.hyde else [],
            llm_text=self.llm_text,
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

#################################################################
# JumpStart Embeddings
#################################################################

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

