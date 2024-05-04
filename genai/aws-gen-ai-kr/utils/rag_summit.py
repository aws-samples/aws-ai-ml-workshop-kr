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
from copy import deepcopy
from pprint import pprint
from operator import itemgetter
from itertools import chain as ch
from typing import Any, Dict, List, Optional, List, Tuple
from opensearchpy import OpenSearch, RequestsHttpConnection

import base64
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

from utils import print_ww
from utils.opensearch_summit import opensearch_utils
from utils.common_utils import print_html

from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from langchain.prompts import PromptTemplate
from langchain.retrievers import AmazonKendraRetriever
from langchain_core.tracers import ConsoleCallbackHandler
from langchain.schema.output_parser import StrOutputParser
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

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
    prompt_types = ["answer_only", "answer_with_ref", "original", "ko_answer_only"]
    
    
    #First, find the paragraphs or sentences from the context that are most relevant to answering the question, 
    #Then, answer the question within <answer></answer> XML tags as much as you can.
    # Answer the question within <answer></answer> XML tags as much as you can.
    # Don't say "According to context" when answering.
    # Don't insert XML tag such as <context> and </context> when answering.

    
    @classmethod
    def get_system_prompt(cls, ):
        
        system_prompt = '''
                        You are a master answer bot designed to answer user's questions.
                        I'm going to give you contexts which consist of texts, tables and images.
                        Read the contexts carefully, because I'm going to ask you a question about it.
                        '''
        return system_prompt

    @classmethod
    def get_human_prompt(cls, images=None, tables=None):

        human_prompt = []

        image_template = {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64," + "IMAGE_BASE64",
            },
        }
        text_template = {
            "type": "text",
            "text": '''
                    Here is the contexts as texts: <contexts>{contexts}</contexts>
                    TABLE_PROMPT

                    First, find a few paragraphs or sentences from the contexts that are most relevant to answering the question.
                    Then, answer the question as much as you can.

                    Skip the preamble and go straight into the answer.
                    Don't insert any XML tag such as <contexts> and </contexts> when answering.
                    Answer in Korean.

                    Here is the question: <question>{question}</question>

                    If the question cannot be answered by the contexts, say "No relevant contexts".
            '''
        }

        table_prompt = '''
                Here is the contexts as tables (table as text): <tables_summay>{tables_text}</tables_summay>
                Here is the contexts as tables (table as html): <tables_html>{tables_html}</tables_html>
        '''
        if tables != None:
            text_template["text"] = text_template["text"].replace("TABLE_PROMPT", table_prompt)
            for table in tables:
                #if table.metadata["image_base64"]:
                if "image_base64" in table.metadata:
                    image_template["image_url"]["url"] = image_template["image_url"]["url"].replace("IMAGE_BASE64", table.metadata["image_base64"])
                    human_prompt.append(image_template)
        else: text_template["text"] = text_template["text"].replace("TABLE_PROMPT", "")

        if images != None:
            for image in images:
                image_template["image_url"]["url"] = image_template["image_url"]["url"].replace("IMAGE_BASE64", image.page_content)
                human_prompt.append(image_template)

        human_prompt.append(text_template)

        return human_prompt

#     @classmethod
#     def get_qa(cls, prompt_type="answer_only"):
        
#         assert prompt_type in cls.prompt_types, "Check your prompt_type"
        
#         if prompt_type == "answer_only":
            
#             prompt = """
#             \n\nHuman:
#             You are a master answer bot designed to answer software developer's questions.
#             I'm going to give you a context. Read the context carefully, because I'm going to ask you a question about it.

#             Here is the context: <context>{context}</context>
            
#             First, find a few paragraphs or sentences from the context that are most relevant to answering the question.
#             Then, answer the question as much as you can.

#             Skip the preamble and go straight into the answer.
#             Don't insert any XML tag such as <context> and </context> when answering.
            
#             Here is the question: <question>{question}</question>

#             If the question cannot be answered by the context, say "No relevant context".
#             \n\nAssistant: Here is the answer. """

#         elif prompt_type == "answer_with_ref":
            
#             prompt = """
#             \n\nHuman:
#             You are a master answer bot designed to answer software developer's questions.
#             I'm going to give you a context. Read the context carefully, because I'm going to ask you a question about it.

#             Here is the context: <context>{context}</context>

#             First, find the paragraphs or sentences from the context that are most relevant to answering the question, and then print them in numbered order.
#             The format of paragraphs or sentences to the question should look like what's shown between the <references></references> tags.
#             Make sure to follow the formatting and spacing exactly.

#             <references>
#             [Examples of question + answer pairs using parts of the given context, with answers written exactly like how Claude’s output should be structured]
#             </references>

#             If there are no relevant paragraphs or sentences, write "No relevant context" instead.

#             Then, answer the question within <answer></answer> XML tags.
#             Answer as much as you can.
#             Skip the preamble and go straight into the answer.
#             Don't say "According to context" when answering.
#             Don't insert XML tag such as <context> and </context> when answering.
#             If needed, answer using bulleted format.
#             If relevant paragraphs or sentences have code block, please show us that as code block.

#             Here is the question: <question>{question}</question>

#             If the question cannot be answered by the context, say "No relevant context".

#             \n\nAssistant: Here is the most relevant sentence in the context:"""

#         elif prompt_type == "original":
#             prompt = """
#             \n\nHuman: Here is the context, inside <context></context> XML tags.

#             <context>
#             {context}
#             </context>

#             Only using the context as above, answer the following question with the rules as below:
#                 - Don't insert XML tag such as <context> and </context> when answering.
#                 - Write as much as you can
#                 - Be courteous and polite
#                 - Only answer the question if you can find the answer in the context with certainty.

#             Question:
#             {question}

#             If the answer is not in the context, just say "I don't know"
#             \n\nAssistant:"""
        
#         if prompt_type == "ko_answer_only":
            
#             prompt = """
#             \n\nHuman:
#             You are a master answer bot designed to answer software developer's questions.
#             I'm going to give you a context. Read the context carefully, because I'm going to ask you a question about it.

#             Here is the context: <context>{context}</context>
            
#             First, find a few paragraphs or sentences from the context that are most relevant to answering the question.
#             Then, answer the question as much as you can.

#             Skip the preamble and go straight into the answer.
#             Don't insert any XML tag such as <context> and </context> when answering.
            
#             Here is the question: <question>{question}</question>

#             Answer in Korean.
#             If the question cannot be answered by the context, say "No relevant context".
#             \n\nAssistant: Here is the answer. """

#         prompt_template = PromptTemplate(
#             template=prompt, input_variables=["context", "question"]
#         )
        
#         return prompt_template

    @staticmethod
    def get_rag_fusion():

        system_prompt = """
                        You are a helpful assistant that generates multiple search queries that is semantically simiar to a single input query.
                        Skip the preamble and generate in Korean.
                        """
        human_prompt = """
                        Generate multiple search queries related to: {query}
                        OUTPUT ({query_augmentation_size} queries):
                       """
        
        system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)
        human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)
        prompt = ChatPromptTemplate.from_messages(
            [system_message_template, human_message_template]
        )
        
        return prompt

    @classmethod
    def get_hyde(cls, template_type):

        assert template_type in cls.template_types, "Check your template_type"
        
        system_prompt = """
                        You are a master answer bot designed to answer user's questions.
                        """
        human_prompt = """
                        Here is the question: <question>{query}</question>
                        
                        HYDE_TEMPLATE
                        Skip the preamble and generate in Korean.
                       """
        

        # There are a few different templates to choose from
        # These are just different ways to generate hypothetical documents
        hyde_template = {
            "web_search": "Please write a concise passage to answer the question.",
            "sci_fact": "Please write a concise scientific paper passage to support/refute the claim.",
            "fiqa": "Please write a concise financial article passage to answer the question.",
            "trec_news": "Please write a concise news passage about the topic."
        }
        human_prompt = human_prompt.replace("HYDE_TEMPLATE", hyde_template[template_type])
        
        system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)
        human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)
        prompt = ChatPromptTemplate.from_messages(
            [system_message_template, human_message_template]
        )
        
        return prompt

############################################################
# RetrievalQA (Langchain)
############################################################
pretty_contexts = None
augmentation = None

class qa_chain():
    
    def __init__(self, **kwargs):

        system_prompt = kwargs["system_prompt"]
        self.llm_text = kwargs["llm_text"]
        self.retriever = kwargs["retriever"]
        self.system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)
        self.return_context = kwargs.get("return_context", False)
        self.verbose = kwargs.get("verbose", False)
        
    def invoke(self, **kwargs):
        global pretty_contexts
        global augmentation
        
        query, verbose = kwargs["query"], kwargs.get("verbose", self.verbose)
        tables, images = None, None
        if self.retriever.complex_doc:
            #retrieval, tables, images = self.retriever.get_relevant_documents(query)
            retrieval, tables, images = self.retriever.invoke(query)

            invoke_args = {
                "contexts": "\n\n".join([doc.page_content for doc in retrieval]),
                "tables_text": "\n\n".join([doc.page_content for doc in tables]),
                "tables_html": "\n\n".join([doc.metadata["text_as_html"] if "text_as_html" in doc.metadata else "" for doc in tables]),
                "question": query
            }
        else:
            #retrieval = self.retriever.get_relevant_documents(query)
            retrieval = self.retriever.invoke(query)
            invoke_args = {
                "contexts": "\n\n".join([doc.page_content for doc in retrieval]),
                "question": query
            }

        human_prompt = prompt_repo.get_human_prompt(
            images=images,
            tables=tables
        )
        human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)
        prompt = ChatPromptTemplate.from_messages(
            [self.system_message_template, human_message_template]
        )

        chain = prompt | self.llm_text | StrOutputParser()
        
        self.verbose = verbose
        response = chain.invoke(
            invoke_args,
            config={'callbacks': [ConsoleCallbackHandler()]} if self.verbose else {}
        )
        pretty_contexts = tuple(pretty_contexts)

        return response, pretty_contexts, retrieval, augmentation

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
def list_up(similar_docs_semantic, similar_docs_keyword, similar_docs_wo_reranker, similar_docs):
    combined_list = []
    combined_list.append(similar_docs_semantic)
    combined_list.append(similar_docs_keyword)
    combined_list.append(similar_docs_wo_reranker)
    combined_list.append(similar_docs)

    return combined_list

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
    token_limit = 300

    @classmethod
    # semantic search based
    def get_semantic_similar_docs_by_langchain(cls, **kwargs):

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

        if kwargs.get("hybrid", False) and results:
            max_score = results[0][1]
            new_results = []
            for doc in results:
                nomalized_score = float(doc[1]/max_score)
                new_results.append((doc[0], nomalized_score))
            results = deepcopy(new_results)

        return results

    @classmethod
    def control_streaming_mode(cls, llm, stream=True):

        if stream:
            llm.streaming = True
            llm.callbacks = [StreamingStdOutCallbackHandler()]
        else:
            llm.streaming = False
            llm.callbacks = None

        return llm

    @classmethod
    # semantic search based
    def get_semantic_similar_docs(cls, **kwargs):

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
            filter=kwargs.get("boolean_filter", []),
            search_type="semantic", # enable semantic search
            vector_field="vector_field", # for semantic search  check by using index_info = os_client.indices.get(index=index_name)
            vector=kwargs["llm_emb"].embed_query(kwargs["query"]),
            k=kwargs["k"]
        )
        query["size"] = kwargs["k"]

        #print ("\nsemantic search query: ")
        #pprint (query)

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
            filter=kwargs["filter"]
        )
        query["size"] = kwargs["k"]

        #print ("\nlexical search query: ")
        #pprint (query)

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
        global augmentation

        search_types = ["approximate_search", "script_scoring", "painless_scripting"]
        space_types = ["l2", "l1", "linf", "cosinesimil", "innerproduct", "hammingbit"]

        assert "llm_emb" in kwargs, "Check your llm_emb"
        assert "query" in kwargs, "Check your query"
        assert "query_transformation_prompt" in kwargs, "Check your query_transformation_prompt"
        assert kwargs.get("search_type", "approximate_search") in search_types, f'Check your search_type: {search_types}'
        assert kwargs.get("space_type", "l2") in space_types, f'Check your space_type: {space_types}'
        assert kwargs.get("llm_text", None) != None, "Check your llm_text"

        llm_text = kwargs["llm_text"]
        query_augmentation_size = kwargs["query_augmentation_size"]
        query_transformation_prompt = kwargs["query_transformation_prompt"]

        llm_text = cls.control_streaming_mode(llm_text, stream=False) ## trun off llm streaming
        generate_queries = query_transformation_prompt | llm_text | StrOutputParser() | (lambda x: x.split("\n"))

        rag_fusion_query = generate_queries.invoke(
            {
                "query": kwargs["query"],
                "query_augmentation_size": kwargs["query_augmentation_size"]
            }
        )

        rag_fusion_query = [query for query in rag_fusion_query if query != ""]
        if len(rag_fusion_query) > query_augmentation_size: rag_fusion_query = rag_fusion_query[-query_augmentation_size:]
        rag_fusion_query.insert(0, kwargs["query"])
        augmentation = rag_fusion_query

        if kwargs["verbose"]:
            print("\n")
            print("===== RAG-Fusion Queries =====")
            print(rag_fusion_query)

        llm_text = cls.control_streaming_mode(llm_text, stream=True)## trun on llm streaming

        tasks = []
        for query in rag_fusion_query:
            semantic_search = partial(
                cls.get_semantic_similar_docs,
                os_client=kwargs["os_client"],
                index_name=kwargs["index_name"],
                query=query,
                k=kwargs["k"],
                boolean_filter=kwargs.get("boolean_filter", []),
                llm_emb=kwargs["llm_emb"],
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
        global augmentation

        def _get_hyde_response(query, prompt, llm_text):

            chain = prompt | llm_text | StrOutputParser()
            
            return chain.invoke({"query": query})

        search_types = ["approximate_search", "script_scoring", "painless_scripting"]
        space_types = ["l2", "l1", "linf", "cosinesimil", "innerproduct", "hammingbit"]

        assert "llm_emb" in kwargs, "Check your llm_emb"
        assert "query" in kwargs, "Check your query"
        assert "hyde_query" in kwargs, "Check your hyde_query"
        assert kwargs.get("search_type", "approximate_search") in search_types, f'Check your search_type: {search_types}'
        assert kwargs.get("space_type", "l2") in space_types, f'Check your space_type: {space_types}'
        assert kwargs.get("llm_text", None) != None, "Check your llm_text"

        query = kwargs["query"]
        llm_text = kwargs["llm_text"]
        hyde_query = kwargs["hyde_query"]

        tasks = []
        llm_text = cls.control_streaming_mode(llm_text, stream=False) ## trun off llm streaming
        for template_type in hyde_query:
            hyde_response = partial(
                _get_hyde_response,
                query=query,
                prompt=prompt_repo.get_hyde(template_type),
                llm_text=llm_text
            )
            tasks.append(cls.hyde_pool.apply_async(hyde_response,))
        hyde_answers = [task.get() for task in tasks]
        hyde_answers.insert(0, query)

        tasks = []
        llm_text = cls.control_streaming_mode(llm_text, stream=True) ## trun on llm streaming
        for hyde_answer in hyde_answers:
            semantic_search = partial(
                cls.get_semantic_similar_docs,
                os_client=kwargs["os_client"],
                index_name=kwargs["index_name"],
                query=hyde_answer,
                k=kwargs["k"],
                boolean_filter=kwargs.get("boolean_filter", []),
                llm_emb=kwargs["llm_emb"],
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
        augmentation = hyde_answers[1]
        if kwargs["verbose"]:
            print("\n")
            print("===== HyDE Answers =====")
            print(hyde_answers)

        return similar_docs

    @classmethod
    # ParentDocument based
    def get_parent_document_similar_docs(cls, **kwargs):

        child_search_results = kwargs["similar_docs"]
        
        parent_info, similar_docs = {}, []
        for rank, (doc, score) in enumerate(child_search_results):
            parent_id = doc.metadata["parent_id"]
            if parent_id != "NA": ## For Tables and Images
                if parent_id not in parent_info:
                    parent_info[parent_id] = (rank+1, score)
            else:
                if kwargs["hybrid"]:
                    similar_docs.append((doc, score))
                else:
                    similar_docs.append((doc))
        
        parent_ids = sorted(parent_info.items(), key=lambda x: x[1], reverse=False)
        parent_ids = list(map(lambda x:x[0], parent_ids))
        
        if parent_ids:
            parent_docs = opensearch_utils.get_documents_by_ids(
                os_client=kwargs["os_client"],
                ids=parent_ids,
                index_name=kwargs["index_name"],
            )

            if parent_docs["docs"]:
                for res in parent_docs["docs"]:
                    doc_id = res["_id"]
                    doc = Document(
                        page_content=res["_source"]["text"],
                        metadata=res["_source"]["metadata"]
                    )
                    if kwargs["hybrid"]:
                        similar_docs.append((doc, parent_info[doc_id][1]))
                    else:
                        similar_docs.append((doc))

        if kwargs["hybrid"]:
            similar_docs = sorted(
                similar_docs,
                key=lambda x: x[1],
                reverse=True
            )
        
        if kwargs["verbose"]:
            print("===== ParentDocument =====")
            print (f'filter: {kwargs["boolean_filter"]}')
            print (f'# child_docs: {len(child_search_results)}')
            print (f'# parent docs: {len(similar_docs)}')
            print (f'# duplicates: {len(child_search_results)-len(similar_docs)}')

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
                if kwargs["verbose"]:
                    print(f"\n[Exeeds ReRanker token limit] Number of chunk_docs after split and chunking= {len(splited_docs)}\n")

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
    def get_element(cls, **kwargs):

        similar_docs = copy.deepcopy(kwargs["similar_docs"])
        tables, images = [], []

        for doc in similar_docs:

            category = doc.metadata.get("category", None)
            if category != None:
                if category == "Table":
                    doc.page_content = doc.metadata["origin_table"]
                    tables.append(doc)
                elif category == "Image":
                    doc.page_content = doc.metadata["image_base64"]
                    images.append(doc)

        return tables, images


    @classmethod
    # hybrid (lexical + semantic) search based
    def search_hybrid(cls, **kwargs):

        assert "query" in kwargs, "Check your query"
        assert "llm_emb" in kwargs, "Check your llm_emb"
        assert "index_name" in kwargs, "Check your index_name"
        assert "os_client" in kwargs, "Check your os_client"

        rag_fusion = kwargs.get("rag_fusion", False)
        hyde = kwargs.get("hyde", False)
        parent_document = kwargs.get("parent_document", False)
        hybrid_search_debugger = kwargs.get("hybrid_search_debugger", "None")
        
        

        assert (rag_fusion + hyde) <= 1, "choose only one between RAG-FUSION and HyDE"
        if rag_fusion:
            assert "query_augmentation_size" in kwargs, "if you use RAG-FUSION, Check your query_augmentation_size"
        if hyde:
            assert "hyde_query" in kwargs, "if you use HyDE, Check your hyde_query"

        verbose = kwargs.get("verbose", False)
        async_mode = kwargs.get("async_mode", True)
        reranker = kwargs.get("reranker", False)
        complex_doc = kwargs.get("complex_doc", False)
        search_filter = deepcopy(kwargs.get("filter", []))

        #search_filter.append({"term": {"metadata.family_tree": "child"}})
        if parent_document:
            parent_doc_filter = {
                "bool":{
                    "should":[ ## or condition
                        {"term": {"metadata.family_tree": "child"}},
                        {"term": {"metadata.family_tree": "parent_table"}},
                        {"term": {"metadata.family_tree": "parent_image"}},   
                    ]
                }
            }
            search_filter.append(parent_doc_filter)
        else:
            parent_doc_filter = {
                "bool":{
                    "should":[ ## or condition
                        {"term": {"metadata.family_tree": "child"}},
                        {"term": {"metadata.family_tree": "parent_table"}},
                        {"term": {"metadata.family_tree": "parent_image"}},   
                    ]
                }
            }
            search_filter.append(parent_doc_filter)
            
            
        def do_sync():

            if rag_fusion:
                similar_docs_semantic = cls.get_rag_fusion_similar_docs(
                    index_name=kwargs["index_name"],
                    os_client=kwargs["os_client"],
                    llm_emb=kwargs["llm_emb"],

                    query=kwargs["query"],
                    k=kwargs.get("k", 5) if not reranker else int(kwargs["k"]*1.5),
                    boolean_filter=search_filter,
                    hybrid=True,

                    llm_text=kwargs.get("llm_text", None),
                    query_augmentation_size=kwargs["query_augmentation_size"],
                    query_transformation_prompt=kwargs.get("query_transformation_prompt", None),
                    fusion_algorithm=kwargs.get("fusion_algorithm", "RRF"), # ["RRF", "simple_weighted"]

                    verbose=kwargs.get("verbose", False),
                )
            elif hyde:
                similar_docs_semantic = cls.get_hyde_similar_docs(
                    index_name=kwargs["index_name"],
                    os_client=kwargs["os_client"],
                    llm_emb=kwargs["llm_emb"],

                    query=kwargs["query"],
                    k=kwargs.get("k", 5) if not reranker else int(kwargs["k"]*1.5),
                    boolean_filter=search_filter,
                    hybrid=True,

                    llm_text=kwargs.get("llm_text", None),
                    hyde_query=kwargs["hyde_query"],
                    fusion_algorithm=kwargs.get("fusion_algorithm", "RRF"), # ["RRF", "simple_weighted"]

                    verbose=kwargs.get("verbose", False),
                )
            else:
                similar_docs_semantic = cls.get_semantic_similar_docs(
                    index_name=kwargs["index_name"],
                    os_client=kwargs["os_client"],
                    llm_emb=kwargs["llm_emb"],

                    query=kwargs["query"],
                    k=kwargs.get("k", 5) if not reranker else int(kwargs["k"]*1.5),
                    boolean_filter=search_filter,
                    hybrid=True
                )

            similar_docs_keyword = cls.get_lexical_similar_docs(
                index_name=kwargs["index_name"],
                os_client=kwargs["os_client"],

                query=kwargs["query"],
                k=kwargs.get("k", 5) if not reranker else int(kwargs["k"]*1.5),
                minimum_should_match=kwargs.get("minimum_should_match", 0),
                filter=search_filter,
                hybrid=True
            )
            
            if hybrid_search_debugger == "semantic": similar_docs_keyword = []
            elif hybrid_search_debugger == "lexical": similar_docs_semantic = []

            return similar_docs_semantic, similar_docs_keyword

        def do_async():

            if rag_fusion:
                semantic_search = partial(
                    cls.get_rag_fusion_similar_docs,
                    index_name=kwargs["index_name"],
                    os_client=kwargs["os_client"],
                    llm_emb=kwargs["llm_emb"],

                    query=kwargs["query"],
                    k=kwargs.get("k", 5) if not reranker else int(kwargs["k"]*1.5),
                    boolean_filter=search_filter,
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
                    index_name=kwargs["index_name"],
                    os_client=kwargs["os_client"],
                    llm_emb=kwargs["llm_emb"],

                    query=kwargs["query"],
                    k=kwargs.get("k", 5) if not reranker else int(kwargs["k"]*1.5),
                    boolean_filter=search_filter,
                    hybrid=True,

                    llm_text=kwargs.get("llm_text", None),
                    hyde_query=kwargs["hyde_query"],
                    fusion_algorithm=kwargs.get("fusion_algorithm", "RRF"), # ["RRF", "simple_weighted"]

                    verbose=kwargs.get("verbose", False),
                )
            else:
                semantic_search = partial(
                    cls.get_semantic_similar_docs,
                    index_name=kwargs["index_name"],
                    os_client=kwargs["os_client"],
                    llm_emb=kwargs["llm_emb"],

                    query=kwargs["query"],
                    k=kwargs.get("k", 5) if not reranker else int(kwargs["k"]*1.5),
                    boolean_filter=search_filter,
                    hybrid=True
                )

            lexical_search = partial(
                cls.get_lexical_similar_docs,
                index_name=kwargs["index_name"],
                os_client=kwargs["os_client"],

                query=kwargs["query"],
                k=kwargs.get("k", 5) if not reranker else int(kwargs["k"]*1.5),
                minimum_should_match=kwargs.get("minimum_should_match", 0),
                filter=search_filter,
                hybrid=True
            )
            semantic_pool = cls.pool.apply_async(semantic_search,)
            lexical_pool = cls.pool.apply_async(lexical_search,)
            similar_docs_semantic, similar_docs_keyword = semantic_pool.get(), lexical_pool.get()
            
            if hybrid_search_debugger == "semantic": similar_docs_keyword = []
            elif hybrid_search_debugger == "lexical": similar_docs_semantic = []
            
            return similar_docs_semantic, similar_docs_keyword

        if async_mode:
            similar_docs_semantic, similar_docs_keyword = do_async()
        else:
            similar_docs_semantic, similar_docs_keyword = do_sync()

        similar_docs = cls.get_ensemble_results(
            doc_lists=[similar_docs_semantic, similar_docs_keyword],
            weights=kwargs.get("ensemble_weights", [.51, .49]),
            algorithm=kwargs.get("fusion_algorithm", "RRF"), # ["RRF", "simple_weighted"]
            c=60,
            k=kwargs.get("k", 5) if not reranker else int(kwargs["k"]*1.5),
        )
        #print (len(similar_docs_keyword), len(similar_docs_semantic), len(similar_docs))
        #print ("1-similar_docs")
        #for i, doc in enumerate(similar_docs): print (i, doc)

        if verbose: 
            similar_docs_wo_reranker = copy.deepcopy(similar_docs)

        if reranker:
            reranker_endpoint_name = kwargs["reranker_endpoint_name"]
            similar_docs = cls.get_rerank_docs(
                llm_text=kwargs["llm_text"],
                query=kwargs["query"],
                context=similar_docs,
                k=kwargs.get("k", 5),
                reranker_endpoint_name=reranker_endpoint_name,
                verbose=verbose
            )

        #print ("2-similar_docs")
        #for i, doc in enumerate(similar_docs): print (i, doc)

        if parent_document:
            similar_docs = cls.get_parent_document_similar_docs(
                index_name=kwargs["index_name"],
                os_client=kwargs["os_client"],
                similar_docs=similar_docs,
                hybrid=True,
                boolean_filter=search_filter,
                verbose=verbose
            )

        if complex_doc:
            tables, images = cls.get_element(
                similar_docs=list(map(lambda x:x[0], similar_docs))
            )

        if verbose:
            similar_docs_semantic_pretty = opensearch_utils.opensearch_pretty_print_documents_with_score("semantic", similar_docs_semantic)
            similar_docs_keyword_pretty = opensearch_utils.opensearch_pretty_print_documents_with_score("keyword", similar_docs_keyword)
            similar_docs_wo_reranker_pretty = []
            if reranker:
                similar_docs_wo_reranker_pretty = opensearch_utils.opensearch_pretty_print_documents_with_score("wo_reranker", similar_docs_wo_reranker)
                
            similar_docs_pretty = opensearch_utils.opensearch_pretty_print_documents_with_score("similar_docs", similar_docs)

        similar_docs = list(map(lambda x:x[0], similar_docs))
        global pretty_contexts
        pretty_contexts = list_up(similar_docs_semantic_pretty, similar_docs_keyword_pretty, similar_docs_wo_reranker_pretty, similar_docs_pretty)
        
        #if complex_doc: return similar_docs, tables, images
        #else: return similar_docs
    
        if complex_doc: return similar_docs, tables, images
        else:
            similar_docs_filtered = []
            for doc in similar_docs:
                category = "None"
                if "category" in doc.metadata:
                    category = doc.metadata["category"]

                if category not in {"Table", "Image"}:
                    similar_docs_filtered.append(doc)
            return similar_docs_filtered
        
        

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
    ensemble_weights = [0.51, 0.49]
    verbose = False
    async_mode = True
    reranker = False
    reranker_endpoint_name = ""
    rag_fusion = False
    query_augmentation_size: Any
    rag_fusion_prompt = prompt_repo.get_rag_fusion()
    llm_text: Any
    llm_emb: Any
    hyde = False
    hyde_query: Any
    parent_document = False
    complex_doc = False
    hybrid_search_debugger = "None"

    def update_search_params(self, **kwargs):

        self.k = kwargs.get("k", 3)
        self.minimum_should_match = kwargs.get("minimum_should_match", 0)
        self.filter = kwargs.get("filter", [])
        self.index_name = kwargs.get("index_name", self.index_name)
        self.fusion_algorithm = kwargs.get("fusion_algorithm", self.fusion_algorithm)
        self.ensemble_weights = kwargs.get("ensemble_weights", self.ensemble_weights)
        self.verbose = kwargs.get("verbose", self.verbose)
        self.async_mode = kwargs.get("async_mode", self.async_mode)
        self.reranker = kwargs.get("reranker", self.reranker)
        self.reranker_endpoint_name = kwargs.get("reranker_endpoint_name", self.reranker_endpoint_name)
        self.rag_fusion = kwargs.get("rag_fusion", self.rag_fusion)
        self.query_augmentation_size = kwargs.get("query_augmentation_size", 3)
        self.hyde = kwargs.get("hyde", self.hyde)
        self.hyde_query = kwargs.get("hyde_query", ["web_search"])
        self.parent_document = kwargs.get("parent_document", self.parent_document)
        self.complex_doc = kwargs.get("complex_doc", self.complex_doc)
        self.hybrid_search_debugger = kwargs.get("hybrid_search_debugger", self.hybrid_search_debugger)

    def _reset_search_params(self, ):

        self.k = 3
        self.minimum_should_match = 0
        self.filter = []

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:

        '''
        It can be called by "retriever.invoke" statements
        '''
        search_hybrid_result = retriever_utils.search_hybrid(
            query=query,
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
            parent_document = self.parent_document,
            complex_doc = self.complex_doc,
            llm_text=self.llm_text,
            llm_emb=self.llm_emb,
            verbose=self.verbose,
            hybrid_search_debugger=self.hybrid_search_debugger
        )
        #self._reset_search_params()

        return search_hybrid_result

#################################################################
# Document visualization
#################################################################
def show_context_used(context_list, limit=10):

    context_list = copy.deepcopy(context_list)

    if type(context_list) == tuple: context_list=context_list[0]
    for idx, context in enumerate(context_list):

        if idx < limit:

            category = "None"
            if "category" in context.metadata:
                category = context.metadata["category"]

            print("\n-----------------------------------------------")
            if category != "None":
                print(f"{idx+1}. Category: {category}, Chunk: {len(context.page_content)} Characters")   
            else:
                print(f"{idx+1}. Chunk: {len(context.page_content)} Characters")
            print("-----------------------------------------------")

            if category == "Image" or (category == "Table" and "image_base64" in context.metadata):
                img = Image.open(BytesIO(base64.b64decode(context.metadata["image_base64"])))
                plt.imshow(img)
                plt.show()
                context.metadata["image_base64"], context.metadata["origin_image"] = "", ""

            context.metadata["orig_elements"] = ""
            print_ww(context.page_content)
            if "text_as_html" in context.metadata: print_html(context.metadata["text_as_html"])
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

