
import os
import json
import copy
import boto3
import shutil
import argparse
from pprint import pprint
from utils import bedrock
from itertools import chain
from utils.bedrock import bedrock_info

from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import math
import base64
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from langchain.schema import Document
from pdf2image import convert_from_path
from requests_toolbelt import MultipartEncoder

import botocore
from utils.common_utils import retry
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from utils.chunk import parant_documents
from utils.opensearch import opensearch_utils
from langchain_community.vectorstores import OpenSearchVectorSearch

class preprocess():

    def __init__(self, args):

        self.args = args
        self.proc_prefix = self.args.proc_prefix #'/opt/ml/processing'

        self.input_dir = os.path.join(self.proc_prefix, "input")
        self.output_dir = os.path.join(self.proc_prefix, "output")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        print (os.listdir(self.input_dir))

        ############# 수정
        #self.file_path = os.path.join(self.input_dir, "sample-2.pdf")
        self.file_path = os.path.join(self.input_dir, self.args.file_name)
        #"./data/complex_pdf/sample-2.pdf"
        ###############

    def _initialization(self, ):

        self.boto3_bedrock = bedrock.get_bedrock_client(
            assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
            endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
            region=os.environ.get("AWS_DEFAULT_REGION", None),
        )

        self.llm_text = ChatBedrock(
            model_id=bedrock_info.get_model_id(model_name="Claude-V3-5-Sonnet"),
            client=self.boto3_bedrock,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            model_kwargs={
                "max_tokens": 2048,
                "stop_sequences": ["\n\nHuman"],
            }
        )

        self.llm_emb = BedrockEmbeddings(
            client=self.boto3_bedrock,
            model_id=bedrock_info.get_model_id(model_name="Titan-Text-Embeddings-V2")
        )
        self.dimension = 1024

    def _document_parsing(self, ):

        runtime_sm_client = boto3.client('runtime.sagemaker')

        # Prepare multipart form data
        encoder = MultipartEncoder(
            fields={
                'document': (os.path.basename(self.file_path), open(self.file_path, 'rb'), 'application/pdf'),
                'model': 'document-parse',
                'ocr': 'auto',
                'coordinates': 'true',
                'output_formats': '["markdown"]', #'["text", "html", "markdown"]',
                'base64_encoding': '["table", "figure"]'
            }
        )

        # Get the raw bytes of the multipart form data
        body = encoder.to_string()

        response = runtime_sm_client.invoke_endpoint(
            EndpointName=self.args.endpoint_document_parser,
            ContentType=encoder.content_type,  # This will be 'multipart/form-data; boundary=...'
            Body=body
        )

        result = response["Body"].read()
        parse_output = json.loads(result)

        return parse_output

    def _extract_image_table(self, parse_output):

        def processing(**kwargs):

            category = kwargs["category"]
            markdown = kwargs["markdown"]
            base64_encoding = kwargs["base64_encoding"]
            coordinates = kwargs["coordinates"]
            page = kwargs["page"]
            docs = kwargs["docs"]

            if page in docs: 
                docs[page].append(
                    {
                        "category": category,
                         "markdown": markdown,
                         "base64_encoding": base64_encoding,
                         "coordinates": coordinates
                    }
                )
            else:
                docs[page] = [
                    {
                        "category": category,
                        "markdown": markdown,
                        "base64_encoding": base64_encoding,
                        "coordinates": coordinates
                    }
                ]

            return docs

        def image_conversion(**kwargs):

            image_path = kwargs["image_path"]
            file_path = kwargs["file_path"]

            image_tmp_path = os.path.join(image_path, "tmp")
            if os.path.isdir(image_tmp_path): shutil.rmtree(image_tmp_path)
            os.mkdir(image_tmp_path)

            # from pdf to image
            pages = convert_from_path(file_path)
            for i, page in enumerate(pages):
                print (f'pdf page {i}, size: {page.size}')    
                page.save(f'{image_tmp_path}/{str(i+1)}.jpg', "JPEG")

            return image_tmp_path

        image_path = os.path.join(self.input_dir, "fig")
        if os.path.isdir(image_path): shutil.rmtree(image_path)
        os.mkdir(image_path)

        docs = {}
        texts = [
            Document(
                page_content=parse_output["content"]["markdown"]
            )
        ]

        ## extract_image_table
        image_tmp_path = image_conversion(
            image_path=image_path,
            file_path=self.file_path
        )

        for idx, value in enumerate(parse_output["elements"]):

            category = value["category"]
            markdown = value["content"]["markdown"]
            page = value["page"]

            if category in ["figure", "table"]:

                base64_encoding = value["base64_encoding"]    
                coordinates = value["coordinates"]    
                img = Image.open(BytesIO(base64.b64decode(base64_encoding)))
                plt.imshow(img)
                plt.show()

                page_img = Image.open(f'{image_tmp_path}/{page}.jpg')
                w, h = page_img.size  # PIL은 (width, height) 순서

                # 좌표 계산
                left = math.ceil(coordinates[0]["x"] * w)
                top = math.ceil(coordinates[0]["y"] * h)
                right = math.ceil(coordinates[1]["x"] * w)
                bottom = math.ceil(coordinates[3]["y"] * h)

                # PIL로 이미지 크롭
                crop_img = page_img.crop((left, top, right, bottom))

                crob_image_path = f'{image_path}/element-{idx}.jpg'
                crop_img.save(crob_image_path)

                w_crop, h_crop = crop_img.size
                image_token = w_crop*h_crop/750
                print (f'image: {crob_image_path}, shape: ({w_crop}, {h_crop}), image_token_for_claude3: {image_token}' )

            else:
                base64_encoding= ""
                coordinates=""

            docs = processing(
                docs=docs,
                page=page,
                category=category,
                markdown=markdown,
                base64_encoding=base64_encoding,
                coordinates=coordinates
            )

        return docs, texts

    def _context_generation_for_image(self, docs):

        def manipulate_docs_for_summary(docs):

            docs_for_summary = []
            for page, elements in docs.items():

                elements = [element for element in elements if element["category"] != "footer"]
                print (f'page: {page}, # elements: {len(elements)}')

                for idx, element in enumerate(elements):

                    category, markdown = element["category"], element["markdown"]
                    print (f'element idx: {idx}, category: {element["category"]}')

                    elements_copy = copy.deepcopy(elements)
                    if category in ("figure", "table"):  

                        summary_target = elements_copy.pop(idx)
                        contexts_markdown = '\n'.join([context["markdown"] for context in elements_copy])
                        docs_for_summary.append(
                            {
                                "target_category": summary_target["category"],
                                "target_base64": summary_target["base64_encoding"],
                                "target_markdown": summary_target["markdown"],
                                "contexts_markdown": contexts_markdown
                            }
                        )

            return docs_for_summary

        def get_summary_chain():

            system_prompt = "You are an assistant tasked with describing table and image."
            system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)

            human_prompt = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + "{image_base64}",
                    },
                },
                {
                    "type": "text",
                    "text": '''
                             <contexts>
                             {contexts}
                             </contexts>

                             주어진 이미지 또는 테이블을 자세히 분석하고 주어진 contexts를 참고하여 다음 정보를 추출해주세요:

                             1. <title> 태그 안의 제목을 정확히 제시해주세요.
                             2. <summary> 태그 안의 내용을 요약해주세요.
                             3. <entities> 태그 안의 모든 항목을 나열하고, 각 항목에 대한 간단한 설명을 제공해주세요.
                             4. <hypothetical_questions> 태그 안의 질문들을 모두 나열해주세요.
                            모든 정보는 원본 내용을 정확히 반영하되, 필요한 경우 약간의 추가 설명을 덧붙여 이해를 돕도록 해주세요.
                    '''
                },
            ]
            human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

            prompt = ChatPromptTemplate.from_messages(
                [
                    system_message_template,
                    human_message_template
                ]
            )

            summarize_chain = prompt | self.llm_text | StrOutputParser()

            return summarize_chain

        @retry(total_try_cnt=5, sleep_in_sec=10, retryable_exceptions=(botocore.exceptions.EventStreamError))
        def summary_img(summarize_chain, image_base64, contexts):

            img = Image.open(BytesIO(base64.b64decode(image_base64)))
            plt.imshow(img)
            plt.show()

            stream = summarize_chain.stream(
                {
                    "image_base64": image_base64,
                    "contexts": contexts
                }
            )
            response = ""
            for chunk in stream: response += chunk

            return response

        docs_for_summary = manipulate_docs_for_summary(docs)

        print ("docs_for_summary", len(docs_for_summary))


        summarize_chain = get_summary_chain()

        summaries = []
        for idx, doc in enumerate(docs_for_summary):
            summary = summary_img(summarize_chain, doc["target_base64"], doc["contexts_markdown"])
            summaries.append(summary)
            print ("\n==")
            print (idx)

        images_preprocessed = []
        for doc, summary in zip(docs_for_summary, summaries):

            metadata = {}
            metadata["markdown"] = doc["target_markdown"]
            metadata["category"] = "Image"
            metadata["image_base64"] = doc["target_base64"]

            doc = Document(
                page_content=summary,
                metadata=metadata
            )
            images_preprocessed.append(doc)

        for image in images_preprocessed:
            image.metadata["family_tree"], image.metadata["parent_id"] = "parent_image", "NA"

        return images_preprocessed, docs_for_summary

    def _context_generation_for_table(self, docs_for_summary):

        def get_summary_chain():

            system_prompt = "You are an assistant tasked with describing table and image."
            system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)

            human_prompt = [
                {
                    "type": "text",
                    "text": '''
                             Here is the table: <table>{table}</table>
                             Given table, give a concise summary.
                             Don't insert any XML tag such as <table> and </table> when answering.
                             Write in Korean.
                    '''
                },
            ]
            human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

            prompt = ChatPromptTemplate.from_messages(
                [
                    system_message_template,
                    human_message_template
                ]
            )

            summarize_chain = {"table": lambda x:x} | prompt | self.llm_text | StrOutputParser()

            return summarize_chain

        tables = [doc for doc in docs_for_summary if doc["target_category"] == "table"]
        summarize_chain = get_summary_chain()
        table_info = [t["target_markdown"] for t in tables]
        table_summaries = summarize_chain.batch(table_info, config={"max_concurrency": 1})

        tables_preprocessed = []
        for doc, summary in zip(tables, table_summaries):

            metadata = {}
            metadata["origin_table"] = doc["target_markdown"]
            metadata["text_as_html"] = doc["target_markdown"]
            metadata["category"] = "Table"
            metadata["image_base64"] = doc["target_base64"]

            doc = Document(
                page_content=summary,
                metadata=metadata
            )
            tables_preprocessed.append(doc)

        for table in tables_preprocessed:
            table.metadata["family_tree"], table.metadata["parent_id"] = "parent_table", "NA"

        return tables_preprocessed

    def _opensearch(self, ):

        index_body = {
            'settings': {
                'analysis': {
                    'analyzer': {
                        'my_analyzer': {
                                 'char_filter':['html_strip'],
                            'tokenizer': 'nori',
                            'filter': [
                                #'nori_number',
                                #'lowercase',
                                #'trim',
                                'my_nori_part_of_speech'
                            ],
                            'type': 'custom'
                        }
                    },
                    'tokenizer': {
                        'nori': {
                            'decompound_mode': 'mixed',
                            'discard_punctuation': 'true',
                            'type': 'nori_tokenizer'
                        }
                    },
                    "filter": {
                        "my_nori_part_of_speech": {
                            "type": "nori_part_of_speech",
                            "stoptags": [
                                "J", "XSV", "E", "IC","MAJ","NNB",
                                "SP", "SSC", "SSO",
                                "SC","SE","XSN","XSV",
                                "UNA","NA","VCP","VSV",
                                "VX"
                            ]
                        }
                    }
                },
                'index': {
                    'knn': True,
                    'knn.space_type': 'cosinesimil'  # Example space type
                }
            },
            'mappings': {
                'properties': {
                    'metadata': {
                        'properties': {
                            'source': {'type': 'keyword'},
                            'page_number': {'type':'long'},
                            'category': {'type':'text'},
                            'file_directory': {'type':'text'},
                            'last_modified': {'type': 'text'},
                            'type': {'type': 'keyword'},
                            'image_base64': {'type':'text'},
                            'origin_image': {'type':'text'},
                            'origin_table': {'type':'text'},
                        }
                    },
                    'text': {
                        'analyzer': 'my_analyzer',
                        'search_analyzer': 'my_analyzer',
                        'type': 'text'
                    },
                    'vector_field': {
                        'type': 'knn_vector',
                        'dimension': f"{self.dimension}" # Replace with your vector dimension
                    }
                }
            }
        }

        opensearch_domain_endpoint = self.args.opensearch_domain_endpoint
        opensearch_user_id = self.args.opensearch_user_id
        opensearch_user_password = self.args.opensearch_user_password
        index_name = self.args.index_name

        http_auth = (opensearch_user_id, opensearch_user_password) # Master username, Master password

        ## opensearch clinet 생성
        aws_region = os.environ.get("AWS_DEFAULT_REGION", None)
        os_client = opensearch_utils.create_aws_opensearch_client(
            aws_region,
            opensearch_domain_endpoint,
            http_auth
        )

        ## opensearch index 생성
        index_exists = opensearch_utils.check_if_index_exists(
            os_client,
            index_name
        )

        if index_exists:
            opensearch_utils.delete_index(
                os_client,
                index_name
            )

        opensearch_utils.create_index(os_client, index_name, index_body)
        index_info = os_client.indices.get(index=index_name)
        print("Index is created")
        pprint(index_info)

        vector_db = OpenSearchVectorSearch(
            index_name=index_name,
            opensearch_url=opensearch_domain_endpoint,
            embedding_function=self.llm_emb,
            http_auth=http_auth, # http_auth
            is_aoss=False,
            engine="faiss",
            space_type="l2",
            bulk_size=100000,
            timeout=60
        )

        return os_client, vector_db, index_name

    def _chunking_and_indexing(self, os_client, vector_db, index_name, texts, images_preprocessed, tables_preprocessed):

        parent_chunk_size = 4096
        parent_chunk_overlap = 0

        child_chunk_size = 1024
        child_chunk_overlap = 256

        opensearch_parent_key_name = "parent_id"
        opensearch_family_tree_key_name = "family_tree"

        parent_chunk_docs = parant_documents.create_parent_chunk(
            docs=texts,
            parent_id_key=opensearch_parent_key_name,
            family_tree_id_key=opensearch_family_tree_key_name,
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=parent_chunk_overlap
        )
        print(f'Number of parent_chunk_docs= {len(parent_chunk_docs)}')
        parent_ids = vector_db.add_documents(
            documents = parent_chunk_docs, 
            vector_field = "vector_field",
            bulk_size = 1000000
        )

        total_count_docs = opensearch_utils.get_count(os_client, index_name)
        print("total count docs: ", total_count_docs)

        # child_chunk_docs = create_child_chunk(parent_chunk_docs[0:1], parent_ids)
        child_chunk_docs = parant_documents.create_child_chunk(
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
            docs=parent_chunk_docs,
            parent_ids_value=parent_ids,
            parent_id_key=opensearch_parent_key_name,
            family_tree_id_key=opensearch_family_tree_key_name
        )

        print(f"Number of child_chunk_docs= {len(child_chunk_docs)}")
        parent_id = child_chunk_docs[0].metadata["parent_id"]
        print("child's parent_id: ", parent_id)
        print("\n###### Search parent in OpenSearch")


        ## Merge
        docs_preprocessed = list(chain(child_chunk_docs, tables_preprocessed, images_preprocessed))

        child_ids = vector_db.add_documents(
            documents=docs_preprocessed, 
            vector_field = "vector_field",
            bulk_size=1000000
        )
        print("length of child_ids: ", len(child_ids))

        return child_chunk_docs

    def execution(self, ):

        ## Initialization for Bedrock
        self._initialization()

        ## Document parsing
        parse_output = self._document_parsing()

        ## Context generation for images and tables
        docs, texts = self._extract_image_table(parse_output)
        images_preprocessed, docs_for_summary = self._context_generation_for_image(docs)
        tables_preprocessed = self._context_generation_for_table(docs_for_summary)

        ## Opensearch setting
        os_client, vector_db, index_name = self._opensearch()

        ## Chunking(Parent document) and indexing
        child_chunk_docs = self._chunking_and_indexing(os_client, vector_db, index_name, texts, images_preprocessed, tables_preprocessed)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="us-west-2")
    parser.add_argument("--proc_prefix", type=str, default="./src")
    parser.add_argument("--endpoint_document_parser", type=str, default="")
    parser.add_argument("--opensearch_domain_endpoint", type=str, default="")
    parser.add_argument("--opensearch_user_id", type=str, default="")
    parser.add_argument("--opensearch_user_password", type=str, default="")
    parser.add_argument("--index_name", type=str, default="")
    parser.add_argument("--file_name", type=str, default="")

    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))

    os.environ['AWS_DEFAULT_REGION'] = args.region

    prep = preprocess(args)
    prep.execution()

