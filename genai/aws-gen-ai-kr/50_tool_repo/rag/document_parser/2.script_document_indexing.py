#!/usr/bin/env python3
"""
Document Indexing Script
This script indexes PDF documents with complex layouts (text, tables, images) into OpenSearch.
"""

import os
import sys
import json
import copy
import shutil
import argparse
import math
import base64
from pathlib import Path
from itertools import chain

import boto3
from dotenv import load_dotenv
from termcolor import colored
from PIL import Image
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
from requests_toolbelt import MultipartEncoder
from botocore.exceptions import ClientError

from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import OpenSearchVectorSearch

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import bedrock
from src.utils.bedrock import bedrock_info
from src.utils.common_utils import retry
from src.utils.chunk import parant_documents
from src.utils.opensearch import opensearch_utils


class DocumentIndexer:
    """Main class for document indexing pipeline"""

    def __init__(self, args):
        self.args = args
        self.load_config()
        self.setup_clients()

    def load_config(self):
        """Load configuration from environment variables"""
        # Load .env file
        env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(env_path)

        # AWS Configuration
        self.aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-west-2')
        os.environ['AWS_DEFAULT_REGION'] = self.aws_region
        os.environ['AWS_REGION'] = self.aws_region

        # Model Configuration
        self.llm_model_name = os.getenv('LLM_MODEL_NAME', 'Claude-V4-5-Sonnet-CRI')
        self.embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME', 'Cohere-Embed-V4-CRI')
        self.embedding_dimension = int(os.getenv('EMBEDDING_DIMENSION', '1024'))

        # Chunking Configuration
        self.parent_chunk_size = int(os.getenv('PARENT_CHUNK_SIZE', '1024'))
        self.parent_chunk_overlap = int(os.getenv('PARENT_CHUNK_OVERLAP', '0'))
        self.child_chunk_size = int(os.getenv('CHILD_CHUNK_SIZE', '256'))
        self.child_chunk_overlap = int(os.getenv('CHILD_CHUNK_OVERLAP', '64'))

        # OpenSearch Configuration
        self.index_name = os.getenv('INDEX_NAME', 'complex-doc-index')
        self.opensearch_domain_endpoint = os.getenv('OPENSEARCH_DOMAIN_ENDPOINT')
        self.opensearch_user_id = os.getenv('OPENSEARCH_USER_ID')
        self.opensearch_user_password = os.getenv('OPENSEARCH_USER_PASSWORD')

        # SageMaker Configuration
        self.document_parse_endpoint = os.getenv('DOCUMENT_PARSE_ENDPOINT_NAME')

        # File paths
        self.file_path = self.args.file_path
        self.image_path = Path(self.args.output_dir) / 'fig'
        self.image_path.mkdir(parents=True, exist_ok=True)

        print(colored("\n[CONFIGURATION LOADED]", "cyan", attrs=["bold"]))
        print(colored(f"  AWS Region: {self.aws_region}", "cyan"))
        print(colored(f"  LLM Model: {self.llm_model_name}", "cyan"))
        print(colored(f"  Embedding Model: {self.embedding_model_name} (dimension={self.embedding_dimension})", "cyan"))
        print(colored(f"  Parent Chunk: size={self.parent_chunk_size}, overlap={self.parent_chunk_overlap}", "cyan"))
        print(colored(f"  Child Chunk: size={self.child_chunk_size}, overlap={self.child_chunk_overlap}", "cyan"))
        print(colored(f"  Index Name: {self.index_name}", "cyan"))

    def setup_clients(self):
        """Setup AWS clients"""
        # Bedrock client
        self.boto3_bedrock = bedrock.get_bedrock_client(
            assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
            endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
            region=self.aws_region,
        )

        # LLM for text generation
        self.llm_text = ChatBedrock(
            model_id=bedrock_info.get_model_id(model_name=self.llm_model_name),
            client=self.boto3_bedrock,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            model_kwargs={
                "max_tokens": 16384,
            }
        )

        # Embedding model
        self.llm_emb = BedrockEmbeddings(
            client=self.boto3_bedrock,
            model_id=bedrock_info.get_model_id(model_name=self.embedding_model_name),
            model_kwargs={
                "output_dimension": self.embedding_dimension,
                "embedding_types": ["float"],
                "input_type": "search_document"
            }
        )

        # SageMaker runtime client
        self.runtime_sm_client = boto3.client('runtime.sagemaker')

        print(colored("\n[AWS CLIENTS INITIALIZED]", "green", attrs=["bold"]))

    def parse_document(self):
        """Parse PDF document using Upstage Document Parse"""
        print(colored("\n[STEP 1: PARSING DOCUMENT]", "yellow", attrs=["bold"]))
        print(colored(f"  File: {self.file_path}", "yellow"))

        # Prepare multipart form data
        encoder = MultipartEncoder(
            fields={
                'document': (os.path.basename(self.file_path), open(self.file_path, 'rb'), 'application/pdf'),
                'model': 'document-parse',
                'ocr': 'auto',
                'coordinates': 'true',
                'output_formats': '["markdown"]',
                'base64_encoding': '["table", "figure"]',
                'chart_recognition': 'false',
            }
        )

        body = encoder.to_string()

        response = self.runtime_sm_client.invoke_endpoint(
            EndpointName=self.document_parse_endpoint,
            ContentType=encoder.content_type,
            Body=body
        )

        result = response["Body"].read()
        parse_output = json.loads(result)

        print(colored(f"  ✓ Parsed successfully: {len(parse_output.get('elements', []))} elements found", "green"))
        return parse_output

    def extract_image_table(self, parse_output):
        """Extract images and tables from parsed document"""
        print(colored("\n[STEP 2: EXTRACTING IMAGES & TABLES]", "yellow", attrs=["bold"]))

        def postprocessing(**kwargs):
            category = kwargs["category"]
            markdown = kwargs["markdown"]
            base64_encoding = kwargs["base64_encoding"]
            coordinates = kwargs["coordinates"]
            page = kwargs["page"]
            docs = kwargs["docs"]

            if page in docs:
                docs[page].append({
                    "category": category,
                    "markdown": markdown,
                    "base64_encoding": base64_encoding,
                    "coordinates": coordinates
                })
            else:
                docs[page] = [{
                    "category": category,
                    "markdown": markdown,
                    "base64_encoding": base64_encoding,
                    "coordinates": coordinates
                }]
            return docs

        def extract_image_from_pdf():
            image_tmp_path = self.image_path / "tmp"
            if image_tmp_path.exists():
                shutil.rmtree(image_tmp_path)
            image_tmp_path.mkdir(parents=True)

            pages = convert_from_path(self.file_path)
            for i, page in enumerate(pages):
                page.save(f'{image_tmp_path}/{i+1}.jpg', "JPEG")
            return image_tmp_path

        docs = {}
        texts = [Document(page_content=parse_output["content"]["markdown"])]

        image_tmp_path = extract_image_from_pdf()

        for idx, value in enumerate(parse_output["elements"]):
            category = value["category"]
            markdown = value["content"]["markdown"]
            page = value["page"]

            if category in ["figure", "table"]:
                base64_encoding = value["base64_encoding"]
                coordinates = value["coordinates"]

                # Crop image from page
                page_img = Image.open(f'{image_tmp_path}/{page}.jpg')
                w, h = page_img.size

                left = math.ceil(coordinates[0]["x"] * w)
                top = math.ceil(coordinates[0]["y"] * h)
                right = math.ceil(coordinates[1]["x"] * w)
                bottom = math.ceil(coordinates[3]["y"] * h)

                crop_img = page_img.crop((left, top, right, bottom))
                crop_img.save(f'{self.image_path}/element-{idx}.jpg')
            else:
                base64_encoding = ""
                coordinates = ""

            docs = postprocessing(
                docs=docs,
                page=page,
                category=category,
                markdown=markdown,
                base64_encoding=base64_encoding,
                coordinates=coordinates
            )

        num_extracted = len([d for page_docs in docs.values() for d in page_docs if d['category'] in ['figure', 'table']])
        print(colored(f"  ✓ Extracted {num_extracted} images/tables", "green"))
        return docs, texts

    def summarize_images_tables(self, docs):
        """Summarize images and tables using Claude"""
        print(colored("\n[STEP 3: SUMMARIZING IMAGES & TABLES]", "yellow", attrs=["bold"]))

        # Prepare docs for summary
        docs_for_summary = []
        for page, elements in docs.items():
            elements = [element for element in elements if element["category"] != "footer"]

            for idx, element in enumerate(elements):
                category = element["category"]

                if category in ("figure", "table"):
                    elements_copy = copy.deepcopy(elements)
                    summary_target = elements_copy.pop(idx)
                    contexts_markdown = '\n'.join([context["markdown"] for context in elements_copy])
                    docs_for_summary.append({
                        "target_category": summary_target["category"],
                        "target_base64": summary_target["base64_encoding"],
                        "target_markdown": summary_target["markdown"],
                        "contexts_markdown": contexts_markdown
                    })

        # Create summary chain
        system_prompt = """You are an expert document analyst specializing in extracting structured information from images and tables."""
        system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)

        human_prompt = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + "{image_base64}"},
            },
            {
                "type": "text",
                "text": '''
# Role
You are a document analysis assistant.

# Context
<contexts>
{contexts}
</contexts>

# Instructions
Analyze the given image or table in detail, referencing the provided contexts. Extract and organize the following information:

1. **Title**: Identify and present the exact title from the <title> tags
2. **Summary**: Summarize the content from the <summary> tags
3. **Entities**: List all items from the <entities> tags with brief descriptions for each
4. **Hypothetical Questions**: List all questions from the <hypothetical_questions> tags

# Output Format
Present all information accurately from the original content. Add clarifying explanations only where necessary to enhance understanding.

# Language
Respond in Korean.
                '''
            },
        ]
        human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

        prompt = ChatPromptTemplate.from_messages([system_message_template, human_message_template])
        summarize_chain = prompt | self.llm_text | StrOutputParser()

        @retry(total_try_cnt=5, sleep_in_sec=60, retryable_exceptions=(ClientError,))
        def summary_img(image_base64, contexts):
            stream = summarize_chain.stream({
                "image_base64": image_base64,
                "contexts": contexts
            })
            response = ""
            for chunk in stream:
                response += chunk
            return response

        # Summarize images and tables
        summaries = []
        for idx, doc in enumerate(docs_for_summary):
            print(colored(f"  Processing {idx+1}/{len(docs_for_summary)}...", "yellow"))
            summary = summary_img(doc["target_base64"], doc["contexts_markdown"])
            summaries.append(summary)

        # Process images
        images_preprocessed = []
        for doc, summary in zip(docs_for_summary, summaries):
            metadata = {
                "markdown": doc["target_markdown"],
                "category": "Image",
                "image_base64": doc["target_base64"],
                "family_tree": "parent_image",
                "parent_id": "NA"
            }
            images_preprocessed.append(Document(page_content=summary, metadata=metadata))

        # Process tables separately for markdown summary
        tables = [doc for doc in docs_for_summary if doc["target_category"] == "table"]

        if tables:
            table_human_prompt = [{
                "type": "text",
                "text": '''
Here is the table: <table>{table}</table>
Given table, give a concise summary.
Don't insert any XML tag such as <table> and </table> when answering.
Write in Korean.
                '''
            }]
            table_human_template = HumanMessagePromptTemplate.from_template(table_human_prompt)
            table_prompt = ChatPromptTemplate.from_messages([system_message_template, table_human_template])
            table_chain = {"table": lambda x: x} | table_prompt | self.llm_text | StrOutputParser()

            table_info = [t["target_markdown"] for t in tables]
            table_summaries = table_chain.batch(table_info, config={"max_concurrency": 1})

            tables_preprocessed = []
            for doc, summary in zip(tables, table_summaries):
                metadata = {
                    "origin_table": doc["target_markdown"],
                    "text_as_html": doc["target_markdown"],
                    "category": "Table",
                    "image_base64": doc["target_base64"],
                    "family_tree": "parent_table",
                    "parent_id": "NA"
                }
                tables_preprocessed.append(Document(page_content=summary, metadata=metadata))
        else:
            tables_preprocessed = []

        print(colored(f"  ✓ Summarized {len(images_preprocessed)} images/tables", "green"))
        return images_preprocessed, tables_preprocessed

    def setup_opensearch(self):
        """Setup OpenSearch index"""
        print(colored("\n[STEP 4: SETTING UP OPENSEARCH]", "yellow", attrs=["bold"]))

        # Validate OpenSearch credentials
        if not all([self.opensearch_domain_endpoint, self.opensearch_user_id, self.opensearch_user_password]):
            raise ValueError(
                "OpenSearch credentials not found in .env file. "
                "Please set OPENSEARCH_DOMAIN_ENDPOINT, OPENSEARCH_USER_ID, and OPENSEARCH_USER_PASSWORD"
            )

        http_auth = (self.opensearch_user_id, self.opensearch_user_password)

        # Create OpenSearch client
        os_client = opensearch_utils.create_aws_opensearch_client(
            self.aws_region,
            self.opensearch_domain_endpoint,
            http_auth
        )

        # Define index schema
        index_body = {
            'settings': {
                'analysis': {
                    'analyzer': {
                        'my_analyzer': {
                            'char_filter': ['html_strip'],
                            'tokenizer': 'nori',
                            'filter': ['my_nori_part_of_speech'],
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
                            "stoptags": ["J", "XSV", "E", "IC", "MAJ", "NNB", "SP", "SSC", "SSO", "SC", "SE", "XSN", "XSV", "UNA", "NA", "VCP", "VSV", "VX"]
                        }
                    }
                },
                'index': {
                    'knn': True
                }
            },
            'mappings': {
                'properties': {
                    'metadata': {
                        'properties': {
                            'source': {'type': 'keyword'},
                            'page_number': {'type': 'long'},
                            'category': {'type': 'text'},
                            'file_directory': {'type': 'text'},
                            'last_modified': {'type': 'text'},
                            'type': {'type': 'keyword'},
                            'image_base64': {'type': 'text'},
                            'origin_image': {'type': 'text'},
                            'origin_table': {'type': 'text'},
                        }
                    },
                    'text': {
                        'analyzer': 'my_analyzer',
                        'search_analyzer': 'my_analyzer',
                        'type': 'text'
                    },
                    'vector_field': {
                        'type': 'knn_vector',
                        'dimension': self.embedding_dimension,
                        'method': {
                            'name': 'hnsw',
                            'space_type': 'cosinesimil',
                            'engine': 'faiss'
                        }
                    }
                }
            }
        }

        # Delete existing index if it exists
        if opensearch_utils.check_if_index_exists(os_client, self.index_name):
            print(colored(f"  Deleting existing index: {self.index_name}", "yellow"))
            opensearch_utils.delete_index(os_client, self.index_name)

        # Create new index
        opensearch_utils.create_index(os_client, self.index_name, index_body)
        print(colored(f"  ✓ Index created: {self.index_name}", "green"))

        # Create vector database object
        vector_db = OpenSearchVectorSearch(
            index_name=self.index_name,
            opensearch_url=self.opensearch_domain_endpoint,
            embedding_function=self.llm_emb,
            http_auth=http_auth,
            is_aoss=False,
            engine="faiss",
            space_type="l2",
            bulk_size=100000,
            timeout=60
        )

        return os_client, vector_db

    def index_documents(self, os_client, vector_db, texts, images_preprocessed, tables_preprocessed):
        """Index documents into OpenSearch using parent-child chunking strategy"""
        print(colored("\n[STEP 5: INDEXING DOCUMENTS]", "yellow", attrs=["bold"]))

        # Create parent chunks
        parent_chunk_docs = parant_documents.create_parent_chunk(
            docs=texts,
            parent_id_key="parent_id",
            family_tree_id_key="family_tree",
            parent_chunk_size=self.parent_chunk_size,
            parent_chunk_overlap=self.parent_chunk_overlap
        )
        print(colored(f"  ✓ Created {len(parent_chunk_docs)} parent chunks", "green"))

        # Index parent chunks
        parent_ids = vector_db.add_documents(
            documents=parent_chunk_docs,
            vector_field="vector_field",
            bulk_size=1000000
        )
        print(colored(f"  ✓ Indexed {len(parent_ids)} parent documents", "green"))

        # Create child chunks
        child_chunk_docs = parant_documents.create_child_chunk(
            child_chunk_size=self.child_chunk_size,
            child_chunk_overlap=self.child_chunk_overlap,
            docs=parent_chunk_docs,
            parent_ids_value=parent_ids,
            parent_id_key="parent_id",
            family_tree_id_key="family_tree"
        )
        print(colored(f"  ✓ Created {len(child_chunk_docs)} child chunks", "green"))

        # Merge all documents
        docs_preprocessed = list(chain(child_chunk_docs, tables_preprocessed, images_preprocessed))

        # Index all documents
        child_ids = vector_db.add_documents(
            documents=docs_preprocessed,
            vector_field="vector_field",
            bulk_size=1000000
        )
        print(colored(f"  ✓ Indexed {len(child_ids)} child/table/image documents", "green"))

        # Get total count
        total_count = opensearch_utils.get_count(os_client, self.index_name)
        print(colored(f"  ✓ Total documents in index: {total_count['count']}", "green"))

        return total_count

    def run(self):
        """Execute the full indexing pipeline"""
        print(colored("\n" + "="*80, "magenta", attrs=["bold"]))
        print(colored("Document Indexing Pipeline", "magenta", attrs=["bold"]))
        print(colored("="*80, "magenta", attrs=["bold"]))

        try:
            # Step 1: Parse document
            parse_output = self.parse_document()

            # Step 2: Extract images and tables
            docs, texts = self.extract_image_table(parse_output)

            # Step 3: Summarize images and tables
            images_preprocessed, tables_preprocessed = self.summarize_images_tables(docs)

            # Step 4: Setup OpenSearch
            os_client, vector_db = self.setup_opensearch()

            # Step 5: Index documents
            total_count = self.index_documents(os_client, vector_db, texts, images_preprocessed, tables_preprocessed)

            print(colored("\n" + "="*80, "magenta", attrs=["bold"]))
            print(colored("✓ INDEXING COMPLETED SUCCESSFULLY!", "green", attrs=["bold"]))
            print(colored(f"  Total documents indexed: {total_count['count']}", "green"))
            print(colored(f"  Index name: {self.index_name}", "green"))
            print(colored("="*80 + "\n", "magenta", attrs=["bold"]))

        except Exception as e:
            print(colored(f"\n✗ Error during indexing: {str(e)}", "red", attrs=["bold"]))
            raise


def main():
    parser = argparse.ArgumentParser(
        description='Index PDF documents with complex layouts into OpenSearch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python document_indexing.py --file_path ../data/sample.pdf
  python document_indexing.py --file_path /path/to/document.pdf --output_dir ./output
        '''
    )

    parser.add_argument(
        '--file_path',
        type=str,
        required=True,
        help='Path to the PDF file to index'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='Directory to save extracted images (default: ./output)'
    )

    args = parser.parse_args()

    # Validate file path
    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        sys.exit(1)

    # Run indexing pipeline
    indexer = DocumentIndexer(args)
    indexer.run()


if __name__ == "__main__":
    main()
