import boto3
import json
import os
import io
import pdfplumber
import requests
import re
import uuid
from datetime import datetime
from requests_aws4auth import AWS4Auth
from botocore.config import Config


def handler(event, context):
    """Lambda handler that processes a PDF document from SQS queue"""
    print(f"Event received: {json.dumps(event)}")

    # Process SQS messages
    if 'Records' not in event:
        print("No SQS records found in event")
        return

    # Get environment variables
    collection_endpoint = os.environ['COLLECTION_ENDPOINT']
    cr_index_name = os.environ['CR_INDEX_NAME']
    region = os.environ.get('REGION') or os.environ.get('AWS_REGION')

    # Initialize processor
    processor = DocumentProcessor(
        collection_endpoint=collection_endpoint,
        cr_index_name=cr_index_name,
        region=region
    )

    # Process each record (should be one per Lambda invocation with batch size 1)
    for record in event['Records']:
        try:
            # Parse message
            message = json.loads(record['body'])
            bucket = message['bucket']
            key = message['key']

            print(f"Processing document {key} from bucket {bucket}")

            # Get file from S3
            s3 = boto3.client('s3')
            response = s3.get_object(Bucket=bucket, Key=key)
            pdf_content = response['Body'].read()

            # Process document
            s3_uri = f"s3://{bucket}/{key}"
            segments_indexed = processor.process_document(pdf_content, key, s3_uri)

            print(f"Successfully processed {key}: indexed {segments_indexed} segments")

        except Exception as e:
            print(f"Error processing message: {str(e)}")
            # Don't raise exception - let SQS delete the message to avoid retries
            # If you want retries, you can raise an exception here


class DocumentProcessor:
    """Handles PDF document processing including extraction, segmentation, and indexing"""

    def __init__(self, collection_endpoint, cr_index_name, region):
        self.collection_endpoint = collection_endpoint
        self.cr_index_name = cr_index_name
        self.region = region

        # Configuration parameters
        self.segment_size = 1000  # Size of text segments
        self.segment_overlap = 200  # Overlap between segments
        self.enable_context = True  # Whether to add contextual information

        # Initialize Bedrock client with retry config
        retry_config = Config(
            region_name=region,
            retries={"max_attempts": 5, "mode": "standard"}
        )
        self.bedrock_client = boto3.client("bedrock-runtime", config=retry_config)

        # Create AWS4Auth for OpenSearch
        credentials = boto3.Session().get_credentials()
        self.auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            'aoss',
            session_token=credentials.token
        )

    def process_document(self, pdf_content, document_name, source_uri):
        """Process a PDF document from content bytes"""
        # Extract text from PDF
        document_text = self._extract_text(pdf_content)
        if not document_text:
            print(f"No text extracted from {document_name}")
            return 0

        # Create segments from document text
        segments = self._create_segments(document_text, document_name)
        print(f"Created {len(segments)} segments from {document_name}")

        # Add contextual information if enabled
        if self.enable_context:
            segments = self._enhance_with_context(segments, document_text)

        # Index segments to OpenSearch
        indexed_count = self._index_segments(segments, document_name, source_uri)

        return indexed_count

    def _extract_text(self, pdf_bytes):
        """Extract text from PDF content"""
        all_text = ""
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text(
                        x_tolerance=1,
                        layout=True,
                        keep_blank_chars=True,
                        use_text_flow=False
                    )
                    if text:
                        # Clean and normalize text
                        text = re.sub(r'\s+', ' ', text).strip()
                        all_text += text + " "
            return all_text.strip()
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def _create_segments(self, text, document_name):
        """Split text into semantic segments at natural boundaries with proper overlap"""
        segments = []
        # Split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_segment = ""
        segment_id = 0

        for sentence in sentences:
            # If adding this sentence would exceed the segment size and we already have content
            if len(current_segment) + len(sentence) > self.segment_size and current_segment:
                segment_id += 1
                segments.append({
                    "id": f"{document_name}_segment_{segment_id}",
                    "content": current_segment.strip(),
                    "position": segment_id
                })

                # Create overlap using the last 1-3 sentences (adjust as needed)
                # This ensures meaningful overlap between segments
                overlap_size = min(3, len(current_segment.split('. ')))
                overlap_sentences = '. '.join(current_segment.split('. ')[-overlap_size:])

                # Start new segment with the overlap text
                current_segment = overlap_sentences + " " + sentence
            else:
                # Add to current segment
                if current_segment:
                    current_segment += " " + sentence
                else:
                    current_segment = sentence

        # Add the last segment if not empty
        if current_segment.strip():
            segment_id += 1
            segments.append({
                "id": f"{document_name}_segment_{segment_id}",
                "content": current_segment.strip(),
                "position": segment_id
            })

        return segments

    def _enhance_with_context(self, segments, full_document):
        """Add contextual information to each segment using LLM"""
        enhanced_segments = []

        system_message = {"text": """
        You are a document context specialist. Your task is to briefly describe how a text chunk 
        fits within a larger document. Provide 2-3 sentences that:
        1. Identify the key information in this segment
        2. Explain how this segment relates to the broader content
        Be concise and specific.
        Provide you answer in the source language.
        """}

        for segment in segments:
            try:
                user_message = {"role": "user", "content": [{"text": f"""
                <document>
                {full_document}
                </document>

                <chunk>
                {segment["content"]}
                </chunk>

                Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
                Answer only with the succinct context and nothing else.
                """}]}

                response = self.bedrock_client.converse(
                    modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
                    messages=[user_message],
                    system=[system_message],
                    inferenceConfig={"temperature": 0.0, "topP": 0.5},
                )

                context_description = response['output']['message']['content'][0]['text'].strip()
                segment["enhanced_content"] = f"Context: {context_description}\n\nContent: {segment['content']}"
                enhanced_segments.append(segment)

            except Exception as e:
                print(f"Error enhancing segment {segment['id']}: {e}")
                # Use original content as fallback
                segment["enhanced_content"] = segment["content"]
                enhanced_segments.append(segment)

        return enhanced_segments

    def _get_embedding(self, text):
        """Generate vector embedding for text"""
        try:
            response = self.bedrock_client.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=json.dumps({"inputText": text})
            )

            response_body = json.loads(response['body'].read())
            return response_body.get('embedding')

        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def _index_segments(self, segments, document_name, source_uri):
        """Index segments to OpenSearch"""
        batch_size = 20
        current_batch = []
        indexed_count = 0

        for segment in segments:
            # Use enhanced content if available
            content_to_index = segment.get("enhanced_content", segment["content"])

            # Generate embedding
            embedding = self._get_embedding(content_to_index)
            if not embedding:
                print(f"Skipping segment {segment['id']} - embedding failed")
                continue

            # Create document for indexing with simple metadata
            doc = {
                "content": content_to_index,
                "content_embedding": embedding,
                "metadata": {
                    "source": source_uri,
                    "doc_id": document_name,
                    "chunk_id": segment["id"],
                    "timestamp": datetime.now().isoformat()
                }
            }

            current_batch.append(doc)

            # Process batch if reached batch size
            if len(current_batch) >= batch_size:
                success = self._bulk_index(current_batch)
                if success:
                    indexed_count += len(current_batch)
                current_batch = []

        # Process any remaining documents
        if current_batch:
            success = self._bulk_index(current_batch)
            if success:
                indexed_count += len(current_batch)

        return indexed_count

    def _bulk_index(self, documents):
        """Index batch of documents to OpenSearch"""
        if not documents:
            return True

        url = f"{self.collection_endpoint}/_bulk"
        headers = {'Content-Type': 'application/x-ndjson'}

        # Prepare bulk request body
        bulk_body = ""
        for doc in documents:
            # Add action line
            action = {"index": {"_index": self.cr_index_name}}
            bulk_body += json.dumps(action) + "\n"

            # Add document line
            bulk_body += json.dumps(doc) + "\n"

        try:
            response = requests.post(
                url,
                auth=self.auth,
                headers=headers,
                data=bulk_body,
                verify=True
            )

            if response.status_code >= 400:
                print(f"Bulk indexing error: {response.text}")
                return False
            else:
                print(f"Successfully indexed {len(documents)} documents")
                return True

        except Exception as e:
            print(f"Bulk indexing exception: {e}")
            return False
