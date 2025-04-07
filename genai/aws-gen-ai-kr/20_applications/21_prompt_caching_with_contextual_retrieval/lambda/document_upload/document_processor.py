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

lambda_client = boto3.client('lambda', region_name='us-west-2')
dynamodb = boto3.resource('dynamodb', region_name='us-west-2')


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
    document_table_name = os.environ.get('DOCUMENT_TABLE')
    kb_sync_lambda_arn = os.environ.get('KB_SYNC_LAMBDA_ARN')
    knowledge_base_id = os.environ.get('KNOWLEDGE_BASE_ID')

    # Initialize DynamoDB connection
    document_table = dynamodb.Table(document_table_name) if document_table_name else None

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
            bucket = message['s3Bucket']
            key = message['s3Key']
            document_id = message.get('documentId')

            print(f"Processing document {key} from bucket {bucket}")

            # First, update status to PROCESSING (direct update)
            if document_table and document_id:
                try:
                    document_table.update_item(
                        Key={'id': document_id},
                        UpdateExpression="SET #status = :status, lastUpdated = :time, statusMessage = :message",
                        ExpressionAttributeNames={'#status': 'status'},
                        ExpressionAttributeValues={
                            ':status': 'PROCESSING',
                            ':time': datetime.utcnow().isoformat(),
                            ':message': "Document is being processed"
                        }
                    )
                    print(f"Updated document {document_id} status to PROCESSING")
                except Exception as e:
                    print(f"Error updating document status to PROCESSING: {str(e)}")

            # Get file from S3 (needed for both KB sync and OpenSearch processing)
            s3 = boto3.client('s3')
            response = s3.get_object(Bucket=bucket, Key=key)
            pdf_content = response['Body'].read()
            s3_uri = f"s3://{bucket}/{key}"

            # 1. Start KB sync job first
            print(f"Starting Bedrock Knowledge Base ingestion job for document {document_id}")
            kb_sync_event = {
                'RequestType': 'Create',
                'ResponseURL': 'https://dummy-url/not-used',
                'StackId': 'direct-invocation',
                'RequestId': document_id,
                'LogicalResourceId': f'DocumentIngestion-{document_id}',
                'ResourceProperties': {
                    'DocumentId': document_id,
                    'S3Bucket': bucket,
                    'S3Key': key
                }
            }

            lambda_client.invoke(
                FunctionName=kb_sync_lambda_arn,
                InvocationType='Event',  # Asynchronous invocation
                Payload=json.dumps(kb_sync_event)
            )

            # Update status to indicate KB sync has been triggered
            update_opensearch_status(document_table, document_id, 'kb_index', 'INGESTING',
                                     "Bedrock Knowledge Base ingestion job started")

            # 2. Process document for contextual retrieval in parallel
            print(f"Starting OpenSearch contextual retrieval processing for document {document_id}")

            # Set CR index status to PROCESSING
            update_opensearch_status(document_table, document_id, 'cr_index', 'PROCESSING',
                                     "Processing document for contextual retrieval")

            # Process the document
            segments_indexed, token_usage = processor.process_document(pdf_content, key, s3_uri, document_id)

            print(f"Successfully processed {key}: indexed {segments_indexed} segments")
            print(f"Total token usage: {token_usage}")

            # Update opensearchStatus for cr_index and set token usage at the same time
            try:
                document_table.update_item(
                    Key={'id': document_id},
                    UpdateExpression="SET opensearchStatus.cr_index = :status, tokenUsage = :tokens, lastUpdated = :time",
                    ExpressionAttributeValues={
                        ':status': 'COMPLETED',
                        ':tokens': token_usage,
                        ':time': datetime.utcnow().isoformat()
                    }
                )
                print(f"Updated CR index status to COMPLETED and set token usage: {token_usage}")
            except Exception as e:
                print(f"Error updating CR index status and token usage: {str(e)}")
                # Fallback to separate updates if the combined update fails
                update_opensearch_status(document_table, document_id, 'cr_index', 'COMPLETED',
                                         f"Successfully indexed {segments_indexed} segments in contextual retrieval index.")

            # Check if both indexes are complete - one might be if KB sync finished quickly
            check_and_update_overall_status(document_table, document_id)

        except Exception as e:
            print(f"Error processing message: {str(e)}")
            # Update status to error if document_id is available (direct update)
            if document_id and document_table:
                try:
                    document_table.update_item(
                        Key={'id': document_id},
                        UpdateExpression="SET #status = :status, lastUpdated = :time, statusMessage = :message",
                        ExpressionAttributeNames={'#status': 'status'},
                        ExpressionAttributeValues={
                            ':status': 'ERROR',
                            ':time': datetime.utcnow().isoformat(),
                            ':message': f"Error processing document: {str(e)}"
                        }
                    )
                    print(f"Updated document {document_id} status to ERROR")
                except Exception as update_error:
                    print(f"Error updating document error status: {str(update_error)}")
            # Don't raise exception - let SQS delete the message to avoid retries

def update_document_status(table, document_id, status, token_usage, message=None):
    """Update document status in DynamoDB"""
    if not table or not document_id:
        print(f"Cannot update document status: Missing table or document ID")
        return

    update_expression = "SET #status = :status, lastUpdated = :time"
    expression_attribute_names = {'#status': 'status'}
    expression_attribute_values = {
        ':status': status,
        ':time': datetime.utcnow().isoformat(),
    }

    # Add token usage if provided and not None
    if token_usage is not None and token_usage:
        print(f"Token usage before update for document {document_id}: {token_usage}")
        update_expression += ", tokenUsage = :tokenUsage"
        expression_attribute_names['#tokenUsage'] = 'tokenUsage'
        expression_attribute_values[':tokenUsage'] = token_usage

    if message:
        update_expression += ", statusMessage = :message"
        expression_attribute_values[':message'] = message

    try:
        table.update_item(
            Key={'id': document_id},
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expression_attribute_names,
            ExpressionAttributeValues=expression_attribute_values
        )
        print(f"Updated document {document_id} status to {status} with token usage: {token_usage}")
    except Exception as e:
        print(f"Error updating document status: {str(e)}")

def update_opensearch_status(table, document_id, index_type, status, message=None):
    """Update the OpenSearch status for a specific index type"""
    try:
        update_expression = "SET opensearchStatus.#indexType = :status, lastUpdated = :time"
        expression_attribute_names = {'#indexType': index_type}
        expression_attribute_values = {
            ':status': status,
            ':time': datetime.utcnow().isoformat()
        }

        if message:
            update_expression += ", #statusMsg = :message"
            expression_attribute_names['#statusMsg'] = 'statusMessage'
            expression_attribute_values[':message'] = message

        table.update_item(
            Key={'id': document_id},
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expression_attribute_names,
            ExpressionAttributeValues=expression_attribute_values
        )

        print(f"Updated document {document_id} OpenSearch status for {index_type} to {status}")
    except Exception as e:
        print(f"Error updating OpenSearch status: {str(e)}")


def check_and_update_overall_status(table, document_id):
    """Check OpenSearch status and update overall status if both are complete"""
    try:
        # Get current document
        response = table.get_item(Key={'id': document_id})
        if 'Item' not in response:
            print(f"Document {document_id} not found")
            return

        document = response['Item']
        opensearch_status = document.get('opensearchStatus', {})

        # Check if both indexes are complete
        cr_index_status = opensearch_status.get('cr_index', 'PENDING')
        kb_index_status = opensearch_status.get('kb_index', 'PENDING')

        print(f"Current status - CR index: {cr_index_status}, KB index: {kb_index_status}")

        # If both are complete, update the overall status
        if cr_index_status == 'COMPLETED' and kb_index_status == 'COMPLETED':
            # Direct update without touching token usage
            try:
                table.update_item(
                    Key={'id': document_id},
                    UpdateExpression="SET #status = :status, lastUpdated = :time, statusMessage = :message",
                    ExpressionAttributeNames={'#status': 'status'},
                    ExpressionAttributeValues={
                        ':status': 'COMPLETED',
                        ':time': datetime.utcnow().isoformat(),
                        ':message': "Document has been fully processed and is ready for use"
                    }
                )
                print(f"Document {document_id} is now fully processed")
            except Exception as e:
                print(f"Error updating final completion status: {str(e)}")

        # If either has errored, update the overall status to ERROR
        elif 'ERROR' in [cr_index_status, kb_index_status]:
            # Direct update for error status
            try:
                table.update_item(
                    Key={'id': document_id},
                    UpdateExpression="SET #status = :status, lastUpdated = :time, statusMessage = :message",
                    ExpressionAttributeNames={'#status': 'status'},
                    ExpressionAttributeValues={
                        ':status': 'ERROR',
                        ':time': datetime.utcnow().isoformat(),
                        ':message': "Error during document processing"
                    }
                )
                print(f"Document {document_id} processing failed")
            except Exception as e:
                print(f"Error updating error status: {str(e)}")

        # Otherwise, keep the current status
        else:
            print(f"Document {document_id} is still processing")

    except Exception as e:
        print(f"Error checking and updating status: {str(e)}")


class DocumentProcessor:
    """Handles PDF document processing including extraction, segmentation, and indexing"""

    def __init__(self, collection_endpoint, cr_index_name, region):
        self.collection_endpoint = collection_endpoint
        self.cr_index_name = cr_index_name
        self.region = region

        # Configuration parameters
        self.segment_size = 2000  # Size of text segments
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

    def process_document(self, pdf_content, document_name, source_uri, document_id=None):
        """Process a PDF document from content bytes"""
        # Extract text from PDF
        document_text = self._extract_text(pdf_content)
        if not document_text:
            print(f"No text extracted from {document_name}")
            return 0, {"input_tokens": 0, "output_tokens": 0,
                       "cache_read_input_tokens": 0, "cache_write_input_tokens": 0}

        # Create segments from document text
        segments = self._create_segments(document_text, document_name)
        print(f"Created {len(segments)} segments from {document_name}")

        # Token usage tracking
        token_usage = {"input_tokens": 0, "output_tokens": 0,
                       "cache_read_input_tokens": 0, "cache_write_input_tokens": 0}

        # Add contextual information if enabled
        if self.enable_context:
            segments, context_token_usage = self._enhance_with_context(segments, document_text)
            token_usage["input_tokens"] += context_token_usage["input_tokens"]
            token_usage["output_tokens"] += context_token_usage["output_tokens"]
            token_usage["cache_read_input_tokens"] += context_token_usage["cache_read_input_tokens"]
            token_usage["cache_write_input_tokens"] += context_token_usage["cache_write_input_tokens"]
        # Index segments to OpenSearch
        indexed_count = self._index_segments(segments, document_name, source_uri)

        return indexed_count, token_usage

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
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_read_input_tokens = 0
        total_cache_write_input_tokens = 0

        system_message = {"text": f"""
        You are a document context specialist. Your task is to briefly describe how a text chunk 
        fits within a larger document. Provide 2-3 sentences that:
        1. Identify the key information in this segment
        2. Explain how this segment relates to the broader content
        Be concise and specific.
        Provide you answer in the document language.
        <document>
        {full_document}
        </document>
        """}

        for segment in segments:
            try:
                user_message = {"role": "user", "content": [{"text": f"""
                <chunk>
                {segment["content"]}
                </chunk>

                Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
                Answer only with the succinct context and nothing else.
                """}]}

                response = self.bedrock_client.converse(
                    modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
                    messages=[user_message],
                    system=[system_message, {"cachePoint": {"type": "default"}}],
                    inferenceConfig={"temperature": 0.0, "topP": 0.5},
                )

                context_description = response['output']['message']['content'][0]['text'].strip()

                # Track tokens for this segment
                segment["token_usage"] = response['usage']
                print(f"Token usage for segment: {response['usage']}")

                # Extract and convert usage from the response
                input_tokens = response['usage'].get('inputTokens', 0)
                output_tokens = response['usage'].get('outputTokens', 0)
                cache_read = response['usage'].get('cacheReadInputTokens', 0)
                cache_write = response['usage'].get('cacheWriteInputTokens', 0)

                # Add to totals
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_cache_read_input_tokens += cache_read
                total_cache_write_input_tokens += cache_write

                segment["enhanced_content"] = f"Context: {context_description}\n\nContent: {segment['content']}"
                enhanced_segments.append(segment)

            except Exception as e:
                print(f"Error enhancing segment {segment['id']}: {e}")
                # Use original content as fallback
                segment["enhanced_content"] = segment["content"]
                segment["token_usage"] = {"inputTokens": 0, "outputTokens": 0}
                enhanced_segments.append(segment)

        # Return segments and token usage totals
        token_usage = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cache_read_input_tokens": total_cache_read_input_tokens,
            "cache_write_input_tokens": total_cache_write_input_tokens
        }

        print(f"Total token usage: {token_usage}")
        return enhanced_segments, token_usage

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