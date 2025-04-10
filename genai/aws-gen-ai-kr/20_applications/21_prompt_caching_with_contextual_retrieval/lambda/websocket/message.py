import boto3
import os
import json
from botocore.exceptions import BotoCoreError, ClientError
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from typing import List, Dict

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['CONNECTIONS_TABLE'])

# Default model if none specified
DEFAULT_MODEL = 'anthropic.claude-3-5-sonnet-20241022-v2:0'

# Map of allowed models - this is for security to prevent arbitrary model injection
ALLOWED_MODELS = {
    'amazon.nova-pro-v1:0': 'arn:aws:bedrock:us-west-2::foundation-model/amazon.nova-pro-v1:0',
    'amazon.nova-lite-v1:0': 'arn:aws:bedrock:us-west-2::foundation-model/amazon.nova-lite-v1:0',
    'anthropic.claude-3-7-sonnet-20250219-v1:0': 'arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-7-sonnet-20250219-v1:0',
    'anthropic.claude-3-5-sonnet-20241022-v2:0': 'arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0',
    'anthropic.claude-3-5-haiku-20241022-v1:0': 'arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-5-haiku-20241022-v1:0'
}


def _generate_embeddings(text: str) -> List[float]:
    """Generate embeddings using Amazon Bedrock"""
    try:
        bedrock_runtime = boto3.client('bedrock-runtime', region_name=os.environ['REGION'])
        payload = {"inputText": text}

        response = bedrock_runtime.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps(payload)
        )

        response_body = json.loads(response['body'].read().decode())
        return response_body.get('embedding', [])

    except Exception as e:
        print(f"Embedding generation error: {str(e)}")
        return None


def _create_vector_query(query: str, top_n: int) -> Dict:
    """Create pure vector search query"""
    embedding = _generate_embeddings(query)

    if not embedding:
        raise ValueError("Failed to generate embedding for vector search")

    return {
        "size": top_n,
        "_source": ["content", "metadata"],
        "query": {
            "knn": {
                "content_embedding": {
                    "vector": embedding,
                    "k": top_n
                }
            }
        }
    }


def _create_keyword_query(query: str, top_n: int) -> Dict:
    """Create pure keyword (BM25) search query"""
    return {
        "size": top_n,
        "_source": ["content", "metadata"],
        "query": {
            "match": {
                "content": {
                    "query": query,
                    "operator": "or"
                }
            }
        }
    }


def _create_hybrid_query(query: str, top_n: int) -> Dict:
    """Create hybrid search query combining semantic and BM25"""
    embedding = _generate_embeddings(query)

    return {
        "size": top_n,
        "_source": ["content", "metadata"],
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "content_embedding": {
                                "vector": embedding,
                                "k": top_n,
                                "boost": 0.7
                            }
                        }
                    },
                    {
                        "match": {
                            "content": {
                                "query": query,
                                "operator": "or",
                                "boost": 0.3
                            }
                        }
                    }
                ]
            }
        }
    }


def execute_search(client, index_name, query_body):
    """Execute a search query and return results"""
    try:
        response = client.search(
            index=index_name,
            body=query_body
        )

        # Process results
        return [{
            "score": hit["_score"],
            "content": hit["_source"]["content"],
            "metadata": hit["_source"]["metadata"],
            "id": hit["_id"]
        } for hit in response["hits"]["hits"]]
    except Exception as e:
        print(f"Search execution error: {str(e)}")
        return []


def rank_fusion(results_lists, k=60):
    """
    Implements Reciprocal Rank Fusion to combine multiple result lists

    Args:
        results_lists: List of result lists, where each result has 'id' and 'score'
        k: Constant to prevent division by zero and reduce impact of high rankings (default: 60)

    Returns:
        Combined and re-ranked list of results
    """
    # Create a dictionary to store combined scores
    fusion_scores = {}

    # Process each result list
    for results in results_lists:
        # Create a rank mapping for this result list
        for rank, result in enumerate(results):
            doc_id = result['id']
            # RRF formula: 1 / (k + rank)
            score = 1.0 / (k + rank)

            # Add to fusion scores
            if doc_id in fusion_scores:
                fusion_scores[doc_id]['score'] += score
                # Keep the result data from the first occurrence
            else:
                fusion_scores[doc_id] = {
                    'score': score,
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'id': doc_id
                }

    # Convert dictionary to list and sort by fusion score
    fused_results = list(fusion_scores.values())
    fused_results.sort(key=lambda x: x['score'], reverse=True)

    return fused_results


def rank_fusion_search(query: str, top_n: int, os_endpoint: str, os_index: str) -> List[Dict]:
    """Perform rank fusion search combining vector and keyword searches"""
    try:
        # Get AWS credentials
        credentials = boto3.Session().get_credentials()
        region = os.environ['REGION']

        # Create authentication object
        auth = AWSV4SignerAuth(credentials, region, 'aoss')

        # Create OpenSearch client
        client = OpenSearch(
            hosts=[{'host': os_endpoint.replace("https://", ""), 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30
        )

        # Generate query bodies - request more results for fusion
        expanded_top_n = top_n * 3
        vector_query = _create_vector_query(query, expanded_top_n)
        keyword_query = _create_keyword_query(query, expanded_top_n)

        # Execute both searches
        vector_results = execute_search(client, os_index, vector_query)
        keyword_results = execute_search(client, os_index, keyword_query)

        # Combine results using rank fusion
        fused_results = rank_fusion([vector_results, keyword_results])

        # Return top_n results
        return fused_results[:top_n]

    except Exception as e:
        print(f"Rank fusion search error: {str(e)}")
        # Fallback to regular hybrid search if rank fusion fails
        try:
            hybrid_query = _create_hybrid_query(query, top_n)
            client = OpenSearch(
                hosts=[{'host': os_endpoint.replace("https://", ""), 'port': 443}],
                http_auth=auth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=30
            )
            return execute_search(client, os_index, hybrid_query)
        except:
            return []


def _rerank_documents(query, documents, top_n):
    """Rerank documents using Cohere Rerank model"""
    bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=os.environ['REGION'])
    model_package_arn = "arn:aws:bedrock:us-west-2::foundation-model/cohere.rerank-v3-5:0"
    text_sources = []
    for doc in documents:
        text_sources.append({
            "type": "INLINE",
            "inlineDocumentSource": {
                "type": "TEXT",
                "textDocument": {
                    "text": doc,
                }
            }
        })

    try:
        response = bedrock_agent_runtime.rerank(
            queries=[
                {
                    "type": "TEXT",
                    "textQuery": {
                        "text": query
                    }
                }
            ],
            sources=text_sources,
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "numberOfResults": top_n,
                    "modelConfiguration": {
                        "modelArn": model_package_arn,
                    }
                }
            }
        )

        results = {
            "results": [
                {
                    "index": result['index'],
                    "relevance_score": result['relevanceScore'],
                } for result in response['results']
            ]
        }
        return results

    except Exception as e:
        print(f"Reranking error: {str(e)}")
        return []


def enhanced_search(query: str, top_n: int, os_endpoint: str, os_index: str) -> List[Dict]:
    """Enhanced search with rank fusion and reranking"""
    # 1. Rank fusion search
    raw_results = rank_fusion_search(query, top_n * 2, os_endpoint, os_index)  # Get extra results for reranking
    results = []
    for hit in raw_results:
        result = {
            "content": hit["content"],
            "score": hit['score'],
            "metadata": hit['metadata']
        }
        results.append(result)

    # 2. Extract content for reranking
    documents = [res['content'] for res in raw_results]

    # 3. Rerank with Cohere
    reranked = _rerank_documents(query, documents, top_n)

    reranked_results = []
    # 4. Map back to original results
    for reranked_doc in reranked["results"]:
        idx = reranked_doc["index"]
        score = reranked_doc["relevance_score"]
        reranked_results.append({
            "content": results[idx]["content"],
            "score": score,
            "metadata": results[idx]["metadata"]
        })

    return reranked_results


def send_to_connection(apigw_client, connection_id, data):
    """Send data to the WebSocket connection."""
    try:
        apigw_client.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps(data)
        )
    except Exception as e:
        print(f"Error sending message to connection {connection_id}: {str(e)}")


def extract_source_url(citation):
    """Extract source URL from citation based on location type."""
    try:
        if 'location' in citation:
            location = citation['location']

            # Handle S3 location
            if 'type' in location and location['type'] == 'S3' and 's3Location' in location:
                return location['s3Location']['uri']

            # Handle web location
            elif 'type' in location and location['type'] == 'WEB' and 'webLocation' in location:
                return location['webLocation']['url']

            # Try to find any location with uri/url field
            else:
                for loc_key in location:
                    if isinstance(location[loc_key], dict):
                        if 'uri' in location[loc_key]:
                            return location[loc_key]['uri']
                        elif 'url' in location[loc_key]:
                            return location[loc_key]['url']
    except Exception as e:
        print(f"Error extracting source URL: {str(e)}")

    return None


def handle_contextual_retrieval(query, model_id, connection_id, apigw_management):
    """Handle search using contextual retrieval with OpenSearch"""
    # Get environment variables for OpenSearch
    os_endpoint = os.environ['OPENSEARCH_ENDPOINT']
    os_index = os.environ['OPENSEARCH_INDEX']
    model_id = model_id or DEFAULT_MODEL
    region = os.environ['REGION']
    language = os.environ.get('RESPONSE_LANGUAGE', 'English')

    try:
        # Get context from enhanced search (now with rank fusion)
        context_results = enhanced_search(query, 5, os_endpoint, os_index)

        # Extract sources for citation tracking
        sources = []
        context_text = ""
        source_map = {}  # Maps source metadata to citation numbers

        # Process search results to build context and extract sources
        for i, result in enumerate(context_results):
            context_text += f"{result['content']}\n\n"

            # Get source metadata
            metadata = result.get('metadata', {})
            source_url = metadata.get('source', metadata.get('url', f"source-{i}"))

            if source_url:
                if source_url not in source_map:
                    citation_num = len(source_map) + 1
                    source_map[source_url] = citation_num
                    sources.append({
                        'sourceId': str(citation_num),
                        'sourceUrl': source_url,
                    })

        # Set up system prompt
        system_prompt = f"""You are a helpful chatbot who answers questions based on the provided context.
        Provide your answer by only using information from the given context.
        If the information to answer the question is not in the context, say that you don't have enough information.
        Include citation numbers [1], [2], etc. when referring to information from specific sources.
        Answer in {language}."""

        # Create Bedrock client for streaming
        bedrock_client = boto3.client('bedrock-runtime', region_name=region)

        # Create the message structure
        message = {
            'role': 'user',
            'content': [
                {
                    'text': f"<context>{context_text}</context>\n<user_query>{query}</user_query>"
                }
            ]
        }

        # Start streaming response
        response = bedrock_client.converse_stream(
            modelId=model_id,
            messages=[message],
            system=[{'text': system_prompt}],
            inferenceConfig={
                'maxTokens': 4096,
                'temperature': 0.5,
                'topP': 0.7,
            }
        )

        # Send citations first
        for source in sources:
            send_to_connection(apigw_management, connection_id, {
                'type': 'citation',
                'sourceId': source['sourceId'],
                'sourceUrl': source['sourceUrl'],
            })

        # Process streaming response
        for event in response['stream']:
            if 'contentBlockDelta' in event:
                if 'delta' in event['contentBlockDelta']:
                    text_chunk = event['contentBlockDelta']['delta']['text']
                    # Send the text chunk
                    send_to_connection(apigw_management, connection_id, {
                        'type': 'text',
                        'content': text_chunk
                    })

        # Send completion message with all source references
        send_to_connection(apigw_management, connection_id, {
            'type': 'complete',
            'sources': {source['sourceId']: source['sourceUrl'] for source in sources}
        })

    except Exception as e:
        error_message = str(e)
        print(f"Error processing request: {error_message}")
        send_to_connection(apigw_management, connection_id, {
            'type': 'error',
            'message': error_message
        })


def handle_knowledge_base(query, model_id, connection_id, apigw_management):
    """Handle search using the Bedrock Knowledge Base"""
    # Get the knowledge base details
    knowledge_base_id = os.environ['KNOWLEDGE_BASE_ID']
    region = os.environ['REGION']

    # Create Bedrock client
    client = boto3.client('bedrock-agent-runtime', region_name=region)

    try:
        print(f"Retrieving from KB: {knowledge_base_id} with model: {model_id}")
        print(f"Query: {query}")

        # Initiate streaming response from Bedrock
        response = client.retrieve_and_generate_stream(
            input={'text': query},
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': knowledge_base_id,
                    'modelArn': model_id
                }
            }
        )

        citation_count = 0
        source_map = {}  # Maps source URLs to citation numbers
        print(response)
        event_stream = response.get('stream')
        # Process streaming response
        for event in event_stream:
            print(event)
            # Stream text output
            if 'output' in event:
                text_chunk = event['output']['text']
                send_to_connection(apigw_management, connection_id, {
                    'type': 'text',
                    'content': text_chunk
                })

            # Process citations
            if 'citation' in event:
                try:
                    citations = event.get('citation', {}).get('citation', {}).get('retrievedReferences', [])
                    for citation in citations:
                        # Extract source URL
                        source_url = extract_source_url(citation)

                        if source_url:
                            # Only assign a new citation number if this source hasn't been seen before
                            if source_url not in source_map:
                                citation_count += 1
                                source_map[source_url] = citation_count

                                # Send the citation marker to be inserted in the text stream
                                citation_marker = f" [{citation_count}] "
                                send_to_connection(apigw_management, connection_id, {
                                    'type': 'text',
                                    'content': citation_marker
                                })

                                # Also send citation information for the document viewer
                                send_to_connection(apigw_management, connection_id, {
                                    'type': 'citation',
                                    'sourceId': str(citation_count),
                                    'sourceUrl': source_url
                                })
                except Exception as citation_error:
                    print(f"Error processing citation: {str(citation_error)}")

        # Send completion message with all source references
        send_to_connection(apigw_management, connection_id, {
            'type': 'complete',
            'sources': {str(num): url for url, num in source_map.items()}
        })

    except Exception as e:
        error_message = str(e)
        print(f"Error processing request: {error_message}")
        send_to_connection(apigw_management, connection_id, {
            'type': 'error',
            'message': error_message
        })


def handler(event, context):
    # Get connection ID
    connection_id = event['requestContext']['connectionId']

    # Parse message from client
    body = json.loads(event['body'])
    query = body.get('query', '')

    # Get model ID from request
    model_id = body.get('modelArn', DEFAULT_MODEL)

    # Security check: only use model if it's in our allowed list
    if model_id not in ALLOWED_MODELS:
        print(f"WARNING: Requested model {model_id} not in allowed list, using default")
        model_id = DEFAULT_MODEL

    # Get the model ARN
    model_arn = ALLOWED_MODELS[model_id]
    print(f"Using model: {model_id} with ARN: {model_arn}")

    # Get search method if provided
    search_method = body.get('searchMethod', 'opensearch')
    print(f"Using search method: {search_method}")

    # Get endpoint URL for sending messages
    domain = event['requestContext']['domainName']
    stage = event['requestContext']['stage']
    endpoint = f"https://{domain}/{stage}"

    # Set up API Gateway management client
    apigw_management = boto3.client(
        'apigatewaymanagementapi',
        endpoint_url=endpoint
    )

    # Use the appropriate search method based on user selection
    if search_method == 'contextual':
        handle_contextual_retrieval(query, model_id, connection_id, apigw_management)
    else:
        # Default to knowledge base for other methods (opensearch, neptune)
        handle_knowledge_base(query, model_id, connection_id, apigw_management)

    return {
        'statusCode': 200,
        'body': 'Streaming process completed'
    }