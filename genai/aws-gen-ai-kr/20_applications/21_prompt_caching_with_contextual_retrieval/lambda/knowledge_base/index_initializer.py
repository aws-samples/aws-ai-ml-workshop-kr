import boto3
import json
import os
import requests
import time
from requests_aws4auth import AWS4Auth


def handler(event, context):
    """Lambda handler that initializes OpenSearch indices"""
    print(f"Event received: {json.dumps(event)}")

    # Get environment variables
    collection_endpoint = os.environ['COLLECTION_ENDPOINT']
    cr_index_name = os.environ['CR_INDEX_NAME']
    kb_index_name = os.environ['KB_INDEX_NAME']
    region = os.environ.get('REGION') or os.environ.get('AWS_REGION')

    # Create AWS4Auth for authentication
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        'aoss',  # Service name for OpenSearch Serverless
        session_token=credentials.token
    )

    try:
        # Create both indices
        create_index(collection_endpoint, cr_index_name, awsauth)
        create_index(collection_endpoint, kb_index_name, awsauth)

        print(f"Successfully created/verified indices: {cr_index_name}, {kb_index_name}")

        # Wait for changes to propagate
        time.sleep(60)

        # Return success response for CloudFormation custom resource
        return {
            'statusCode': 200,
            'body': 'Indices successfully initialized'
        }

    except Exception as e:
        print(f"Error initializing indices: {str(e)}")

        # Return error response - but don't fail the deployment
        # This makes the custom resource more robust
        return {
            'statusCode': 500,
            'body': f'Error initializing indices: {str(e)}'
        }


def create_index(endpoint, index_name, auth):
    """Create OpenSearch index with mapping if it doesn't exist"""
    url = f"{endpoint}/{index_name}"
    headers = {'Content-Type': 'application/json'}

    # Define the index mapping
    mapping = {
        "settings": {
            "index.knn": True,
            "index.knn.algo_param.ef_search": 512
        },
        "mappings": {
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "content_embedding": {
                    "type": "knn_vector",
                    "dimension": 1024,
                    "method": {
                        "engine": "faiss",
                        "name": "hnsw",
                        "parameters": {
                            "ef_construction": 512,
                            "m": 16
                        },
                        "space_type": "l2"
                    }
                }
            }
        }
    }

    # Check if index exists
    try:
        response = requests.head(url, auth=auth, verify=True)
        if response.status_code == 200:
            print(f"Index {index_name} already exists")
            return
    except Exception as e:
        print(f"Error checking if index exists: {str(e)}")

    # Create the index with mapping
    try:
        response = requests.put(url, auth=auth, headers=headers, json=mapping, verify=True)
        print(f"Index creation status code: {response.status_code}")
        print(f"Response body: {response.text}")
        response.raise_for_status()
        print(f"Created index {index_name} with mapping")
    except Exception as e:
        print(f"Error creating index: {str(e)}")
        raise