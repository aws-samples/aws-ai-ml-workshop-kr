import json
import boto3
import os
from boto3.dynamodb.conditions import Key
from decimal import Decimal

dynamodb = boto3.resource('dynamodb')
bedrock = boto3.client('bedrock')


# Custom JSON encoder to handle Decimal types
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)


def handler(event, context):
    """
    Lambda function to get the status of uploaded documents.
    Also checks Bedrock ingestion job status for INGESTING documents.
    """
    try:
        # Get the document table
        document_table = dynamodb.Table(os.environ.get('DOCUMENT_TABLE'))

        # Check for query parameters
        query_params = event.get('queryStringParameters', {})
        if query_params and query_params.get('documentId'):
            # Get a specific document
            document_id = query_params.get('documentId')
            response = document_table.get_item(
                Key={'id': document_id}
            )
            items = [response.get('Item')] if 'Item' in response else []
        else:
            # Get all documents, sorted by upload time (newest first)
            response = document_table.scan()
            items = response.get('Items', [])
            items.sort(key=lambda x: x.get('uploadTime', ''), reverse=True)

        # Update ingestion status for INGESTING documents
        updated_items = []
        for item in items:
            # Only check ingestion status for documents that are INGESTING
            if item.get('status') == 'INGESTING' and item.get('ingestionJobId'):
                try:
                    # Get ingestion job status from Bedrock
                    ingestion_job_response = bedrock.get_ingestion_job(
                        knowledgeBaseId=os.environ.get('KNOWLEDGE_BASE_ID'),
                        dataSourceId=os.environ.get('DATA_SOURCE_ID'),
                        ingestionJobId=item.get('ingestionJobId')
                    )

                    job_status = ingestion_job_response.get('status')

                    # If job status has changed, update in DynamoDB
                    if job_status != item.get('ingestionStatus'):
                        update_expr = "SET ingestionStatus = :status"
                        expr_attrs = {':status': job_status}

                        # If job is complete, update document status
                        if job_status == 'COMPLETE':
                            update_expr += ", #st = :docstatus, statusMessage = :msg"
                            expr_attrs[':docstatus'] = 'COMPLETED'
                            expr_attrs[':msg'] = 'Document has been successfully ingested and is ready for querying'

                        # If job failed, update document status
                        elif job_status == 'FAILED':
                            update_expr += ", #st = :docstatus, statusMessage = :msg"
                            expr_attrs[':docstatus'] = 'ERROR'
                            err_msg = ingestion_job_response.get('failureReason', 'Unknown error during ingestion')
                            expr_attrs[':msg'] = f'Ingestion failed: {err_msg}'

                        # Update item in DynamoDB if needed
                        if ':docstatus' in expr_attrs:
                            document_table.update_item(
                                Key={'id': item['id']},
                                UpdateExpression=update_expr,
                                ExpressionAttributeValues=expr_attrs,
                                ExpressionAttributeNames={'#st': 'status'}
                            )

                            # Also update our local copy for the response
                            item['ingestionStatus'] = job_status
                            item['status'] = expr_attrs[':docstatus']
                            item['statusMessage'] = expr_attrs[':msg']

                except Exception as e:
                    print(f"Error checking ingestion job status: {str(e)}")

            updated_items.append(item)

        # Return the document information using custom JSON encoder to handle Decimal types
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'OPTIONS,GET'
            },
            'body': json.dumps({
                'documents': updated_items
            }, cls=DecimalEncoder)  # Use the custom encoder here
        }

    except Exception as e:
        print(f"Error getting document status: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'OPTIONS,GET'
            },
            'body': json.dumps({'error': str(e)})
        }


def handler_options(event, context):
    """Handle OPTIONS requests for CORS"""
    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'OPTIONS,GET'
        },
        'body': ''
    }