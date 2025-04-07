import json
import boto3
import base64
import os
import uuid
from datetime import datetime

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
sqs = boto3.client('sqs')


def handler(event, context):
    """
    Lambda function to handle file uploads from the UI.
    Stores the file in S3 and tracks it in DynamoDB.
    """
    try:
        # Check if this is an OPTIONS request (CORS preflight)
        if event.get('httpMethod') == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': get_cors_headers(),
                'body': ''
            }

        # Parse the request
        body = json.loads(event['body'])
        file_content_base64 = body.get('file')
        file_name = body.get('fileName')
        file_type = body.get('fileType')

        if not file_content_base64 or not file_name:
            return {
                'statusCode': 400,
                'headers': get_cors_headers(),
                'body': json.dumps({'error': 'Missing file content or file name'})
            }

        # Validate file type (only PDFs allowed)
        if file_type != 'application/pdf' and not file_name.lower().endswith('.pdf'):
            return {
                'statusCode': 400,
                'headers': get_cors_headers(),
                'body': json.dumps({'error': 'Only PDF files are supported'})
            }

        # Decode file content from base64
        try:
            file_content = base64.b64decode(
                file_content_base64.split(',')[1] if ',' in file_content_base64 else file_content_base64)
        except Exception as e:
            return {
                'statusCode': 400,
                'headers': get_cors_headers(),
                'body': json.dumps({'error': f'Invalid file content: {str(e)}'})
            }

        # Generate a unique file key (preserve original name but add UUID)
        document_bucket = os.environ.get('DOCUMENT_BUCKET')
        file_base = os.path.splitext(file_name)[0]
        file_ext = os.path.splitext(file_name)[1].lower()
        file_key = f"{file_base}-{uuid.uuid4()}{file_ext}"

        # Save file to S3
        s3.put_object(
            Bucket=document_bucket,
            Key=file_key,
            Body=file_content,
            ContentType='application/pdf'
        )

        # Get the table for tracking documents
        document_table = dynamodb.Table(os.environ.get('DOCUMENT_TABLE'))

        # Create entry in DynamoDB
        document_id = str(uuid.uuid4())
        document_table.put_item(
            Item={
                'id': document_id,
                'fileName': file_name,
                's3Key': file_key,
                's3Bucket': document_bucket,
                'uploadTime': datetime.utcnow().isoformat(),
                'status': 'UPLOADED',
                'statusMessage': 'Document uploaded successfully, waiting for processing',
                'opensearchStatus': {
                    'kb_index': 'PENDING',
                    'cr_index': 'PENDING'
                },
                's3Url': f"s3://{document_bucket}/{file_key}",
                'tokenUsage': {
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'cache_write_input_tokens': 0
                }
            }
        )

        # Trigger processing of the document
        processing_queue_url = os.environ.get('PROCESSING_QUEUE_URL')

        sqs.send_message(
            QueueUrl=processing_queue_url,
            MessageBody=json.dumps({
                'documentId': document_id,
                's3Bucket': document_bucket,
                's3Key': file_key,
                'fileName': file_name
            })
        )

        # Return success response
        return {
            'statusCode': 200,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'documentId': document_id,
                'fileName': file_name,
                's3Url': f"s3://{document_bucket}/{file_key}",
                'status': 'UPLOADED'
            })
        }

    except Exception as e:
        print(f"Error processing upload: {str(e)}")
        return {
            'statusCode': 500,
            'headers': get_cors_headers(),
            'body': json.dumps({'error': str(e)})
        }


def get_cors_headers():
    """Return CORS headers for all responses"""
    return {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Requested-With',
        'Access-Control-Allow-Methods': 'OPTIONS,POST'
    }