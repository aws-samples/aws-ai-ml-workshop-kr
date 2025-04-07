import boto3
import json
import os


def handler(event, context):
    """Lambda handler that scans for PDFs and adds them to SQS queue"""
    print(f"Event received: {json.dumps(event)}")

    # Get environment variables
    pdf_bucket = os.environ['PDF_BUCKET']
    queue_url = os.environ['SQS_QUEUE_URL']

    # Initialize clients
    s3 = boto3.client('s3')
    sqs = boto3.client('sqs')

    try:
        # List objects in bucket
        response = s3.list_objects_v2(Bucket=pdf_bucket)

        if 'Contents' not in response:
            print(f"No files found in bucket {pdf_bucket}")
            return {
                'statusCode': 200,
                'body': 'No files found in bucket'
            }

        # Filter for PDF files
        pdf_files = [obj['Key'] for obj in response['Contents']
                     if obj['Key'].lower().endswith('.pdf')]

        if not pdf_files:
            print(f"No PDF files found in bucket {pdf_bucket}")
            return {
                'statusCode': 200,
                'body': 'No PDF files found in bucket'
            }

        # Add each PDF to queue
        enqueued_count = 0
        for pdf_file in pdf_files:
            # Send message to queue
            sqs.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps({
                    'bucket': pdf_bucket,
                    'key': pdf_file
                })
            )
            enqueued_count += 1
            print(f"Enqueued {pdf_file}")

        print(f"Successfully enqueued {enqueued_count} PDF files for processing")

        return {
            'statusCode': 200,
            'body': f'Successfully enqueued {enqueued_count} PDF files for processing'
        }

    except Exception as e:
        print(f"Error queuing PDF files: {str(e)}")
        return {
            'statusCode': 500,
            'body': f'Error queuing PDF files: {str(e)}'
        }