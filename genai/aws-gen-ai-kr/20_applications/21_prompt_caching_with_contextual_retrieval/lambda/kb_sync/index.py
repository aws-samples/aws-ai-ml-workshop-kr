import boto3
import os
import json
import cfnresponse
import time
from datetime import datetime


def handler(event, context):
    try:
        print(f"Starting KB sync handler with event: {event}")

        kb_id = os.environ['KNOWLEDGE_BASE_ID']
        ds_id = os.environ['DATA_SOURCE_ID']
        region = os.environ['REGION']
        document_table_name = os.environ.get('DOCUMENT_TABLE')

        # Get document ID from event
        document_id = None
        if 'ResourceProperties' in event and 'DocumentId' in event['ResourceProperties']:
            document_id = event['ResourceProperties']['DocumentId']
        elif 'RequestId' in event:
            document_id = event['RequestId']

        print(f"Processing for document ID: {document_id}")

        # Initialize DynamoDB if we have a document ID
        if document_id and document_table_name:
            dynamodb = boto3.resource('dynamodb', region_name=region)
            document_table = dynamodb.Table(document_table_name)
        else:
            document_table = None

        bedrock = boto3.client('bedrock-agent', region_name=region)

        if event['RequestType'] in ['Create', 'Update']:
            # Start ingestion job
            response = bedrock.start_ingestion_job(
                knowledgeBaseId=kb_id,
                dataSourceId=ds_id
            )

            ingestion_job_id = response['ingestionJob']['ingestionJobId']
            print(f"Started ingestion job: {ingestion_job_id}")

            # Update document record with ingestion job ID if we have a document ID
            if document_id and document_table:
                document_table.update_item(
                    Key={'id': document_id},
                    UpdateExpression="SET ingestionJobId = :jobId, ingestionStatus = :status",
                    ExpressionAttributeValues={
                        ':jobId': ingestion_job_id,
                        ':status': 'STARTED'
                    }
                )

            # Optional: Wait for job to complete (may timeout for large datasets)
            max_wait_time = 240  # seconds
            wait_interval = 10  # seconds
            elapsed_time = 0
            final_status = 'UNKNOWN'

            while elapsed_time < max_wait_time:
                job_status = bedrock.get_ingestion_job(
                    knowledgeBaseId=kb_id,
                    dataSourceId=ds_id,
                    ingestionJobId=ingestion_job_id
                )

                status = job_status['ingestionJob']['status']
                print(f"Job status: {status}")
                final_status = status

                # Update document status
                if document_id and document_table:
                    document_table.update_item(
                        Key={'id': document_id},
                        UpdateExpression="SET ingestionStatus = :status, lastUpdated = :timestamp",
                        ExpressionAttributeValues={
                            ':status': status,
                            ':timestamp': datetime.utcnow().isoformat()
                        }
                    )

                # If job completed or failed, update opensearchStatus for kb_index
                if status in ['COMPLETE', 'FAILED', 'STOPPED']:
                    if document_id and document_table:
                        update_opensearch_status(
                            document_table,
                            document_id,
                            'kb_index',
                            'COMPLETED' if status == 'COMPLETE' else 'ERROR',
                            f"Knowledge base ingestion job {status.lower()}"
                        )
                        # Check and update overall status
                        check_and_update_overall_status(document_table, document_id)
                    break

                time.sleep(wait_interval)
                elapsed_time += wait_interval

            # If timeout reached, update with status
            if elapsed_time >= max_wait_time and document_id and document_table:
                document_table.update_item(
                    Key={'id': document_id},
                    UpdateExpression="SET statusMessage = :msg",
                    ExpressionAttributeValues={
                        ':msg': f"Knowledge base ingestion job is still running (status: {final_status})"
                    }
                )

            cfnresponse.send(event, context, cfnresponse.SUCCESS, {
                'IngestionJobId': ingestion_job_id,
                'Status': final_status if elapsed_time < max_wait_time else 'STILL_RUNNING'
            })
        else:
            # Nothing to do for Delete
            cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
    except Exception as e:
        print(f"Error in KB sync: {str(e)}")
        # Update document status to error if possible
        if document_id and document_table:
            update_opensearch_status(
                document_table,
                document_id,
                'kb_index',
                'ERROR',
                f"Error in KB sync: {str(e)}"
            )
            # Update overall status
            document_table.update_item(
                Key={'id': document_id},
                UpdateExpression="SET #status = :status, statusMessage = :msg",
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={
                    ':status': 'ERROR',
                    ':msg': f"Error in KB sync: {str(e)}"
                }
            )
        cfnresponse.send(event, context, cfnresponse.FAILED, {'Error': str(e)})


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
            update_expression += ", statusMessage = :message"
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
            table.update_item(
                Key={'id': document_id},
                UpdateExpression="SET #status = :status, statusMessage = :msg, lastUpdated = :time",
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={
                    ':status': 'COMPLETED',
                    ':msg': "Document has been fully processed and is ready for use",
                    ':time': datetime.utcnow().isoformat()
                }
            )
            print(f"Document {document_id} is now fully processed")

        # If either has errored, update the overall status to ERROR
        elif 'ERROR' in [cr_index_status, kb_index_status]:
            table.update_item(
                Key={'id': document_id},
                UpdateExpression="SET #status = :status, statusMessage = :msg, lastUpdated = :time",
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={
                    ':status': 'ERROR',
                    ':msg': "Error during document processing",
                    ':time': datetime.utcnow().isoformat()
                }
            )
            print(f"Document {document_id} processing failed")

        # Otherwise, keep the current status
        else:
            print(f"Document {document_id} is still processing")

    except Exception as e:
        print(f"Error checking and updating status: {str(e)}")