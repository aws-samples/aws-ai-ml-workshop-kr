import json
import boto3
import os
import cfnresponse  # Import the local module

s3 = boto3.client('s3')
cloudfront = boto3.client('cloudfront')


def handler(event, context):
    try:
        print(f"Starting custom resource handler with RequestType: {event['RequestType']}")

        if event['RequestType'] in ['Create', 'Update']:
            bucket = os.environ['WEBSITE_BUCKET']
            websocket_url = os.environ['WEBSOCKET_URL']
            cloudfront_domain = os.environ['CLOUDFRONT_DOMAIN']
            distribution_id = os.environ['DISTRIBUTION_ID']
            upload_api_url = os.environ['UPLOAD_API_URL']

            print(f"Using bucket: {bucket}")
            print(f"Using websocket URL: {websocket_url}")
            print(f"Using CloudFront domain: {cloudfront_domain}")

            # Create updated config
            config = {
                'websocketUrl': websocket_url,
                'cloudfrontDomain': cloudfront_domain,
                'uploadApiUrl': upload_api_url
            }
            print("Config created, uploading to S3...")

            # Upload to S3
            s3.put_object(
                Bucket=bucket,
                Key='config.json',
                Body=json.dumps(config),
                ContentType='application/json'
            )
            print("S3 upload complete")

            # Create CloudFront invalidation
            print(f"Creating CloudFront invalidation for distribution: {distribution_id}")
            invalidation_response = cloudfront.create_invalidation(
                DistributionId=distribution_id,
                InvalidationBatch={
                    'Paths': {
                        'Quantity': 1,
                        'Items': ['/config.json']
                    },
                    'CallerReference': f'config-update-{context.aws_request_id}'
                }
            )
            print(f"Invalidation created: {invalidation_response['Invalidation']['Id']}")

            cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
        else:
            print(f"Non-create/update event: {event['RequestType']}")
            cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        cfnresponse.send(event, context, cfnresponse.FAILED, {})