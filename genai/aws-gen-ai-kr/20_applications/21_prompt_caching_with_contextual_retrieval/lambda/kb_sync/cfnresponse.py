import json
import urllib.request

SUCCESS = "SUCCESS"
FAILED = "FAILED"

def send(event, context, responseStatus, responseData, physicalResourceId=None, noEcho=False, reason=None):
    responseUrl = event['ResponseURL']

    print(responseUrl)

    responseBody = {
        'Status': responseStatus,
        'Reason': reason or f"See the details in CloudWatch Log Stream: {context.log_stream_name}",
        'PhysicalResourceId': physicalResourceId or context.log_stream_name,
        'StackId': event['StackId'],
        'RequestId': event['RequestId'],
        'LogicalResourceId': event['LogicalResourceId'],
        'NoEcho': noEcho,
        'Data': responseData
    }

    responseBody = json.dumps(responseBody)

    print("Response body:")
    print(responseBody)

    headers = {
        'content-type': '',
        'content-length': str(len(responseBody))
    }

    try:
        req = urllib.request.Request(responseUrl, data=responseBody.encode('utf-8'), headers=headers, method='PUT')
        response = urllib.request.urlopen(req)
        print(f"Status code: {response.getcode()}")
        return True
    except Exception as e:
        print("send(..) failed executing request.urlopen(..): " + str(e))
        return False