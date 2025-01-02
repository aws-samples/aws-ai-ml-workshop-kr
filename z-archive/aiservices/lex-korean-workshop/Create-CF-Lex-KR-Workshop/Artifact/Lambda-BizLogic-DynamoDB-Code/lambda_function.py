import json
import boto3
from boto3.dynamodb.types import TypeDeserializer
import cfnresponse


# function to import the data from source file into give table
def import_data(table_name):
    deserializer = TypeDeserializer()

    my_session = boto3.session.Session()
    my_region = my_session.region_name
    # my_region = 'ap-northeast-2'
    dynamodb = boto3.resource('dynamodb',region_name=my_region)
    table = dynamodb.Table(table_name)
    print("Table status:", table.table_status)

    # Read the JSON file
    with open('BankingBotData.json') as json_data:
        items = json.load(json_data)
    itemRecords = items['Items']
    itemRecordsCnt=len(itemRecords)
    for x in range(itemRecordsCnt):
        deserialized_document = {k: deserializer.deserialize(v) for k, v in \
                                    itemRecords[x].items()}
        table.put_item(Item=deserialized_document)
    return 'SUCCESS'


# --- Main handler ---
def lambda_handler(event, context):
    try:
        print('event: ', event)
        request_type = event.get('RequestType')
        print('request_type: ', request_type)        
        table_name = event.get('ResourceProperties')['TableName']
        # table_name = event.get('TableName')
        print("table_name: ", table_name)
        
        response = {}
        response_id = event.get('RequestId')
        
        if request_type == 'Create':
            status = import_data(table_name)
            if status == 'SUCCESS':
                reason = 'Imported db data succesfully'
                print("Success with reason: ", reason)
                cfnresponse.send(
                    event, context, cfnresponse.SUCCESS, response,
                    response_id, reason)
            else:
                reason = 'Imported db data failed'
                print("Failure with reason: ", reason)        
                cfnresponse.send(
                    event, context, cfnresponse.FAILED, response,
                    response_id, 'DB data load failed')
        elif request_type == 'Delete':
            reason = 'No action required'
            print("Request_type", request_type, " Reason: ", reason)                        
            response_status = cfnresponse.SUCCESS
            cfnresponse.send(
                event, context, response_status, response, response_id, reason)
        
    except Exception as e:
        reason = 'DB data load failed because of exception. '+ str(e)
        print("exception occured: \n", reason)                   
        response_status = cfnresponse.FAILED
        response_id = event.get('RequestId')        
        cfnresponse.send(
            event, context, cfnresponse.FAILED, {}, response_id, reason)
