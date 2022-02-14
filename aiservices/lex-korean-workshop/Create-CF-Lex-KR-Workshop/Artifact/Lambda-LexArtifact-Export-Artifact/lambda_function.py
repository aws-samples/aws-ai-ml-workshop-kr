import json
import boto3
import requests
import cfnresponse
import time
from zipfile import ZipFile
from io import BytesIO
BOT_DETAILS_FILE = 'bot_details.json'
client = boto3.client('lexv2-models')
ALIAS = 'PROD'
TIMEOUT = 300 #seconds
BOT_ALIAS_ID = 'TSTALIASID'

def wait_till(**kwargs):
    bot_id = kwargs['bot_id']
    status_type = kwargs['status_type']
    status = kwargs['status']
    current_status = 'Unknown'
    for attempts in range(TIMEOUT):
        if status_type == 'bot':
            response = client.describe_bot(
                botId=bot_id
            )
            current_status = response['botStatus']
        elif status_type == 'alias':
            response = client.describe_bot_alias(
                botId=bot_id,
                botAliasId=kwargs['botAliasId']
            )
            current_status = response['botAliasStatus']
        elif status_type == 'build':
            response = client.describe_bot_locale(
                botId=bot_id,
                botVersion='DRAFT',
                localeId='ko_KR'
            )
            current_status = response['botLocaleStatus']
        if current_status == status:
            return response
        time.sleep(1)
    raise Exception('Timeout (300secs). Bot status is :', current_status)

def wait_till_import(import_id):
    current_status = 'Unknown'
    for attempts in range(TIMEOUT):
        response = client.describe_import(
            importId=import_id
        )
        current_status = response['importStatus']
        if current_status == 'Completed':
            return response
        time.sleep(1)
    raise Exception('Timeout (300secs). Bot status is :', current_status)

def import_lex_bot(roleArn, bot_name, lambda_function_name):
    f = open(BOT_DETAILS_FILE,)
    bot_details = json.load(f)
    presigned = client.create_upload_url()
    bot_definition_src = bot_details['bot_definition']
    if not bot_name:
        bot_name = bot_details['name']

    with open(bot_definition_src, 'rb') as f:
        uploadbase = requests.put(
            presigned['uploadUrl'], data=f,
            headers={'X-File-Name' : bot_definition_src,
                'Content-Disposition': 'form-data; name="{0}"; filename="{0}"'\
                .format(bot_definition_src),
                'content-type': 'multipart/form-data'})

    # Step 1: Import the bot
    response = client.start_import(
        importId=presigned['importId'],
        resourceSpecification={
            'botImportSpecification': {
                'botName': bot_name,
                'roleArn': roleArn,
                'dataPrivacy': {
                    'childDirected': False
                },
                'idleSessionTTLInSeconds': 300
            }
        },
        mergeStrategy='FailOnConflict'
    )

    import_details = wait_till_import(presigned['importId'])
    bot_id = import_details['importedResourceId']

    # Step 2: Build the bot
    response = client.build_bot_locale(
        botId=bot_id,
        botVersion='DRAFT',
        localeId='ko_KR'
    )
    wait_till(
        status_type='build',
        status='Built',
        bot_id=bot_id)

    # Step 3: Create bot version
    response = client.create_bot_version(
        botId=bot_id,
        description='Production version created by the cloud formation',
        botVersionLocaleSpecification={
            'ko_KR': {
                'sourceBotVersion': 'DRAFT'
            }
        }
    )
    bot_version = response['botVersion']

    wait_till(
        status_type='bot',
        status='Available',
        bot_id=bot_id)

    # Step 4: Create lambda version
    lambda_client = boto3.client('lambda')
    response = lambda_client.publish_version(
        FunctionName=lambda_function_name,
    )
    function_arn = response['FunctionArn']
    function_version = response['Version']

    # Step 5: Create alias, associate newly created version and lambda function
    response = client.create_bot_alias(
        botAliasName='Prod',
        description='Alias created by the cloud formation',
        botVersion=bot_version,
        botId=bot_id,
        botAliasLocaleSettings={
            'ko_KR': {
                'enabled': True,
                'codeHookSpecification': {
                    'lambdaCodeHook': {
                        'lambdaARN': function_arn,
                        'codeHookInterfaceVersion': '1.0'
                    }
                }
            }
        }
    )

    alias_id = response['botAliasId']
    alias_details = wait_till(
        status_type='alias',
        status='Available',
        bot_id=bot_id,
        botAliasId=alias_id)

    account_id = boto3.client('sts').get_caller_identity()['Account']
    my_session = boto3.session.Session()
    my_region = my_session.region_name
    lex_bot_arn = 'arn:aws:lex:'+my_region+':'+ str(account_id)\
                + ':bot-alias/' + bot_id + '/' + alias_id

    # step 6: Set permissions in lambda for the new version
    response = lambda_client.add_permission(
        FunctionName=function_arn,
        StatementId='LexAccess_'+ alias_id,
        Action='lambda:InvokeFunction',
        Principal='lexv2.amazonaws.com',
        SourceArn=lex_bot_arn
    )

    # step 7: Associate the draft version of lex with the $LATEST lambda function
    # and set permissions in lambda for the draft version as part of cft
    response = lambda_client.get_function(
        FunctionName=lambda_function_name
    )
    function_arn_base = response.get('Configuration').get('FunctionArn')
    response = client.update_bot_alias(
        botAliasId=BOT_ALIAS_ID,
        botAliasName='TestBotAlias',
        botVersion='DRAFT',
        botId=bot_id,
        botAliasLocaleSettings={
            'ko_KR': {
                'enabled': True,
                'codeHookSpecification': {
                    'lambdaCodeHook': {
                        'lambdaARN': function_arn_base,
                        'codeHookInterfaceVersion': '1.0'
                    }
                }
            }
        }
    )

    lex_bot_arn = 'arn:aws:lex:'+my_region+':'+ str(account_id)\
                + ':bot-alias/' + bot_id + '/' + BOT_ALIAS_ID

    return {
        'lex_arn': lex_bot_arn,
        'bot_id': bot_id,
        'bot_alias_id': alias_id
    }

def delete_lex_bot(bot_name):
    client = boto3.client('lexv2-models')
    with open('bot_details.json', 'r') as f:
        bot_details = json.loads(f.read())
    if not bot_name:
        bot_name = bot_details['name']
    bots_list = client.list_bots(
        filters=[
            {
                'name': 'BotName',
                'values': [bot_name],
                'operator': 'EQ'
            },
        ],
        maxResults=200).get('botSummaries')

    if bots_list and len(bots_list) > 0:
        current_bot = bots_list[0]
        response = client.delete_bot(
            botId=current_bot.get('botId'),
            skipResourceInUseCheck=True
        )
        return True
    else:
        return False

# import_lex_bot('arn:aws:iam::049982172265:role/aws-service-role/lexv2.amazonaws.com/AWSServiceRoleForLexV2Bots_QV0YH7IDOM', 'testbyjp1234', 'AutoInsurance')
# --- Main handler ---
def lambda_handler(event, context):
    try:
        print('event: ', event)
        roleArn = event['ResourceProperties']['RoleARN']
        lambda_function_name = event['ResourceProperties']['LambdaFunctionName']
        bot_name = event['ResourceProperties']['BotName']
        request_type = event['RequestType']
        response = {}
        response_id = event.get('RequestId')
        
#         reason = 'Lex bot import failed because intentional purpose.'
#         print("reason with intentional purpose. ", reason)                        
#         cfnresponse.send(
#             event, context, cfnresponse.FAILED, {}, response_id, reason)

                
        if request_type == 'Create':
            import_response = import_lex_bot(roleArn, bot_name, lambda_function_name)
            if import_response.get('lex_arn'):
                response['lex_arn'] = import_response.get('lex_arn')
                response['bot_id'] = import_response.get('bot_id')
                response['bot_alias_id'] = import_response.get('bot_alias_id')
                reason = 'Imported bot succesfully'
                print("response with lex_arn: ", response)
                cfnresponse.send(
                    event, context, cfnresponse.SUCCESS, response,
                    response_id, reason)
            else:
                reason = 'Bot import failed'
                cfnresponse.send(
                    event, context, cfnresponse.FAILED, {},
                    response_id, reason)
        elif request_type == 'Delete':
            status = delete_lex_bot(bot_name)
            if status:
                reason = 'Deleted bot succesfully'
                print("reason with Delete: ", reason)                
                cfnresponse.send(
                    event, context, cfnresponse.SUCCESS, {},
                    response_id, reason)
            else:
                reason = 'Delete bot failed'
                cfnresponse.send(
                    event, context, cfnresponse.FAILED, {},
                    response_id, reason)

    except Exception as e:
        response_id = event.get('RequestId')
        reason = 'Lex bot import failed because of exception. '+ str(e)
        print("reason with Exception: ", reason)                        
        cfnresponse.send(
            event, context, cfnresponse.FAILED, {}, response_id, reason)
