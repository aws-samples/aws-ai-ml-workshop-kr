from termcolor import colored

def opensearch_pretty_print_documents(response):
    '''
    OpenSearch 결과인 LIST 를 파싱하는 함수
    '''
    for doc, score in response:
        print(f'\nScore: {score}')
        # print(f'Document Number: {doc.metadata["row"]}')

        # Split the page content into lines
        lines = doc.page_content.split("\n")
        metadata = doc.metadata
        print(lines)
        print(metadata)        
        
        
        
        # print(doc.metadata['origin'])    

        # Extract and print each piece of information if it exists
        # for line in lines:
        #     split_line = line.split(": ")
        #     if len(split_line) > 1:
        #         print(f'{split_line[0]}: {split_line[1]}')

        # print("Metadata:")
        # print(f'Type: {doc.metadata["type"]}')
        # print(f'Source: {doc.metadata["source"]}')        

        print('-' * 50)

def put_parameter(boto3_clinet, parameter_name, parameter_value):

    # Specify the parameter name, value, and type
    parameter_type = 'SecureString'

    try:
        # Put the parameter
        response = boto3_clinet.put_parameter(
            Name=parameter_name,
            Value=parameter_value,
            Type=parameter_type,
            Overwrite=True  # Set to True if you want to overwrite an existing parameter
        )

        # Print the response
        print('Parameter stored successfully.')
        print(response)

    except Exception as e:
        print('Error storing parameter:', str(e))
    

def get_parameter(boto3_clinet, parameter_name):
    # Create a SSM Client

    try:
        # Get the parameter
        response = boto3_clinet.get_parameter(
            Name=parameter_name,
            WithDecryption=True  # Set to True if the parameter is a SecureString
        )

        # Retrieve parameter value from response
        parameter_value = response['Parameter']['Value']

        # Print the parameter value
        # print('Parameter Value:', parameter_value)
        
        return parameter_value

    except Exception as e:
        print('Error retrieving parameter:', str(e))


