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


