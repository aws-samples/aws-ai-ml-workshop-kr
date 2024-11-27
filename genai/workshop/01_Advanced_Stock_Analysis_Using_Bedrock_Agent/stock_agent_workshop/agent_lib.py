import boto3


def get_agent_response(agent_id, agent_alias_id, session_id, prompt):
    """Get a response from the Bedrock agent using specified parameters."""

    # Create a Boto3 client for the Bedrock Runtime service
    session = boto3.Session()
    bedrock_agent = session.client(service_name='bedrock-agent-runtime', region_name='us-west-2')

    # Invoke the Bedrock agent with the specified parameters
    response = bedrock_agent.invoke_agent(
        agentId=agent_id,
        agentAliasId=agent_alias_id,
        enableTrace=True,
        sessionId=session_id,
        inputText=prompt,
    )

    return response
