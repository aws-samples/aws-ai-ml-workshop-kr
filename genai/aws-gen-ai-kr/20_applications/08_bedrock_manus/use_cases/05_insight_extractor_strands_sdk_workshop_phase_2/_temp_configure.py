
from bedrock_agentcore_starter_toolkit import Runtime

agentcore_runtime = Runtime()
response = agentcore_runtime.configure(
    agent_name='bedrock_manus_runtime',
    entrypoint='agentcore_runtime.py',
    execution_role='agent_role_custom',
    auto_create_execution_role=True,
    auto_create_ecr=True,
    requirements_file='requirements.txt',
    region='us-west-2'
)
print('CONFIGURE_SUCCESS')
