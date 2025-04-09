import chainlit as cl
from InlineAgent.agent import InlineAgent
from InlineAgent.tools import MCPHttp
import os
from dotenv import load_dotenv


load_dotenv()

# Configuration
EC2_HOST = os.getenv('BASTION_HOST')
MCP_PORT = os.getenv('MCP_PORT')
EKS_CLUSTER = os.getenv("EKS_CLUSTER")

# Global variables
agent = None
mcp_client = None


async def initialize_agent():
    """Initialize the MCP client and InlineAgent"""
    global agent, mcp_client

    try:
        # Configure connection to the Kubernetes MCP server
        mcp_client = await MCPHttp.create(
            url=f"http://{EC2_HOST}:{MCP_PORT}/sse",
            headers={},
            timeout=10,
            sse_read_timeout=300
        )

        # Create the InlineAgent with the MCP client
        agent = InlineAgent(
            foundation_model="us.anthropic.claude-3-5-haiku-20241022-v1:0",
            instruction="""You are a Kubernetes cluster management assistant that helps users manage their EKS cluster.
            You have access to various kubectl commands through an MCP server.
            When users ask you about Kubernetes resources or want to perform actions, use the appropriate tools.
            Always show the relevant information clearly and explain what you're doing.
            """,
            agent_name="kubernetes-assistant",
            action_groups=[
                {
                    "name": "KubernetesActions",
                    "description": "Tools for managing Kubernetes clusters",
                    "mcp_clients": [mcp_client]
                }
            ]
        )

        return agent
    except Exception as e:
        # Send error message
        await cl.Message(f"Error initializing agent: {str(e)}").send()
        return None


async def close_mcp_client():
    """Close the MCP client connection"""
    global mcp_client
    if mcp_client:
        try:
            await mcp_client.aclose()
        except Exception:
            pass
        mcp_client = None


@cl.on_chat_start
async def start_chat():
    """Initialize the chat session"""
    # Send a welcome message
    await cl.Message("# ☸️ Welcome to the Kubernetes Cluster Manager\n\n"
                     "This assistant helps you manage your Kubernetes cluster through natural language commands.").send()

    # Show connection settings
    await cl.Message(f"## Connection Settings\n"
                     f"- EC2 Host: `{EC2_HOST}`\n"
                     f"- MCP Port: `{MCP_PORT}`\n"
                     f"- Cluster: `{EKS_CLUSTER}`").send()

    # Initialize the agent
    await cl.Message("Initializing Kubernetes assistant...").send()

    global agent
    agent = await initialize_agent()

    if agent:
        await cl.Message("✅ Kubernetes assistant initialized successfully!").send()

        # Initial greeting from the assistant
        try:
            initial_response = await agent.invoke(
                "Hello! I'm your Kubernetes assistant. How can I help you with your EKS cluster today?"
            )
            await cl.Message(initial_response).send()
        except Exception as e:
            await cl.Message(f"Error: {str(e)}").send()
    else:
        await cl.Message("⚠️ Failed to initialize Kubernetes assistant.").send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages"""
    global agent

    if not agent:
        await cl.Message("Initializing agent...").send()
        agent = await initialize_agent()

        if not agent:
            await cl.Message(
                "⚠️ Failed to initialize the Kubernetes assistant. Please check your connection settings.").send()
            return

        await cl.Message("✅ Agent initialized successfully!").send()

    # Process the message
    try:
        # Let the user know we're processing
        processing_msg = await cl.Message("Processing your request...").send()

        # Get response from agent
        response = await agent.invoke(message.content)

        # Send the response as a new message
        await cl.Message(response).send()

    except Exception as e:
        await cl.Message(f"⚠️ Error: {str(e)}").send()


@cl.on_stop
async def on_stop():
    """Clean up when the chat stops"""
    await close_mcp_client()