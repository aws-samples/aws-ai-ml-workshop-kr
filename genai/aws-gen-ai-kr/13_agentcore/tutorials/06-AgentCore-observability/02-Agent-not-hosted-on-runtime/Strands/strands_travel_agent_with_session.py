import os
import logging
import sys
import argparse
from opentelemetry import baggage, context

def parse_arguments():
    parser = argparse.ArgumentParser(description='Strands Travel Agent with Session Tracking')
    parser.add_argument('--session-id', 
                       type=str, 
                       required=True,
                       help='Session ID to associate with this agent run')
    parser.add_argument('--exp-id', 
                       type=str, 
                       required=False,
                       help='experiment id')
    return parser.parse_args()

def set_session_context(session_id):
    """Set the session ID in OpenTelemetry baggage for trace correlation"""
    ctx = baggage.set_baggage("session.id", session_id)
    token = context.attach(ctx)
    logging.info(f"Session ID '{session_id}' attached to telemetry context")
    return token

###########################
#### Agent Code below: ####
###########################

from strands import Agent, tool
from strands.models import BedrockModel
from duckduckgo_search import DDGS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Strands logging
logging.getLogger("strands").setLevel(logging.INFO)

@tool
def web_search(query: str) -> str:
    """Search the web for current information about travel destinations, attractions, and events."""
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=5)

        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"{i}. {result.get('title', 'No title')}\n"
                f"   {result.get('body', 'No summary')}\n"
                f"   Source: {result.get('href', 'No URL')}\n"
            )

        return "\n".join(formatted_results) if formatted_results else "No results found."

    except Exception as e:
        return f"Search error: {str(e)}"

def get_bedrock_model():
    model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
    region = os.getenv("AWS_DEFAULT_REGION", "us-west-2")

    try:
        bedrock_model = BedrockModel(
            model_id=model_id,
            region_name=region,
            temperature=0.7,
            max_tokens=512
        )
        logger.info(f"Successfully initialized Bedrock model: {model_id} in region: {region}")
        return bedrock_model
    except Exception as e:
        logger.error(f"Failed to initialize Bedrock model: {str(e)}")
        logger.error("Please ensure you have proper AWS credentials configured and access to the Bedrock model")
        raise

def main():
    # Parse command line arguments
    args = parse_arguments()

    # Set session context for telemetry
    context_token = set_session_context(args.session_id)

    try:
        # Initialize Bedrock model
        bedrock_model = get_bedrock_model()

        # Create travel agent
        travel_agent = Agent(
            model=bedrock_model,
            system_prompt="""You are an experienced travel agent specializing in personalized travel recommendations 
            with access to real-time web information. Your role is to find dream destinations matching user preferences 
            using web search for current information. You should provide comprehensive recommendations with current 
            information, brief descriptions, and practical travel details.""",
            tools=[web_search],
            trace_attributes={
                "user.id": "user@domain.com",
                "tags": ["Strands", "Observability"],
            }
        )

        # Execute the travel research task
        query = """Research and recommend suitable travel destinations for someone looking for cowboy vibes, 
        rodeos, and museums in New York city. Use web search to find current information about venues, 
        events, and attractions."""

        result = travel_agent(query)
        print("Result:", result)

    finally:
        # Detach context when done
        context.detach(context_token)
        logger.info(f"Session context for '{args.session_id}' detached")

if __name__ == "__main__":
    main()
