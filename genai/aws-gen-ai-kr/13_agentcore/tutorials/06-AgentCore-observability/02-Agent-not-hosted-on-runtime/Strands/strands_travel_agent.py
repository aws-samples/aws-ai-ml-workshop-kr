###########################
#### Agent Code below: ####
###########################
import os
import logging
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

            print (formatted_results)

        return "\n".join(formatted_results) if formatted_results else "No results found."

    except Exception as e:
        return f"Search error: {str(e)}"

def get_bedrock_model():
    model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
    region = os.getenv("AWS_DEFAULT_REGION", "us-west-2")

    print (region)

    try:
        bedrock_model = BedrockModel(
            model_id=model_id,
            region_name=region,
            temperature=0.0,
            max_tokens=512
        )
        logger.info(f"Successfully initialized Bedrock model: {model_id} in region: {region}")
        return bedrock_model
    except Exception as e:
        logger.error(f"Failed to initialize Bedrock model: {str(e)}")
        logger.error("Please ensure you have proper AWS credentials configured and access to the Bedrock model")
        raise

# Initialize the model
bedrock_model = get_bedrock_model()

# Create the travel agent
travel_agent = Agent(
    model=bedrock_model,
    system_prompt="""You are an experienced travel agent specializing in personalized travel recommendations 
    with access to real-time web information. Your role is to find dream destinations matching user preferences 
    using web search for current information. You should provide comprehensive recommendations with current 
    information, brief descriptions, and practical travel details.""",
    tools=[web_search]
)

# Execute the travel research task
query = """Research and recommend suitable travel destinations for someone looking for cowboy vibes, 
rodeos, and museums in New York city. Use web search to find current information about venues, 
events, and attractions."""

result = travel_agent(query)
print("Result:", result)
