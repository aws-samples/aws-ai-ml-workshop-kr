import os
import logging
import sys
import argparse
from opentelemetry import baggage, context
from opentelemetry import trace

def parse_arguments():
    parser = argparse.ArgumentParser(description='Agent with Custom Span Creation')
    parser.add_argument(
        '--session-id', 
        type=str,
        required=True,
        help='Session ID to associate with this agent run'
    )
    return parser.parse_args()

###########################
####   Session info    ####
###########################

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
    # Get the tracer - use the service name for better trace organization
    tracer = trace.get_tracer("web_search", "1.0.0")

    # Start a new span for the web search operation
    with tracer.start_as_current_span("custom span web search tool") as span:
        try:
            # Add query attribute
            span.set_attribute("search.query", query) # set_attribute의 경우 "Traces" 및 GenAI Observability의 metadata에서 확인가능 
            span.set_attribute("tool.name", "web_search") # set_attribute의 경우 "Traces" 및 GenAI Observability의 metadata에서 확인가능 
            span.set_attribute("search.provider", "duckduckgo") # set_attribute의 경우 "Traces" 및 GenAI Observability의 metadata에서 확인가능 

            # Add event for search start
            span.add_event(
                "search_started",
                {"query": query}
            )

            ddgs = DDGS()
            results = ddgs.text(query, max_results=5)

            # Add results count attribute
            span.set_attribute(
                "search.results-count",
                len(results)
            )

            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_result = (
                    f"{i}. {result.get('title', 'No title')}\n"
                    f"   {result.get('body', 'No summary')}\n"
                    f"   Source: {result.get('href', 'No URL')}\n"
                )
                formatted_results.append(formatted_result)

                # Add individual result attributes (limit to avoid too much data)
                if i <= 3:  # Only add details for first 3 results
                    span.set_attribute(f"search.result_{i}.title", result.get('title', 'No title')[:100])
                    span.set_attribute(f"search.result_{i}.url", result.get('href', 'No URL'))

            # Add the formatted results as an attribute (truncated to avoid too much data)
            search_results_text = "\n".join(formatted_results) if formatted_results else "No results found."
            span.set_attribute("search.results_summary", search_results_text[:500])  # Truncate for telemetry

            # Add success event
            span.add_event( # add_event 사용 시 (CloudWatch)GenAI Observability의 trace의 event에서 바로 확인 x, "Traces"의 event에서는 확인 가능
                "search_completed",
                {
                    "results_count": len(results),
                    "success": True
                }
            )

            # Add tool results event
            span.add_event( # add_event 사용 시 (CloudWatch)GenAI Observability의 trace의 event에서 바로 확인 x, "Traces"의 event에서는 확인 가능
                "search_results",
                {
                    "tool-results": str(search_results_text),
                }
            )

            # Set span status to OK
            span.set_status(trace.Status(trace.StatusCode.OK))

            logger.info(f"Web search completed successfully for query: {query[:50]}...") # logger 사용시 (CloudWatch)GenAI Observability의 trace의 event에서 바로 확인 가능, "Traces"의 event에서는 확인 x
            return search_results_text

        except Exception as e:
            # Add error information
            span.set_attribute("search.error", str(e))
            span.set_attribute("search.error_type", type(e).__name__)

            # Add error event
            span.add_event("search_failed", {
                "error": str(e),
                "error_type": type(e).__name__
            })

            # Set span status to ERROR
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))

            logger.error(f"Web search failed: {str(e)}") # set_attribute의 경우 "Traces" 및 GenAI Observability의 metadata에서 확인가능 
            return f"Search error: {str(e)}"

def get_bedrock_model():
    model_id = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    region = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
    logger.info(f"Bedrock model: {model_id} in region: {region}") # set_attribute의 경우 "Traces" 및 GenAI Observability의 metadata에서 확인가능 
    logger.info(f"this is event?: this is event") # set_attribute의 경우 "Traces" 및 GenAI Observability의 metadata에서 확인가능 
    try:
        bedrock_model = BedrockModel(
            model_id=model_id,
            region_name=region,
            temperature=0.7,
            max_tokens=512
        )
        logger.info(f"Successfully initialized Bedrock model: {model_id} in region: {region}") # set_attribute의 경우 "Traces" 및 GenAI Observability의 metadata에서 확인가능 
        return bedrock_model
    except Exception as e:
        logger.error(f"Failed to initialize Bedrock model: {str(e)}") # set_attribute의 경우 "Traces" 및 GenAI Observability의 metadata에서 확인가능 
        logger.error("Please ensure you have proper AWS credentials configured and access to the Bedrock model") # set_attribute의 경우 "Traces" 및 GenAI Observability의 metadata에서 확인가능 
        raise

def main():
    # Parse command line arguments
    args = parse_arguments()

    # Set session context for telemetry
    context_token = set_session_context(args.session_id)

    # Get tracer for main application
    tracer = trace.get_tracer("strands_travel_agent", "1.0.0")
    with tracer.start_as_current_span("travel_agent_session") as main_span:
        try:
            # Add session attributes to main span
            main_span.set_attribute("session.id", args.session_id) ## travel_agent_session 여기 span에 보면 나온다. # set_attribute의 경우 "Traces" 및 GenAI Observability의 metadata에서 확인가능 
            main_span.set_attribute("agent.type", "travel_agent") # set_attribute의 경우 "Traces" 및 GenAI Observability의 metadata에서 확인가능 

            # Initialize Bedrock model
            bedrock_model = get_bedrock_model()

            system_prompt ="""
                You are an expert web search agent specializing in finding accurate and relevant information
                with access to real-time web data. Your role is to efficiently search, analyze, and synthesize
                information from multiple sources to answer user queries comprehensively. You should provide
                well-researched responses with current information, clear summaries, and cite reliable sources
                when presenting your findings.
                """

            # Create travel agent
            travel_agent = Agent(
                model=bedrock_model,
                system_prompt=system_prompt,
                tools=[web_search],
                trace_attributes={
                    "user.id": "user@domain.com",
                    "tags": ["Strands", "Observability"],
                }
            )

            # Execute the travel research task
            query = """Which of restaurants are vegetarian friendly in Seoul?"""
            result = travel_agent(query)
            main_span.set_status(trace.Status(trace.StatusCode.OK))

            print("Result:", result)

            main_span.add_event( # add_event 사용 시 (CloudWatch)GenAI Observability의 trace의 event에서 바로 확인 x, "Traces"의 event에서는 확인 가능
                "agent_system_prompt",
                {"system-prompt": str(system_prompt)}
            ) # attribute.name에 _(under bar)가 들어가면 안된다. 
            main_span.add_event( # add_event 사용 시 (CloudWatch)GenAI Observability의 trace의 event에서 바로 확인 x, "Traces"의 event에서는 확인 가능
                "agent_query_started",
                {"query": str(query[:100])}
            )
            main_span.add_event( # add_event 사용 시 (CloudWatch)GenAI Observability의 trace의 event에서 바로 확인 x, "Traces"의 event에서는 확인 가능
                "agent_query_completed",
                {
                    "result": str(result),
                    "success": True
                }
            )

        except Exception as e:
            main_span.set_attribute("error", str(e))
            main_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            logger.error(f"Main execution failed: {str(e)}")
            raise
        finally:
            # Detach context when done
            context.detach(context_token)
            logger.info(f"Session context for '{args.session_id}' detached")

if __name__ == "__main__":
    main()
