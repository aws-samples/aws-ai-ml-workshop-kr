import os
import json
import argparse
import asyncio
from datetime import datetime
from typing import Dict, Any

from strands import Agent
from strands.models import BedrockModel
from botocore.config import Config


from bedrock_agentcore.runtime import BedrockAgentCoreApp
app = BedrockAgentCoreApp()


# You'll need to import these from your project
# from src.utils.bedrock import bedrock_info

class bedrock_info:
    @staticmethod
    def get_model_id(model_name):
        # Placeholder - replace with actual implementation
        model_mapping = {
            "Claude-V3-5-V-2-Sonnet-CRI": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "Claude-V3-7-Sonnet-CRI": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        }
        return model_mapping.get(model_name, model_name)

def get_model(**kwargs):
    llm_type = kwargs["llm_type"]
    cache_type = kwargs["cache_type"]
    enable_reasoning = kwargs["enable_reasoning"]

    if llm_type == "reasoning":    
        llm = BedrockModel(
            model_id=bedrock_info.get_model_id(model_name="Claude-V3-7-Sonnet-CRI"),
            streaming=True,
            max_tokens=8192*5,
            stop_sequences=["\n\nHuman"],
            temperature=1 if enable_reasoning else 0.01, 
            additional_request_fields={
                "thinking": {
                    "type": "enabled" if enable_reasoning else "disabled", 
                    **({"budget_tokens": 8192} if enable_reasoning else {}),
                }
            },
            cache_prompt=cache_type,
            boto_client_config=Config(
                read_timeout=900,
                connect_timeout=900,
                retries=dict(max_attempts=50, mode="adaptive"),
            )
        )

    elif llm_type == "basic":
        llm = BedrockModel(
            model_id=bedrock_info.get_model_id(model_name="Claude-V3-5-V-2-Sonnet-CRI"),
            streaming=True,
            max_tokens=8192,
            stop_sequences=["\n\nHuman"],
            temperature=0.01,
            cache_prompt=cache_type,
            boto_client_config=Config(
                read_timeout=900,
                connect_timeout=900,
                retries=dict(max_attempts=50, mode="standard"),
            )
        )
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")

    return llm

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def apply_prompt_template(prompt_name: str, prompt_context={}) -> str:
    try:
        system_prompts = open(os.path.join("./prompts", f"{prompt_name}.md")).read()    
    except FileNotFoundError:
        # Fallback system prompt
        system_prompts = "You are a helpful AI assistant."

    context = {"CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z")}
    context.update(prompt_context)
    system_prompts = system_prompts.format(**context)
    return system_prompts

def get_agent(**kwargs):
    agent_name, system_prompts = kwargs["agent_name"], kwargs["system_prompts"]
    agent_type = kwargs.get("agent_type", "basic")
    prompt_cache_info = kwargs.get("prompt_cache_info", (False, None))
    tools = kwargs.get("tools", None)
    streaming = kwargs.get("streaming", True)

    if "reasoning" in agent_type: 
        enable_reasoning = True
    else: 
        enable_reasoning = False

    prompt_cache, cache_type = prompt_cache_info
    if prompt_cache: 
        print(f"{Colors.GREEN}{agent_name.upper()} - Prompt Cache Enabled{Colors.END}")
    else: 
        print(f"{Colors.GREEN}{agent_name.upper()} - Prompt Cache Disabled{Colors.END}")

    llm = get_model(llm_type=agent_type, cache_type=cache_type, enable_reasoning=enable_reasoning)
    llm.config["streaming"] = streaming

    agent = Agent(
        model=llm,
        system_prompt=system_prompts,
        tools=tools,
        callback_handler=None
    )
    return agent

async def _convert_to_agentcore_event(
    strands_event: Dict[str, Any],
    agent_name: str,
    session_id: str
) -> Dict[str, Any]:
    """Strands 이벤트를 AgentCore 스트리밍 형식으로 변환"""

    base_event = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "agent_name": agent_name,
        "source": "strands_data_analysis_graph"
    }

    # 텍스트 데이터 이벤트
    if "data" in strands_event:
        return {
            **base_event,
            "type": "agent_text_stream",
            "event_type": "text_chunk",
            "data": strands_event["data"],
            "chunk_size": len(strands_event["data"])
        }

    # 도구 사용 이벤트
    elif "current_tool_use" in strands_event:
        tool_info = strands_event["current_tool_use"]
        return {
            **base_event,
            "type": "agent_tool_stream",
            "event_type": "tool_use",
            "tool_name": tool_info.get("name", "unknown"),
            "tool_id": tool_info.get("toolUseId"),
            "tool_input": tool_info.get("input", {})
        }

    # 추론 이벤트
    elif "reasoning" in strands_event and strands_event.get("reasoning"):
        return {
            **base_event,
            "type": "agent_reasoning_stream",
            "event_type": "reasoning",
            "reasoning_text": strands_event.get("reasoningText", "")[:200]
        }

    return None

async def process_agent_stream(agent, message):
    coordinator_result = ""
    agent_stream = agent.stream_async(message)
    session_id = "123"

    async for event in agent_stream:
        #Strands 이벤트를 AgentCore 형식으로 변환
        agentcore_event = await _convert_to_agentcore_event(
            event, "coordinator", session_id
        )
        if agentcore_event:
            yield agentcore_event

            # 결과 텍스트 누적
            if agentcore_event.get("event_type") == "text_chunk":
                coordinator_result += agentcore_event.get("data", "")

async def node(agent, message):
    async for event in process_agent_stream(agent, message):
        yield event

# Create agent instance
agent = get_agent(
    agent_name="task_agent",
    system_prompts=apply_prompt_template(prompt_name="task_agent", prompt_context={}),
    agent_type="reasoning",
    prompt_cache_info=(True, "default"),
    streaming=True,
)

@app.entrypoint
async def strands_agent_bedrock(payload):
    """
    Invoke the agent with a payload
    """
    user_input = payload.get("prompt")
    async for event in node(agent, user_input):
        #print(f"Event: {event}")
        #yield f"ss:{event}"
        if "data" in event:
            print (event["data"])
            yield event["data"]

if __name__ == "__main__":

    app.run()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("payload", type=str)
    # args = parser.parse_args()

    # async def main():
    #     async for event in strands_agent_bedrock(json.loads(args.payload)):
    #         print(f"Final event: {event}")

    # asyncio.run(main())
