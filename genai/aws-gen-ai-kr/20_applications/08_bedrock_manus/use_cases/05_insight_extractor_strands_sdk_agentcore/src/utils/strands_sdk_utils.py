import logging
import traceback
from src.utils.bedrock import bedrock_info
from strands import Agent, tool
from strands.models import BedrockModel
from botocore.config import Config
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 새 핸들러와 포맷터 설정
logger = logging.getLogger(__name__)
logger.propagate = False  # 상위 로거로 메시지 전파 중지
for handler in logger.handlers[:]: logger.removeHandler(handler)
handler = logging.StreamHandler()
formatter = logging.Formatter('\n%(levelname)s [%(name)s] %(message)s')  # 로그 레벨이 동적으로 표시되도록 변경
handler.setFormatter(formatter)
logger.addHandler(handler)
# DEBUG와 INFO 중 원하는 레벨로 설정
logger.setLevel(logging.INFO)  # 기본 레벨은 INFO로 설정

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class ColoredStreamingCallback(StreamingStdOutCallbackHandler):
    COLORS = {
        'blue': '\033[94m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
    }
    
    def __init__(self, color='blue'):
        super().__init__()
        self.color_code = self.COLORS.get(color, '\033[94m')
        self.reset_code = '\033[0m'
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"{self.color_code}{token}{self.reset_code}", end="", flush=True)

class strands_utils():

    @staticmethod
    def get_model(**kwargs):

        llm_type = kwargs["llm_type"]
        cache_type = kwargs["cache_type"]
        enable_reasoning = kwargs["enable_reasoning"]
    
        if llm_type == "reasoning":    
            ## BedrockModel params: https://strandsagents.com/latest/api-reference/models/?h=bedrockmodel#strands.models.bedrock.BedrockModel
            llm = BedrockModel(
                model_id=bedrock_info.get_model_id(model_name="Claude-V3-7-Sonnet-CRI"),
                streaming=True,
                max_tokens=8192*5,
                stop_sequencesb=["\n\nHuman"],
                temperature=1 if enable_reasoning else 0.01, 
                additional_request_fields={
                    "thinking": {
                        "type": "enabled" if enable_reasoning else "disabled", 
                        **({"budget_tokens": 8192} if enable_reasoning else {}),
                    }
                },
                cache_prompt=cache_type, # None/ephemeral/defalut
                #cache_tools: Cache point type for tools
                boto_client_config=Config(
                    read_timeout=900,
                    connect_timeout=900,
                    retries=dict(max_attempts=50, mode="adaptive"),
                )
            )
            
        elif llm_type == "basic":
            ## BedrockModel params: https://strandsagents.com/latest/api-reference/models/?h=bedrockmodel#strands.models.bedrock.BedrockModel
            llm = BedrockModel(
                model_id=bedrock_info.get_model_id(model_name="Claude-V3-5-V-2-Sonnet-CRI"),
                streaming=True,
                max_tokens=8192,
                stop_sequencesb=["\n\nHuman"],
                temperature=0.01,
                cache_prompt=cache_type, # None/ephemeral/defalut
                #cache_tools: Cache point type for tools
                boto_client_config=Config(
                    read_timeout=900,
                    connect_timeout=900,
                    retries=dict(max_attempts=50, mode="standard"),
                )
            )

        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")
            
        return llm


    @staticmethod
    def get_agent(**kwargs):

        agent_name, system_prompts = kwargs["agent_name"], kwargs["system_prompts"]
        agent_type = kwargs.get("agent_type", "basic") # "reasoning"
        prompt_cache_info = kwargs.get("prompt_cache_info", (False, None)) # (True, "default")
        tools = kwargs.get("tools", None)
        streaming = kwargs.get("streaming", True)
            
        if "reasoning" in agent_type: enable_reasoning = True
        else: enable_reasoning = False

        ## Use reasoning model but not use reasoning capability
        #if agent_name == "supervisor": enable_reasoning = False

        prompt_cache, cache_type = prompt_cache_info
        if prompt_cache: logger.info(f"{Colors.GREEN}{agent_name.upper()} - Prompt Cache Enabled{Colors.END}")
        else: logger.info(f"{Colors.GREEN}{agent_name.upper()} - Prompt Cache Disabled{Colors.END}")

        #llm = get_llm_by_type(AGENT_LLM_MAP[agent_name], cache_type, enable_reasoning)
        llm = strands_utils.get_model(llm_type=agent_type, cache_type=cache_type, enable_reasoning=enable_reasoning)
        llm.config["streaming"] = streaming

        agent = Agent(
            model=llm,
            system_prompt=system_prompts,
            tools=tools,
            callback_handler=None # async iterator로 대체 하기 때문에 None 설정
        )

        return agent
    
    @staticmethod
    def get_agent_state(agent, key, default_value=None):
      """Strands Agent의 state에서 안전하게 값을 가져오는 메서드"""
      value = agent.state.get(key)
      if value is None: return default_value
      return value
      
    @staticmethod
    def get_agent_state_all(agent):
        return agent.state.get()
    
    @staticmethod
    def update_agent_state(agent, key, value):
        agent.state.set(key, value)
        #return agent
    
    @staticmethod
    def update_agent_state_all(target_agent, source_agent):
        """다른 에이전트의 state를 현재 에이전트에 복사"""
        source_state = source_agent.state.get()
        if source_state:
            for key, value in source_state.items():
                target_agent.state.set(key, value)
        return target_agent

    @staticmethod
    async def process_streaming_response(agent, message):
        callback_reasoning, callback_answer = ColoredStreamingCallback('purple'), ColoredStreamingCallback('white')
        response = {"text": "","reasoning": "", "signature": "", "tool_use": None, "cycle": 0}
        try:
            agent_stream = agent.stream_async(message)
            async for event in agent_stream:
                if "reasoningText" in event:
                    response["reasoning"] += event["reasoningText"]
                    callback_reasoning.on_llm_new_token(event["reasoningText"])
                elif "reasoning_signature" in event:
                    response["signature"] += event["reasoning_signature"]
                elif "data" in event:
                    response["text"] += event["data"]
                    callback_answer.on_llm_new_token(event["data"])
                elif "current_tool_use" in event and event["current_tool_use"].get("name"):
                    response["tool_use"] = event["current_tool_use"]["name"]
                    if "event_loop_metrics" in event:
                        if response["cycle"] != event["event_loop_metrics"].cycle_count:
                            response["cycle"] = event["event_loop_metrics"].cycle_count
                            callback_answer.on_llm_new_token(f' \n## Calling tool: {event["current_tool_use"]["name"]} - # Cycle: {event["event_loop_metrics"].cycle_count}\n')
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            logger.error(traceback.format_exc())  # Detailed error logging
        
        return agent, response
    
    @staticmethod
    async def process_streaming_response_agentcore(agent, message):
        callback_reasoning, callback_answer = ColoredStreamingCallback('purple'), ColoredStreamingCallback('white')
        response = {"text": "","reasoning": "", "signature": "", "tool_use": None, "cycle": 0}
        try:
            agent_stream = agent.stream_async(message)
            async for event in agent_stream:
                if "reasoningText" in event:
                    response["reasoning"] += event["reasoningText"]
                    callback_reasoning.on_llm_new_token(event["reasoningText"])
                elif "reasoning_signature" in event:
                    response["signature"] += event["reasoning_signature"]
                elif "data" in event:
                    response["text"] += event["data"]
                    callback_answer.on_llm_new_token(event["data"])
                elif "current_tool_use" in event and event["current_tool_use"].get("name"):
                    response["tool_use"] = event["current_tool_use"]["name"]
                    if "event_loop_metrics" in event:
                        if response["cycle"] != event["event_loop_metrics"].cycle_count:
                            response["cycle"] = event["event_loop_metrics"].cycle_count
                            callback_answer.on_llm_new_token(f' \n## Calling tool: {event["current_tool_use"]["name"]} - # Cycle: {event["event_loop_metrics"].cycle_count}\n')
                # Yield the event for AgentCore streaming
                #yield event
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            logger.error(traceback.format_exc())  # Detailed error logging
        
        return agent, response

    @staticmethod
    def parsing_text_from_response(response):

        """
        Usage (async iterator x): 
        agent = Agent()
        response = agent(query)
        response = strands_utils.parsing_text_from_response(response)
        """

        output = {}
        if len(response.message["content"]) == 2: ## reasoning
            output["reasoning"] = response.message["content"][0]["reasoningContent"]["reasoningText"]["text"]
            output["signature"] = response.message["content"][0]["reasoningContent"]["reasoningText"]["signature"]
        
        output["text"] = response.message["content"][-1]["text"]
    
        return output  