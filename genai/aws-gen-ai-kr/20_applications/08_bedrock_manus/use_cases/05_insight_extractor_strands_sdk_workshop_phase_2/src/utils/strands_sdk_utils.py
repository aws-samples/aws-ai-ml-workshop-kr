
import logging
import traceback
import asyncio
import time
from src.utils.bedrock import bedrock_info
from src.utils.common_utils import retry
from strands import Agent, tool
from strands.models import BedrockModel
from botocore.config import Config
from botocore.exceptions import ClientError
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from strands.types.exceptions import EventLoopException

from datetime import datetime

from strands.agent.agent_result import AgentResult
from strands.types.content import ContentBlock, Message
from strands.multiagent.base import MultiAgentBase, NodeResult, MultiAgentResult, Status

# Simple logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

        if llm_type == "claude-sonnet-3-7":    
            ## BedrockModel params: https://strandsagents.com/latest/api-reference/models/?h=bedrockmodel#strands.models.bedrock.BedrockModel
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
                cache_prompt=cache_type, # None/ephemeral/defalut
                #cache_tools: Cache point type for tools
                boto_client_config=Config(
                    read_timeout=900,
                    connect_timeout=900,
                    retries=dict(max_attempts=50, mode="adaptive"),
                )
            )   
        elif llm_type == "claude-sonnet-3-5-v-2":
            ## BedrockModel params: https://strandsagents.com/latest/api-reference/models/?h=bedrockmodel#strands.models.bedrock.BedrockModel
            llm = BedrockModel(
                model_id=bedrock_info.get_model_id(model_name="Claude-V3-5-V-2-Sonnet-CRI"),
                streaming=True,
                max_tokens=8192,
                stop_sequences=["\n\nHuman"],
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
        agent_type = kwargs.get("agent_type", "claude-sonnet-3-7")
        enable_reasoning = kwargs.get("enable_reasoning", False)
        prompt_cache_info = kwargs.get("prompt_cache_info", (False, None)) # (True, "default")
        tools = kwargs.get("tools", None)
        streaming = kwargs.get("streaming", True)

        prompt_cache, cache_type = prompt_cache_info
        if prompt_cache: logger.info(f"{Colors.GREEN}{agent_name.upper()} - Prompt Cache Enabled{Colors.END}")
        else: logger.info(f"{Colors.GREEN}{agent_name.upper()} - Prompt Cache Disabled{Colors.END}")

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
    async def process_streaming_response_yield(agent, message, agent_name="coordinator", source=None):
        from src.utils.event_queue import put_event
        
        # Retry configuration
        max_attempts = 5
        base_delay = 30  # seconds
        
        for attempt in range(max_attempts):
            try:
                session_id = "ABC"
                agent_stream = agent.stream_async(message)

                async for event in agent_stream:
                    # Strands 이벤트를 AgentCore 형식으로 변환
                    agentcore_event = await strands_utils._convert_to_agentcore_event(event, agent_name, session_id, source)
                    if agentcore_event: 
                        # Put event in global queue for unified processing
                        put_event(agentcore_event)
                        yield agentcore_event
                
                # If we get here, streaming was successful
                break
                
            except (EventLoopException, ClientError) as e:
                # Check if it's a throttling error
                is_throttling = False
                if isinstance(e, EventLoopException):
                    # Check if the underlying error is throttling
                    error_msg = str(e).lower()
                    is_throttling = 'throttling' in error_msg or 'too many requests' in error_msg
                elif isinstance(e, ClientError):
                    error_code = e.response.get('Error', {}).get('Code', '')
                    is_throttling = error_code == 'ThrottlingException'
                
                if is_throttling and attempt < max_attempts - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    next_attempt = attempt + 2  # Next attempt number (attempt is 0-indexed)
                    logger.warning(f"🔄 Throttling detected - Retry Step {attempt + 1}/{max_attempts}")
                    logger.warning(f"⏱️  Waiting {delay} seconds before Step {next_attempt} retry...")
                    await asyncio.sleep(delay)
                    logger.info(f"🚀 Starting retry attempt {next_attempt}/{max_attempts}")
                    continue
                else:
                    logger.error(f"Error in streaming response (attempt {attempt + 1}/{max_attempts}): {e}")
                    logger.error(traceback.format_exc())
                    if attempt == max_attempts - 1:
                        raise  # Re-raise on final attempt
            except Exception as e:
                logger.error(f"Unexpected error in streaming response: {e}")
                logger.error(traceback.format_exc())
                raise

    # 툴 사용 ID와 툴 이름 매핑을 위한 클래스 변수
    _tool_use_mapping = {}

    @staticmethod
    async def _convert_to_agentcore_event(strands_event, agent_name, session_id, source=None):
        """Strands 이벤트를 AgentCore 스트리밍 형식으로 변환"""

        base_event = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "agent_name": agent_name,
            "source": source or f"{agent_name}_node",
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
            tool_id = tool_info.get("toolUseId")
            tool_name = tool_info.get("name", "unknown")

            # toolUseId와 tool_name 매핑 저장
            if tool_id and tool_name: strands_utils._tool_use_mapping[tool_id] = tool_name

            return {
                **base_event,
                "type": "agent_tool_stream",
                "event_type": "tool_use",
                "tool_name": tool_name,
                "tool_id": tool_id,
                "tool_input": tool_info.get("input", {})
            }

        # message 래퍼 안의 tool result 처리
        if "message" in strands_event:
            message = strands_event["message"]
            if isinstance(message, dict) and "content" in message and isinstance(message["content"], list):
                for content_item in message["content"]:
                    if isinstance(content_item, dict) and "toolResult" in content_item:
                        tool_result = content_item["toolResult"]
                        tool_id = tool_result.get("toolUseId")

                        # 저장된 매핑에서 툴 이름 찾기
                        tool_name = strands_utils._tool_use_mapping.get(tool_id, "external_tool")
                        output = str(tool_result.get("content", [{}])[0].get("text", "")) if tool_result.get("content") else ""

                        return {
                            **base_event,
                            "type": "agent_tool_stream",
                            "event_type": "tool_result", 
                            "tool_name": tool_name,
                            "tool_id": tool_id,
                            "output": output
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

    #########################
    ## modification STRART ##
    #########################

    @staticmethod
    def process_event_for_display(event):
        """Process events for colored terminal output"""
        # Initialize colored callbacks for terminal display
        callback_default = ColoredStreamingCallback('white')
        callback_reasoning = ColoredStreamingCallback('cyan')        
        callback_tool = ColoredStreamingCallback('yellow')

        if event:
            if event.get("event_type") == "text_chunk":
                callback_default.on_llm_new_token(event.get('data', ''))

            elif event.get("event_type") == "reasoning":
                callback_reasoning.on_llm_new_token(event.get('reasoning_text', ''))

            elif event.get("event_type") == "tool_use": 
                pass

            elif event.get("event_type") == "tool_result":
                tool_name = event.get("tool_name", "unknown")
                output = event.get("output", "")
                print(f"\n[TOOL RESULT - {tool_name}]", flush=True)

                # Parse output based on function name
                if tool_name == "python_repl_tool" and len(output.split("||")) == 3:
                    status, code, stdout = output.split("||")
                    callback_tool.on_llm_new_token(f"Status: {status}\n")

                    if code: callback_tool.on_llm_new_token(f"Code:\n```python\n{code}\n```\n")
                    if stdout and stdout != 'None': callback_tool.on_llm_new_token(f"Output:\n{stdout}\n")

                elif tool_name == "bash_tool" and len(output.split("||")) == 2:
                    cmd, stdout = output.split("||")
                    if cmd: callback_tool.on_llm_new_token(f"CMD:\n```bash\n{cmd}\n```\n")
                    if stdout and stdout != 'None': callback_tool.on_llm_new_token(f"Output:\n{stdout}\n")

                elif tool_name == "file_read":
                    # file_read 결과는 보통 길어서 앞부분만 표시
                    truncated_output = output[:500] + "..." if len(output) > 500 else output
                    callback_tool.on_llm_new_token(f"File content preview:\n{truncated_output}\n")

                else: # 기타 모든 툴 결과 표시, 코더 툴, 리포터 툴 결과도 다 출력 (for debug)
                    callback_tool.on_llm_new_token(f"Output: pass - you can see that in debug mode\n")
                    #callback_default.on_llm_new_token(f"Output: {output}\n")
                    #pass

    #########################
    ## modification END    ##
    #########################

class FunctionNode(MultiAgentBase):
    """Execute deterministic Python functions as graph nodes."""

    def __init__(self, func, name: str = None):
        super().__init__()
        self.func = func
        self.name = name or func.__name__

    def __call__(self, task=None, **kwargs):
        """Synchronous execution for compatibility with MultiAgentBase"""
        # Pass task and kwargs directly to function
        if asyncio.iscoroutinefunction(self.func): 
            return asyncio.run(self.func(task=task, **kwargs))
        else: 
            return self.func(task=task, **kwargs)

    # Execute function and return standard MultiAgentResult
    async def invoke_async(self, task=None, **kwargs):
        # Execute function (nodes now use global state for data sharing)  
        # Pass task and kwargs directly to function
        if asyncio.iscoroutinefunction(self.func): 
            response = await self.func(task=task, **kwargs)
        else: 
            response = self.func(task=task, **kwargs)

        agent_result = AgentResult(
            stop_reason="end_turn",
            message=Message(role="assistant", content=[ContentBlock(text=str(response["text"]))]),
            metrics={},
            state={}
        )

        # Return wrapped in MultiAgentResult
        return MultiAgentResult(
            status=Status.COMPLETED,
            results={self.name: NodeResult(result=agent_result)}
        )




