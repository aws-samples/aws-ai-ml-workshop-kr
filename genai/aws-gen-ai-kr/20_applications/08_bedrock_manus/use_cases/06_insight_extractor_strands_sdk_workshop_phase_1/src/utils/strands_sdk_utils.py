
import logging
import traceback
import asyncio
from datetime import datetime
from src.utils.bedrock import bedrock_info
from strands import Agent
from strands.models import BedrockModel
from botocore.config import Config
from botocore.exceptions import ClientError
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from strands.types.exceptions import EventLoopException

from strands.agent.agent_result import AgentResult
from strands.types.content import ContentBlock, Message, SystemContentBlock
from strands.multiagent.base import MultiAgentBase, NodeResult, MultiAgentResult, Status

from strands.agent.conversation_manager import SummarizingConversationManager
from src.prompts.template import apply_prompt_template

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

# Wrap agent with StreamableAgent for queue-based streaming (Agent as a tool 사용할 경우, tool response 또한 스트리밍 하기 위해서)
# Graph를 사용한다면 에이전트 마다 StreamableAgent를 감싸면 안된다. 그래프는 그래프 완성 후 StreamableGprah로 래핑함. 
class StreamableAgent:
    """Agent wrapper that adds streaming capability with event queue pattern."""

    def __init__(self, agent):
        """Wrap a Strands Agent to add queue-based streaming."""
        self.agent = agent
        # Expose common agent attributes for compatibility
        self.messages = agent.messages
        self.system_prompt = agent.system_prompt
        self.model = agent.model
        # Only expose tools if agent has it
        if hasattr(agent, 'tools'): self.tools = agent.tools

    def __getattr__(self, name):
        """Delegate attribute access to wrapped agent."""
        return getattr(self.agent, name)

    async def _cleanup_agent(self, agent_task):
        """Handle agent completion and cleanup."""
        if not agent_task.done():
            try: await asyncio.wait_for(agent_task, timeout=2.0)
            except asyncio.TimeoutError:
                agent_task.cancel()
                try: await agent_task
                except asyncio.CancelledError: pass

    async def _yield_pending_events(self):
        """Yield any pending events from global queue."""
        from src.utils.event_queue import get_event, has_events

        while has_events():
            event = get_event()
            if event: yield event

    async def stream_async_with_queue(self, message, agent_name=None, source=None):
        """
        Stream agent response using background task + event queue pattern.
        Following pattern from StreamableGraph.stream_async()

        Args:
            message: Message to send to agent
            agent_name: Name of the agent for event tagging (optional)
            source: Source identifier for the event (optional)

        Yields:
            Events from global queue (formatted for display)
        """
        from src.utils.event_queue import clear_queue

        # Use agent's name if not provided
        if agent_name is None: agent_name = getattr(self.agent, 'name', 'agent')
        if source is None: source = agent_name

        # Clear queue before starting
        clear_queue()

        # Step 1: Run agent in background - events go to global queue
        async def run_agent():
            try:
                full_text = ""
                async for event in strands_utils.process_streaming_response_yield(
                    self.agent, message, agent_name=agent_name, source=source
                ):
                    if event.get("event_type") == "text_chunk": full_text += event.get("data", "")
                return full_text
            except Exception as e:
                print(f"Agent error: {e}")
                raise

        agent_task = asyncio.create_task(run_agent())

        # Step 2: Consume events from global queue
        try:
            while not agent_task.done():
                async for event in self._yield_pending_events():
                    yield event
                await asyncio.sleep(0.005)
        finally:
            await self._cleanup_agent(agent_task)
            async for event in self._yield_pending_events():
                yield event

        yield {"type": "agent_complete", "event_type": "complete", "message": f"{agent_name} processing complete"}

class strands_utils():

    @staticmethod
    def get_model(**kwargs):

        llm_type = kwargs["llm_type"]
        enable_reasoning = kwargs["enable_reasoning"]
        tool_cache = kwargs["tool_cache"]

        if llm_type in ["claude-sonnet-3-7", "claude-sonnet-4", "claude-sonnet-4-5"]:
            
            if llm_type == "claude-sonnet-3-7": model_name = "Claude-V3-7-Sonnet-CRI"
            elif llm_type == "claude-sonnet-4": model_name = "Claude-V4-Sonnet-CRI"
            elif llm_type == "claude-sonnet-4-5": model_name = "Claude-V4-5-Sonnet-CRI"

            ## BedrockModel params: https://strandsagents.com/latest/api-reference/models/?h=bedrockmodel#strands.models.bedrock.BedrockModel
            llm = BedrockModel(
                model_id=bedrock_info.get_model_id(model_name=model_name),
                streaming=True,
                cache_tools="default" if tool_cache else None,
                max_tokens=8192*5,
                stop_sequences=["\n\nHuman"],
                temperature=1 if enable_reasoning else 0.01,
                additional_request_fields={
                    "thinking": {
                        "type": "enabled" if enable_reasoning else "disabled",
                        **({"budget_tokens": 8192} if enable_reasoning else {}),
                    }
                },
                # cache_prompt parameter removed - use SystemContentBlock with cachePoint instead
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
                # cache_prompt parameter removed - use SystemContentBlock with cachePoint instead
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
        agent_type = kwargs.get("agent_type", "claude-sonnet-4-5")
        enable_reasoning = kwargs.get("enable_reasoning", False)
        prompt_cache_info = kwargs.get("prompt_cache_info", (False, None)) # (True, "default")
        tool_cache = kwargs.get("tool_cache", False)
        tools = kwargs.get("tools", None)
        streaming = kwargs.get("streaming", True)
        
        # Context management parameters for SummarizingConversationManager
        context_overflow_summary_ratio = kwargs.get("context_overflow_summary_ratio", 0.5)  # Summarize 50% of older messages
        context_overflow_preserve_recent_messages = kwargs.get("context_overflow_preserve_recent_messages", 10)  # Keep recent 10 messages

        prompt_cache, cache_type = prompt_cache_info
        llm = strands_utils.get_model(llm_type=agent_type, enable_reasoning=enable_reasoning, tool_cache=tool_cache)
        llm.config["streaming"] = streaming

        # Convert system_prompt to SystemContentBlock array with cachePoint if caching is enabled
        if prompt_cache:
            logger.info(f"{Colors.GREEN}{agent_name.upper()} - Prompt Cache Enabled{Colors.END}")
            system_prompt_with_cache = [
                SystemContentBlock(text=system_prompts),
                SystemContentBlock(cachePoint={"type": cache_type})
            ]
        else:
            # If caching is disabled, pass the string as-is
            logger.info(f"{Colors.GREEN}{agent_name.upper()} - Prompt Cache Disabled{Colors.END}")
            system_prompt_with_cache = system_prompts
        
        if tool_cache: logger.info(f"{Colors.GREEN}{agent_name.upper()} - Tool Cache Enabled{Colors.END}")
        else: logger.info(f"{Colors.GREEN}{agent_name.upper()} - Tool Cache Disabled{Colors.END}")

        agent = Agent(
            model=llm,
            system_prompt=system_prompt_with_cache,
            tools=tools,
            conversation_manager=SummarizingConversationManager(
                summary_ratio=context_overflow_summary_ratio,
                preserve_recent_messages=context_overflow_preserve_recent_messages,
                summarization_system_prompt=apply_prompt_template(prompt_name="summarization", prompt_context={})
            ),
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
    async def _retry_agent_streaming(agent, message, max_attempts=5, base_delay=10):
        """
        Agent streaming with throttling retry logic

        Args:
            agent: The Strands agent instance
            message: Message to send to agent
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff

        Yields:
            Raw agent streaming events
        """
        for attempt in range(max_attempts):
            try:
                agent_stream = agent.stream_async(message)
                async for event in agent_stream:
                    yield event
                # If we get here, streaming was successful
                return

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

                # Retry for throttling errors
                if is_throttling and attempt < max_attempts - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    next_attempt = attempt + 2  # Next attempt number (attempt is 0-indexed)

                    logger.info(f"🔄 Throttling detected - Retry Step {attempt + 1}/{max_attempts}")
                    logger.info(f"⏱️  Waiting {delay} seconds before Step {next_attempt} retry...")
                    await asyncio.sleep(delay)
                    logger.info(f"🚀 Starting retry attempt {next_attempt}/{max_attempts}")
                    continue
                else:
                    # Only log errors on final attempt to reduce noise
                    if attempt == max_attempts - 1:
                        logger.error(f"Error in streaming response (attempt {attempt + 1}/{max_attempts}): {e}")
                        logger.error(traceback.format_exc())
                        raise  # Re-raise on final attempt
                    else:
                        # Silent retry for other errors
                        delay = base_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                        continue
            except Exception as e:
                logger.error(f"Unexpected error in streaming response: {e}")
                logger.error(traceback.format_exc())
                raise

    @staticmethod
    async def process_streaming_response_yield(agent, message, agent_name="coordinator", source=None):
        """
        Process streaming response from agent with event conversion and global queue management

        Args:
            agent: The Strands agent instance
            message: Message to send to agent
            agent_name: Name of the agent for event tagging
            source: Source identifier for the event

        Yields:
            AgentCore formatted events
        """
        from src.utils.event_queue import put_event

        session_id = "ABC"

        # Use retry helper for robust streaming
        async for event in strands_utils._retry_agent_streaming(agent, message):
            # Convert Strands events to AgentCore format
            agentcore_event = await strands_utils._convert_to_agentcore_event(event, agent_name, session_id, source)
            if agentcore_event:
                # Put event in global queue for unified processing
                put_event(agentcore_event)
                yield agentcore_event

        # After streaming completes, extract usage info from agent's metrics (에이전트 응답이 종료된 이후 최종적으로 한번만 보낸다)
        # Reference: https://strandsagents.com/latest/documentation/docs/user-guide/observability-evaluation/metrics/
        try:
            usage_info = None

            # Strands SDK stores token usage in agent.event_loop_metrics.accumulated_usage
            if hasattr(agent, 'event_loop_metrics'):
                metrics = agent.event_loop_metrics
                if hasattr(metrics, 'accumulated_usage'):
                    usage_info = metrics.accumulated_usage

            # If we found usage info, create and yield the event
            if usage_info:
                usage_event = {
                    "timestamp": datetime.now().isoformat(),
                    "session_id": session_id,
                    "agent_name": agent_name,
                    "source": source or f"{agent_name}_node",
                    "type": "agent_usage_stream",
                    "event_type": "usage_metadata",
                    "input_tokens": usage_info.get("inputTokens", 0),
                    "output_tokens": usage_info.get("outputTokens", 0),
                    "total_tokens": usage_info.get("totalTokens", 0),
                    "cache_read_input_tokens": usage_info.get("cacheReadInputTokens", 0),
                    "cache_write_input_tokens": usage_info.get("cacheWriteInputTokens", 0)
                }
                put_event(usage_event)
                yield usage_event

        except Exception as e:
            logger.warning(f"Could not extract usage info from {agent_name}: {e}")

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

        # 사용량/메타데이터 이벤트
        elif "metadata" in strands_event and "usage" in strands_event["metadata"]:
            usage = strands_event["metadata"]["usage"]
            return {
                **base_event,
                "type": "agent_usage_stream",
                "event_type": "usage_metadata",
                "input_tokens": usage.get("inputTokens", 0),
                "output_tokens": usage.get("outputTokens", 0),
                "total_tokens": usage.get("totalTokens", 0)
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
                
                elif tool_name == "rag_tool":
                    callback_tool.on_llm_new_token(f"rag response:\n{output}\n")

                else: # 기타 모든 툴 결과 표시, 코더 툴, 리포터 툴 결과도 다 출력 (for debug)
                    callback_tool.on_llm_new_token(f"Output: pass - you can see that in debug mode\n")
                    #callback_default.on_llm_new_token(f"Output: {output}\n")
                    #pass

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
    async def invoke_async(self, task=None, invocation_state=None, **kwargs):
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


# ============================================================================
# Token Tracking Helper Class
# ============================================================================

class TokenTracker:
    """Helper class for tracking and reporting token usage across agents."""

    # ANSI color codes
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    END = '\033[0m'

    @staticmethod
    def initialize(shared_state):
        """Initialize token tracking structure in shared state if not exists."""
        if 'token_usage' not in shared_state:
            shared_state['token_usage'] = {
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_tokens': 0,
                'cache_read_input_tokens': 0,   # Cache hits (90% discount)
                'cache_write_input_tokens': 0,  # Cache creation (25% extra cost)
                'by_agent': {}
            }

    @staticmethod
    def accumulate(event, shared_state):
        """Accumulate token usage from metadata events into shared state."""
        if event.get("event_type") == "usage_metadata":
            TokenTracker.initialize(shared_state)
            usage = shared_state['token_usage']

            input_tokens = event.get('input_tokens', 0)
            output_tokens = event.get('output_tokens', 0)
            total_tokens = event.get('total_tokens', 0)
            cache_read = event.get('cache_read_input_tokens', 0)
            cache_write = event.get('cache_write_input_tokens', 0)

            # Accumulate total tokens
            usage['total_input_tokens'] += input_tokens
            usage['total_output_tokens'] += output_tokens
            usage['total_tokens'] += total_tokens
            usage['cache_read_input_tokens'] += cache_read
            usage['cache_write_input_tokens'] += cache_write

            # Track by agent
            agent_name = event.get('agent_name')
            if agent_name:
                if agent_name not in usage['by_agent']:
                    usage['by_agent'][agent_name] = {
                        'input': 0,
                        'output': 0,
                        'cache_read': 0,
                        'cache_write': 0
                    }
                usage['by_agent'][agent_name]['input'] += input_tokens
                usage['by_agent'][agent_name]['output'] += output_tokens
                usage['by_agent'][agent_name]['cache_read'] += cache_read
                usage['by_agent'][agent_name]['cache_write'] += cache_write

    @staticmethod
    def print_current(shared_state):
        """Print current cumulative token usage."""
        token_usage = shared_state.get('token_usage', {})
        if token_usage and token_usage.get('total_tokens', 0) > 0:
            total_input = token_usage.get('total_input_tokens', 0)
            total_output = token_usage.get('total_output_tokens', 0)
            total = token_usage.get('total_tokens', 0)
            cache_read = token_usage.get('cache_read_input_tokens', 0)
            cache_write = token_usage.get('cache_write_input_tokens', 0)

            # Display breakdown showing total includes all token types
            print(f"{TokenTracker.CYAN}>>> Cumulative Tokens (Total: {total:,}):{TokenTracker.END}")
            print(f"{TokenTracker.CYAN}    Regular Input: {total_input:,} | Cache Read: {cache_read:,} (90% off) | Cache Write: {cache_write:,} (25% extra) | Output: {total_output:,}{TokenTracker.END}")