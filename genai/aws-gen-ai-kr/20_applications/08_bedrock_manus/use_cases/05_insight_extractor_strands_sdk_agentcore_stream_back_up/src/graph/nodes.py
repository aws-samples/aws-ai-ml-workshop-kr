 
import pprint
import logging

import asyncio
from src.utils.strands_sdk_utils import strands_utils
from src.prompts.template import apply_prompt_template
from src.utils.common_utils import get_message_from_string

from src.tools import coder_agent_tool, reporter_agent_tool, tracker_agent_tool

from strands.agent.agent_result import AgentResult
from strands.types.content import ContentBlock, Message
from strands.multiagent.base import MultiAgentBase, NodeResult, Status, MultiAgentResult

# 새 핸들러와 포맷터 설정
logger = logging.getLogger(__name__)
logger.propagate = False  # 상위 로거로 메시지 전파 중지
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
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

# Global state storage for sharing between nodes
_global_node_states = {}

# Global callback for streaming events
_streaming_callback = None

def set_streaming_callback(callback):
    """Set a callback function to handle streaming events"""
    global _streaming_callback
    _streaming_callback = callback

RESPONSE_FORMAT = "Response from {}:\n\n<response>\n{}\n</response>\n\n*Please execute the next step.*"
FULL_PLAN_FORMAT = "Here is full plan :\n\n<full_plan>\n{}\n</full_plan>\n\n*Please consider this to select the next step.*"
CLUES_FORMAT = "Here is clues from {}:\n\n<clues>\n{}\n</clues>\n\n"

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

    # 여기서 invoke_async_streaming는 FunctionNode의 invoke_async을 대체하는 것이다.
    # 원래 strands로 graph를 만들고 invoke (혹은 invoke_async)를 하게 되면 FunctionNode의 invoke_async가 실행되는데, 
    # 여기에서는 yeild기반 스트리밍을 위해 만들어진 그래프를 StreamingGraphWrapper로 감싸기 때문에 invoke_async_streaming가 실행되는 것이다. 
    # StreamingFunctionNode 과 합쳐보려고 했으나, yield로 streaming 할 때는 return 값을 보낼 수 없기에 합치지 않았음
    # 추가로 FunctionNode에는 MultiAgentResult return하는 부분이 있지만 StreamingFunctionNode에는 retrun이 없음 (yield 이기 때문)
    # 즉, streaming 시 각 노드간에 task 형태로 리턴되는 것은 없고, 전달되는 값은 모두 shared var로 하고 있음
    async def invoke_async(self, task=None, **kwargs):
        # Execute function (nodes now use global state for data sharing)  
        # Pass task and kwargs directly to function
        if asyncio.iscoroutinefunction(self.func): 
            agent, response = await self.func(task=task, **kwargs)
        else: 
            agent, response = self.func(task=task, **kwargs)

        agent_result = AgentResult(
            stop_reason="end_turn",
            message=Message(role="assistant", content=[ContentBlock(text=str(response["text"]))]),
            metrics={},
            state=strands_utils.get_agent_state_all(agent)
        )

        # Return wrapped in MultiAgentResult
        return MultiAgentResult(
            status=Status.COMPLETED,
            results={self.name: NodeResult(result=agent_result)},
            # ... execution details
        )

class StreamingFunctionNode:
    """Streaming version of FunctionNode that yields events in real-time."""

    def __init__(self, func, name: str = None):
        self.func = func
        self.name = name or func.__name__

    # 여기서 invoke_async_streaming는 FunctionNode의 invoke_async을 대체하는 것이다.
    # 원래 strands로 graph를 만들고 invoke (혹은 invoke_async)를 하게 되면 FunctionNode의 invoke_async가 실행되는데, 
    # 여기에서는 yeild기반 스트리밍을 위해 만들어진 그래프를 StreamingGraphWrapper로 감싸기 때문에 invoke_async_streaming가 실행되는 것이다. 
    # StreamingFunctionNode 과 합쳐보려고 했으나, yield로 streaming 할 때는 return 값을 보낼 수 없기에 합치지 않았음
    # 추가로 FunctionNode에는 MultiAgentResult return하는 부분이 있지만 StreamingFunctionNode에는 retrun이 없음 (yield 이기 때문)
    # 즉, streaming 시 각 노드간에 task 형태로 리턴되는 것은 없고, 전달되는 값은 모두 shared var로 하고 있음
    async def invoke_async_streaming(self, task=None, **kwargs):
        """Execute function and yield streaming events in real-time"""
        if asyncio.iscoroutinefunction(self.func): 
            result = await self.func(task=task, **kwargs)
        else: 
            result = self.func(task=task, **kwargs)
        
        # Check if result is an async generator (streaming case)
        if hasattr(result, '__aiter__'):
            # For streaming functions, yield events in real-time
            async for event in result:
                #print(f"StreamingNode yielding: {event}")
                yield event
                
                # Also call the global streaming callback if set (없으면 skip, 콜백 함수 만들어서 사용도 가능)
                global _streaming_callback
                if _streaming_callback: _streaming_callback(event)
        else:
            # Normal case: just return the result
            _, response = result
            yield {
                "type": "final_result",
                "agent": self.name,
                "response": response
            }

def should_handoff_to_planner(state):
    """Check if coordinator requested handoff to planner."""
    # Check coordinator's response for handoff request
    global _global_node_states
    shared_state = _global_node_states.get('shared', {})
    history = shared_state.get('history', [])
    
    # Look for coordinator's last message
    for entry in reversed(history):
        if entry.get('agent') == 'coordinator':
            message = entry.get('message', '')
            return 'handoff_to_planner' in message
    
    return False


async def coordinator_node(task=None, **kwargs):
    """Coordinator node that communicate with customers."""
    logger.info(f"{Colors.GREEN}===== Coordinator talking...... ====={Colors.END}")

    # Extract user request from task (now passed as dictionary)
    if isinstance(task, dict):
        request = task.get("request", "")
        request_prompt = task.get("request_prompt", request)
    else:
        request = str(task) if task else ""
        request_prompt = request

    agent = strands_utils.get_agent(
        agent_name="coordinator",
        system_prompts=apply_prompt_template(prompt_name="coordinator", prompt_context={}), # apply_prompt_template(prompt_name="task_agent", prompt_context={"TEST": "sdsd"})
        agent_type="basic", #"reasoning", "basic"
        prompt_cache_info=(False, None), #(False, None), (True, "default")
        streaming=True,
    )
        
    # Collect streaming events and build final response
    streaming_events = []
    async for event in strands_utils.process_streaming_response_yield(agent, request_prompt, agent_name="coordinator"):
        streaming_events.append(event)
        yield(event)

    # Reconstruct the full response from streaming events
    full_text = ""
    for event in streaming_events:
        if event.get("event_type") == "text_chunk":
            full_text += event.get("data", "")
    
    response = {"text": full_text}
    
    ## your logic here ##
    logger.debug(f"\n{Colors.RED}Current state messages:\n{pprint.pformat(agent.messages[:-1], indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Coordinator response:\n{pprint.pformat(response["text"], indent=2, width=100)}{Colors.END}")

    # Store data directly in shared global storage
    global _global_node_states
    
    # Initialize shared state if not exists
    if 'shared' not in _global_node_states: _global_node_states['shared'] = {}
    shared_state = _global_node_states['shared']
    
    # Update shared global state (remove goto, let graph handle routing)
    shared_state['messages'] = agent.messages
    shared_state['request'] = request
    shared_state['request_prompt'] = request_prompt
    
    # Build and update history
    if 'history' not in shared_state: shared_state['history'] = []
    shared_state['history'].append({"agent":"coordinator", "message": response["text"]})
    
    logger.debug(f"\n{Colors.RED}Shared global state:\n{pprint.pformat(shared_state, indent=2, width=100)}{Colors.END}")
    logger.info(f"{Colors.BLUE}===== Coordinator updated shared global state with {len(shared_state)} keys ====={Colors.END}")
    logger.info(f"{Colors.GREEN}===== Coordinator completed task ====={Colors.END}")
    
    # Note: Can't return values in async generators, so final result handled through global state
    # The invoke_async method will collect all events and create the final response

async def planner_node(task=None, **kwargs):
    """Planner node that generates detailed plans for task execution."""
    logger.info(f"{Colors.GREEN}===== Planner generating plan ====={Colors.END}")
    
    # Log what we received  
    logger.info(f"\n{Colors.YELLOW}Planner received task:\n{pprint.pformat(task, indent=2, width=100)}{Colors.END}")
    # FunctionNode를 통해 return되는 것은 이전 노드의 결과물(text)만 가능. 다른 것을 보내는 것은 아직 안되는 것 같음
    logger.info(f"\n{Colors.YELLOW}Planner received kwargs:\n{pprint.pformat(kwargs, indent=2, width=100)}{Colors.END}")

    # Extract shared state from global storage
    global _global_node_states
    shared_state = _global_node_states.get('shared', None)
    
    # Get request from kwargs state or shared state
    state = kwargs.get("state", {})
    if state:
        request = state.get("request", "")
    else:
        request = shared_state.get("request", "") if shared_state else (task or "")
    
    if shared_state:
        logger.info(f"{Colors.BLUE}===== Successfully retrieved shared state from global storage ====={Colors.END}")
        logger.info(f"\n{Colors.YELLOW}Shared state:\n{pprint.pformat(shared_state, indent=2, width=100)}{Colors.END}")
    else:
        logger.warning(f"{Colors.RED}No shared state found in global storage{Colors.END}")
        logger.debug(f"Global states available: {list(_global_node_states.keys())}")

    agent = strands_utils.get_agent(
        agent_name="planner",
        system_prompts=apply_prompt_template(prompt_name="planner", prompt_context={"USER_REQUEST": request}),
        agent_type="basic",  # planner uses reasoning LLM reasoning
        #prompt_cache_info=(True, "default"),  # enable prompt caching for reasoning agent
        prompt_cache_info=(False, None), #(False, None), (True, "default")
        #prompt_cache_info=(True, "default"),  # enable prompt caching for reasoning agent
        streaming=True,
    )
    
    full_plan, messages = shared_state.get("full_plan", ""), shared_state["messages"]
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], FULL_PLAN_FORMAT.format(full_plan)])
    
    # Collect streaming events and build final response
    streaming_events = []
    async for event in strands_utils.process_streaming_response_yield(agent, message, agent_name="planner"):
        #print(f"planner_node: {event}")
        streaming_events.append(event)
        yield(event)

    # Reconstruct the full response from streaming events
    full_text = ""
    for event in streaming_events:
        if event.get("event_type") == "text_chunk":
            full_text += event.get("data", "")
    
    response = {"text": full_text}
    
    ## Planner logic: create detailed plan with agent assignments ##
    logger.info(f"{Colors.GREEN}===== Planner analyzing and creating execution plan ====={Colors.END}")
    logger.debug(f"\n{Colors.RED}Planner input message:\n{pprint.pformat(message, indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Planner response:\n{pprint.pformat(response["text"], indent=2, width=100)}{Colors.END}")

    # Update shared global state (remove goto, let graph handle routing)
    shared_state['messages'] = [get_message_from_string(role="user", string=response["text"], imgs=[])]
    shared_state['full_plan'] = response["text"]  # store the generated plan
    shared_state['history'].append({"agent":"planner", "message": response["text"]})

    logger.debug(f"\n{Colors.RED}Updated shared state:\n{pprint.pformat(shared_state, indent=2, width=100)}{Colors.END}")
    logger.info(f"{Colors.GREEN}===== Planner completed plan generation ====={Colors.END}")
    
    # Note: Can't return values in async generators, so final result handled through global state

async def supervisor_node(task=None, **kwargs):
    """Supervisor node that decides which agent should act next."""
    logger.info(f"{Colors.GREEN}===== Supervisor evaluating next action ====={Colors.END}")
    
    # Log what we received  
    logger.info(f"\n{Colors.YELLOW}Supervisor received task:\n{pprint.pformat(task, indent=2, width=100)}{Colors.END}")
    logger.info(f"\n{Colors.YELLOW}Supervisor received kwargs:\n{pprint.pformat(kwargs, indent=2, width=100)}{Colors.END}")
    
    # Extract shared state from global storage
    global _global_node_states
    shared_state = _global_node_states.get('shared', None)
    
    if shared_state:
        logger.info(f"{Colors.YELLOW}===== Successfully retrieved shared state from global storage ====={Colors.END}")
        logger.info(f"\n{Colors.YELLOW}Shared state:\n{pprint.pformat(shared_state, indent=2, width=100)}{Colors.END}")
    else:
        logger.warning(f"{Colors.RED}No shared state found in global storage{Colors.END}")
        logger.debug(f"Global states available: {list(_global_node_states.keys())}")

    agent = strands_utils.get_agent(
        agent_name="supervisor",
        system_prompts=apply_prompt_template(prompt_name="supervisor", prompt_context={}),
        agent_type="reasoning",  #"reasoning", "basic"
        prompt_cache_info=(True, "default"),  # enable prompt caching for reasoning agent
        tools=[coder_agent_tool, reporter_agent_tool, tracker_agent_tool],  # Add coder, reporter and tracker agents as tools
        streaming=True,
    )

    clues, full_plan, messages = shared_state.get("clues", ""), shared_state.get("full_plan", ""), shared_state["messages"]
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], FULL_PLAN_FORMAT.format(full_plan), clues])
    
    # Collect streaming events and build final response
    streaming_events = []
    async for event in strands_utils.process_streaming_response_yield(agent, message, agent_name="supervisor"):
        #print(f"supervisor_node: {event}")
        streaming_events.append(event)
        yield(event)

    # Reconstruct the full response from streaming events
    full_text = ""
    for event in streaming_events:
        if event.get("event_type") == "text_chunk":
            full_text += event.get("data", "")
        elif event.get("event_type") == "reasoning":
            full_text += event.get("reasoning_text", "")
    
    response = {"text": full_text}

    #full_response = response["text"]
    logger.debug(f"\n{Colors.RED}Supervisor - current state messages:\n{pprint.pformat(shared_state['messages'], indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Supervisor response:\n{pprint.pformat(response["text"], indent=2, width=100)}{Colors.END}")
    
    logger.info(f"\n{Colors.GREEN}===== Workflow completed ====={Colors.END}")

    # Update shared global state (remove goto, supervisor is the end node)
    shared_state['history'].append({"agent":"supervisor", "message": response["text"]})
    
    logger.debug(f"\n{Colors.RED}Updated shared state:\n{pprint.pformat(shared_state, indent=2, width=100)}{Colors.END}")
    logger.info(f"{Colors.GREEN}===== Supervisor completed task ====={Colors.END}")
    
    # Note: Can't return values in async generators, so final result handled through global state


