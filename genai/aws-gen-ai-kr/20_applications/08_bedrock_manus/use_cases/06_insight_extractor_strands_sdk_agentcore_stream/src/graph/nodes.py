 
import pprint
import logging
import json

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

    async def invoke_async(self, task=None, **kwargs):
        # Execute function (nodes now use global state for data sharing)  
        # Pass task and kwargs directly to function
        if asyncio.iscoroutinefunction(self.func): 
            result = await self.func(task=task, **kwargs)
        else: 
            result = self.func(task=task, **kwargs)
        
        # Check if result is an async generator (streaming case)
        if hasattr(result, '__aiter__'):
            # For streaming functions, collect all events
            events = []
            async for event in result:
                events.append(event)
                print(f"Streaming event: {event}")
                
                # Call the global streaming callback if set
                global _streaming_callback
                if _streaming_callback:
                    _streaming_callback(event)
            
            # Reconstruct response text from streaming events
            full_text = ""
            for event in events:
                if event.get("event_type") == "text_chunk":
                    full_text += event.get("data", "")
            
            response = {"text": full_text}
            agent = None  # Agent might not be available in streaming case
        else:
            # Normal case: unpack agent and response
            agent, response = result

        # Create agent result
        if agent:
            agent_result = AgentResult(
                stop_reason="end_turn",
                message=Message(role="assistant", content=[ContentBlock(text=str(response["text"]))]),
                metrics={},
                state=strands_utils.get_agent_state_all(agent)
            )
        else:
            # Fallback for streaming case without agent
            agent_result = AgentResult(
                stop_reason="end_turn",
                message=Message(role="assistant", content=[ContentBlock(text=str(response["text"]))]),
                metrics={},
                state={}
            )

        # Return wrapped in MultiAgentResult
        return MultiAgentResult(
            status=Status.COMPLETED,
            results={self.name: NodeResult(result=agent_result)},
            # ... execution details
        )

def should_handoff_to_planner(state):
    """Check if coordinator requested handoff to planner."""
    # Check global shared state for goto field
    global _global_node_states
    shared_state = _global_node_states.get('shared', {})
    goto = shared_state.get('goto', '__end__')
    return goto == 'planner'


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
    async for event in strands_utils.process_streaming_response_yield(agent, request_prompt):
        print(f"in_node: {event}")
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

    goto = "__end__"
    if "handoff_to_planner" in response["text"]: goto = "planner"
    
    # Store data directly in shared global storage
    global _global_node_states
    
    # Initialize shared state if not exists
    if 'shared' not in _global_node_states: _global_node_states['shared'] = {}
    shared_state = _global_node_states['shared']
    
    # Update shared global state
    shared_state['messages'] = agent.messages
    shared_state['goto'] = goto
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
        agent_type="reasoning",  # planner uses reasoning LLM
        prompt_cache_info=(True, "default"),  # enable prompt caching for reasoning agent
        streaming=True,
    )
    
    full_plan, messages = shared_state.get("full_plan", ""), shared_state["messages"]
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], FULL_PLAN_FORMAT.format(full_plan)])
    agent, response = await strands_utils.process_streaming_response(agent, message)
    
    ## Planner logic: create detailed plan with agent assignments ##
    logger.info(f"{Colors.GREEN}===== Planner analyzing and creating execution plan ====={Colors.END}")
    logger.debug(f"\n{Colors.RED}Planner input message:\n{pprint.pformat(message, indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Planner response:\n{pprint.pformat(response["text"], indent=2, width=100)}{Colors.END}")

    # Planner determines next agent based on the plan
    goto = "supervisor"
    
    # Update shared global state
    shared_state['messages'] = [get_message_from_string(role="user", string=response["text"], imgs=[])]
    shared_state['goto'] = goto
    shared_state['full_plan'] = response["text"]  # store the generated plan
    shared_state['history'].append({"agent":"planner", "message": response["text"]})

    logger.debug(f"\n{Colors.RED}Updated shared state:\n{pprint.pformat(shared_state, indent=2, width=100)}{Colors.END}")
    logger.info(f"{Colors.GREEN}===== Planner completed plan generation ====={Colors.END}")
    
    return agent, response

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
    agent, response = await strands_utils.process_streaming_response(agent, message)

    #full_response = response["text"]
    if response["text"].startswith("```json"): response["text"] = response["text"].removeprefix("```json")
    if response["text"].endswith("```"): response["text"] = response["text"].removesuffix("```")
    
    response["text"] = json.loads(response["text"])   
    goto = response["text"]["next"]

    logger.debug(f"\n{Colors.RED}Supervisor - current state messages:\n{pprint.pformat(shared_state['messages'], indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Supervisor response:{response["text"]}{Colors.END}")

    if goto == "FINISH":
        goto = "__end__"
        logger.info(f"\n{Colors.GREEN}===== Workflow completed ====={Colors.END}")
    else:
        logger.info(f"{Colors.GREEN}Supervisor delegating to: {goto}{Colors.END}")

    # Update shared global state
    shared_state['goto'] = goto
    shared_state['history'].append({"agent":"supervisor", "message": response["text"]})
    
    logger.debug(f"\n{Colors.RED}Updated shared state:\n{pprint.pformat(shared_state, indent=2, width=100)}{Colors.END}")
    logger.info(f"{Colors.GREEN}===== Supervisor completed task ====={Colors.END}")
    
    return agent, response


