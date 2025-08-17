 
import pprint
import logging


import asyncio
#from strands.models import BedrockModel
#from src.utils.bedrock import bedrock_info
from src.utils.strands_sdk_utils import strands_utils
from src.prompts.template import apply_prompt_template

from strands.agent.agent_result import AgentResult
from strands.types.content import ContentBlock, Message
from strands.multiagent.base import MultiAgentBase, NodeResult, Status, MultiAgentResult

#from strands.multiagent import GraphBuilder


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

RESPONSE_FORMAT = "Response from {}:\n\n<response>\n{}\n</response>\n\n*Please execute the next step.*"
FULL_PLAN_FORMAT = "Here is full plan :\n\n<full_plan>\n{}\n</full_plan>\n\n*Please consider this to select the next step.*"
CLUES_FORMAT = "Here is clues from {}:\n\n<clues>\n{}\n</clues>\n\n"

class FunctionNode(MultiAgentBase):
    """Execute deterministic Python functions as graph nodes."""

    def __init__(self, func, name: str = None):
        super().__init__()
        self.func = func
        self.name = name or func.__name__

    def __call__(self, task, **kwargs):
        """Synchronous execution for compatibility with MultiAgentBase"""
        if asyncio.iscoroutinefunction(self.func):
            if isinstance(task, dict):
                return asyncio.run(self.func(**task))
            else:
                return asyncio.run(self.func(task if isinstance(task, str) else str(task)))
        else:
            if isinstance(task, dict):
                return self.func(**task)
            else:
                return self.func(task if isinstance(task, str) else str(task))

    async def invoke_async(self, task, **kwargs):
        # Execute function (nodes now use global state for data sharing)
        if asyncio.iscoroutinefunction(self.func):
            if isinstance(task, dict):
                agent, response = await self.func(**task)
            else:
                agent, response = await self.func(task if isinstance(task, str) else str(task))
        else:
            if isinstance(task, dict):
                agent, response = self.func(**task)
            else:
                agent, response = self.func(task if isinstance(task, str) else str(task))

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

def should_handoff_to_planner(state):
    """Check if coordinator requested handoff to planner."""
    # Check global shared state for goto field
    global _global_node_states
    shared_state = _global_node_states.get('shared', {})
    goto = shared_state.get('goto', '__end__')
    return goto == 'planner'

async def coordinator_node(**kwargs):
    """Coordinator node that communicate with customers."""
    logger.info(f"{Colors.GREEN}===== Coordinator talking...... ====={Colors.END}")

    # Extract user request from kwargs (for coordinator only)
    request = kwargs.get("request", "")
    request_prompt = kwargs.get("request_prompt", request)

    agent = strands_utils.get_agent(
        agent_name="coordinator",
        system_prompts=apply_prompt_template(prompt_name="coordinator", prompt_context={}), # apply_prompt_template(prompt_name="task_agent", prompt_context={"TEST": "sdsd"})
        agent_type="basic", #"reasoning", "basic"
        prompt_cache_info=(False, None), #(False, None), (True, "default")
        streaming=True,
    )
        
    agent, response = await strands_utils.process_streaming_response(agent, request_prompt)
    
    ## your logic here ##
    #logger.info(f"{Colors.GREEN}===== 여기서 무언가를 해도 되겠죠? ====={Colors.END}")
    
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
    
    return agent, response

async def planner_node(message=None):
    """Planner node that generates detailed plans for task execution."""
    logger.info(f"{Colors.GREEN}===== Planner generating plan ====={Colors.END}")
    
    # Log what we received  
    logger.debug(f"\n{Colors.YELLOW}Planner received message:\n{pprint.pformat(message, indent=2, width=100)}{Colors.END}")
    
    # Extract shared state from global storage
    global _global_node_states
    shared_state = _global_node_states.get('shared', None)
    request = shared_state.get("request", "")
    
    if shared_state:
        logger.info(f"{Colors.BLUE}===== Successfully retrieved shared state from global storage ====={Colors.END}")
        logger.debug(f"\n{Colors.YELLOW}Shared state:\n{pprint.pformat(shared_state, indent=2, width=100)}{Colors.END}")
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
    shared_state['messages'] = agent.messages
    shared_state['goto'] = goto
    shared_state['full_plan'] = response["text"]  # store the generated plan
    shared_state['history'].append({"agent":"planner", "message": response["text"]})

    logger.debug(f"\n{Colors.RED}Updated shared state:\n{pprint.pformat(shared_state, indent=2, width=100)}{Colors.END}")
    logger.info(f"{Colors.GREEN}===== Planner completed plan generation ====={Colors.END}")
    
    return agent, response