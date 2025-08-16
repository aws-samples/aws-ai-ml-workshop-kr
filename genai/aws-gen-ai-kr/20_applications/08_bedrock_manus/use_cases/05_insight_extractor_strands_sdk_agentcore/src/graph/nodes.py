 
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

class FunctionNode(MultiAgentBase):
    """Execute deterministic Python functions as graph nodes."""

    def __init__(self, func, name: str = None):
        super().__init__()
        self.func = func
        self.name = name or func.__name__

    def __call__(self, task, **kwargs):
        """Synchronous execution for compatibility with MultiAgentBase"""
        if asyncio.iscoroutinefunction(self.func):
            return asyncio.run(self.func(task if isinstance(task, str) else str(task)))
        return self.func(task if isinstance(task, str) else str(task))

    async def invoke_async(self, task, **kwargs):
        # Execute function (nodes now use global state for data sharing)
        if asyncio.iscoroutinefunction(self.func):
            agent, response = await self.func(task if isinstance(task, str) else str(task))
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
    
async def coordinator_node(message: str):

    """Coordinator node that communicate with customers."""
    logger.info(f"{Colors.GREEN}===== Coordinator talking...... ====={Colors.END}")

    agent = strands_utils.get_agent(
        agent_name="coordinator",
        system_prompts=apply_prompt_template(prompt_name="coordinator", prompt_context={}), # apply_prompt_template(prompt_name="task_agent", prompt_context={"TEST": "sdsd"})
        agent_type="basic", #"reasoning", "basic"
        prompt_cache_info=(False, None), #(False, None), (True, "default")
        streaming=True,
    )
        
    agent, response = await strands_utils.process_streaming_response(agent, message)
    
    ## your logic here ##
    logger.info(f"{Colors.GREEN}===== 여기서 무언가를 해도 되겠죠? ====={Colors.END}")
    
    logger.debug(f"\n{Colors.RED}Current state messages:\n{pprint.pformat(agent.messages[:-1], indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Coordinator response:\n{pprint.pformat(response["text"], indent=2, width=100)}{Colors.END}")

    goto = "__end__"
    if "handoff_to_planner" in response["text"]: goto = "planner"
    
    # Prepare state data for next node
    strands_utils.update_agent_state(agent, "messages", agent.messages)
    strands_utils.update_agent_state(agent, "goto", goto)
    history = strands_utils.get_agent_state(agent, "history", [])
    history.append({"agent":"coordinator", "message": response["text"]})
    strands_utils.update_agent_state(agent, "history", history)
    
    # Add additional context for planner
    strands_utils.update_agent_state(agent, "user_request", message)
    strands_utils.update_agent_state(agent, "coordinator_response", response["text"])
    
    current_state = strands_utils.get_agent_state_all(agent)
    logger.debug(f"\n{Colors.RED}Coordinator state to pass:\n{pprint.pformat(current_state, indent=2, width=100)}{Colors.END}")
    
    # Store coordinator state in global storage for next nodes
    global _global_node_states
    _global_node_states['coordinator'] = current_state
    
    logger.info(f"{Colors.BLUE}===== Coordinator passing state with {len(current_state)} keys via global storage ====={Colors.END}")
    logger.info(f"{Colors.GREEN}===== Coordinator completed task ====={Colors.END}")
    
    return agent, response

async def planner_node(message: str):
    """Planner node that generates detailed plans for task execution."""
    logger.info(f"{Colors.GREEN}===== Planner generating plan ====={Colors.END}")
    
    # Log what we received
    logger.debug(f"{Colors.YELLOW}Planner received message: {message}{Colors.END}")
    
    # Extract coordinator state from global storage
    global _global_node_states
    coordinator_state = _global_node_states.get('coordinator', None)
    
    if coordinator_state:
        logger.info(f"{Colors.BLUE}===== Successfully retrieved coordinator state from global storage ====={Colors.END}")
        logger.debug(f"{Colors.YELLOW}Coordinator state keys: {list(coordinator_state.keys())}{Colors.END}")
    else:
        logger.warning(f"{Colors.RED}No coordinator state found in global storage{Colors.END}")
        logger.debug(f"Global states available: {list(_global_node_states.keys())}")

    agent = strands_utils.get_agent(
        agent_name="planner",
        system_prompts=apply_prompt_template(prompt_name="planner", prompt_context={}),
        agent_type="reasoning",  # planner uses reasoning LLM
        prompt_cache_info=(True, "default"),  # enable prompt caching for reasoning agent
        streaming=True,
    )
    
    # Load coordinator state into planner agent
    if coordinator_state:
        logger.info(f"{Colors.BLUE}===== Loading coordinator state into planner ====={Colors.END}")
        
        # Copy each state value to planner agent
        for key, value in coordinator_state.items():
            strands_utils.update_agent_state(agent, key, value)
            logger.debug(f"  - Loaded {key}: {type(value)}")
        
        # Access the loaded state
        user_request = strands_utils.get_agent_state(agent, "user_request", message)
        coordinator_response = strands_utils.get_agent_state(agent, "coordinator_response", "")
        history = strands_utils.get_agent_state(agent, "history", [])
        messages = strands_utils.get_agent_state(agent, "messages", [])
        
        logger.info(f"{Colors.BLUE}Loaded coordinator state:{Colors.END}")
        logger.info(f"  - user_request: {user_request}")
        logger.info(f"  - coordinator_response: {coordinator_response}")
        logger.info(f"  - history count: {len(history)}")
        logger.info(f"  - messages count: {len(messages)}")
    else:
        logger.warning(f"{Colors.RED}No coordinator state loaded{Colors.END}")
        
    agent, response = await strands_utils.process_streaming_response(agent, message)
    
    ## Planner logic: create detailed plan with agent assignments ##
    logger.info(f"{Colors.GREEN}===== Planner analyzing and creating execution plan ====={Colors.END}")
    
    logger.debug(f"\n{Colors.RED}Planner input message:\n{pprint.pformat(message, indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Planner response:\n{pprint.pformat(response["text"], indent=2, width=100)}{Colors.END}")

    # Planner determines next agent based on the plan
    goto = "__end__"
    if "supervisor" in response["text"].lower(): goto = "supervisor"
    elif "coder" in response["text"].lower(): goto = "coder"
    elif "reporter" in response["text"].lower(): goto = "reporter"
    
    strands_utils.update_agent_state(agent, "messages", agent.messages)
    strands_utils.update_agent_state(agent, "goto", goto)
    strands_utils.update_agent_state(agent, "full_plan", response["text"])  # store the generated plan
    history = strands_utils.get_agent_state(agent, "history", [])
    history.append({"agent":"planner", "message": response["text"]})
    strands_utils.update_agent_state(agent, "history", history)

    logger.debug(f"\n{Colors.RED}Planner state:\n{pprint.pformat(strands_utils.get_agent_state_all(agent), indent=2, width=100)}{Colors.END}")
    logger.info(f"{Colors.GREEN}===== Planner completed plan generation ====={Colors.END}")
    
    return agent, response