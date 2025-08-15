import pprint
import logging


import asyncio
from strands.models import BedrockModel
from src.utils.bedrock import bedrock_info
from src.utils.strands_sdk_utils import strands_utils
from src.prompts.template import apply_prompt_template

from strands.agent.agent_result import AgentResult
from strands.types.content import ContentBlock, Message
from strands.multiagent.base import MultiAgentBase, NodeResult, Status, MultiAgentResult

from strands.multiagent import GraphBuilder


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
        # Execute function
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
    
    agent = strands_utils.update_agent_state(agent, "goto", goto)
    history = strands_utils.get_agent_state(agent, "history", [])
    history.append({"agent":"coordinator", "message": response["text"]})
    agent = strands_utils.update_agent_state(agent, "history", history)
    
    logger.info(f"{Colors.GREEN}===== Coordinator completed task ====={Colors.END}")
    
    return agent, response
    
# def coordinator_node(state: State) -> Command[Literal["planner", "__end__"]]:
#     """Coordinator node that communicate with customers."""
#     logger.info(f"{Colors.GREEN}===== Coordinator talking...... ====={Colors.END}")

#     agent = strands_utils.get_agent(
#         agent_name="coordinator",
#         state=state,
#         streaming=True
#     )

#     message = state["request_prompt"]
#     agent, response = asyncio.run(strands_utils.process_streaming_response(agent, message, agent_name="coordinator"))
    
#     state["messages"] = agent.messages
#     logger.debug(f"\n{Colors.RED}Current state messages:\n{pprint.pformat(state['messages'][:-1], indent=2, width=100)}{Colors.END}")
#     logger.debug(f"\n{Colors.RED}Coordinator response:\n{pprint.pformat(response["text"], indent=2, width=100)}{Colors.END}")

#     goto = "__end__"
#     if "handoff_to_planner" in response["text"]: goto = "planner"

#     history = state.get("history", [])
#     history.append({"agent":"coordinator", "message": response["text"]})

#     logger.info(f"{Colors.GREEN}===== Coordinator completed task ====={Colors.END}")

#     return Command(
#         update={"history": history},
#         goto=goto,
#     )