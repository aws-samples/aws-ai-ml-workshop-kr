import pprint
import logging
import asyncio
from strands import Agent
from src.utils.strands_sdk_utils import strands_utils
from src.prompts.template import apply_prompt_template
from src.utils.common_utils import get_message_from_string


# Initialize logger
logger = logging.getLogger(__name__)
logger.propagate = False  # 상위 로거로 메시지 전파 중지
for handler in logger.handlers[:]: 
    logger.removeHandler(handler)
handler = logging.StreamHandler()
formatter = logging.Formatter('\n%(levelname)s [%(name)s] %(message)s')  # 로그 레벨이 동적으로 표시되도록 변경
handler.setFormatter(formatter)
logger.addHandler(handler)
# DEBUG와 INFO 중 원하는 레벨로 설정
logger.setLevel(logging.INFO)  # DEBUG 이상 모든 로그 표시

TOOL_SPEC = {
    "name": "planner_tool",
    "description": "Professional Deep Researcher tool that orchestrates a team of agents [Coder, Reporter] to complete complex requirements. Creates detailed plans with agent assignments, task tracking, and execution monitoring.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "user_request": {
                    "type": "string",
                    "description": "The user's requirement or question that needs to be analyzed and planned for execution by the agent team."
                },
                "analysis_materials": {
                    "type": "string",
                    "description": "Optional. Information about analysis materials including name, location, format, or other relevant details if mentioned in the user's request."
                },
                "existing_plan": {
                    "type": "string",
                    "description": "Optional. Previously created plan in full_plan format for task tracking and progress updates. If provided, the tool will perform task tracking instead of creating a new plan."
                },
                "task_response": {
                    "type": "string",
                    "description": "Optional. Response from an agent that completed a task, used to update the task completion status in the checklist."
                }
            },
            "required": ["user_request"]
        }
    }
}

#RESPONSE_FORMAT = "Response from {}:\n\n<response>\n{}\n</response>\n\n*Please execute the next step.*"
FULL_PLAN_FORMAT = "Here is full plan :\n\n<full_plan>\n{}\n</full_plan>\n\n*Please consider this to select the next step.*"
#CLUES_FORMAT = "Here is clues from {}:\n\n<clues>\n{}\n</clues>\n\n"

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def planner_node(owner_agent: Agent, message: str):
    """Planner node that generate the full plan."""
    logger.info(f"{Colors.GREEN}===== Planner generating full plan ====={Colors.END}")
    logger.debug(f"\n{Colors.RED}Planner - current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")
    
    
    agent = strands_utils.get_agent(
        agent_name="planner",
        system_prompts=apply_prompt_template(prompt_name="planner", prompt_context={}), # apply_prompt_template(prompt_name="task_agent", prompt_context={"TEST": "sdsd"})
        agent_type="reasoning", #"reasoning", "basic"
        prompt_cache_info=(True, "default"), #(False, None), (True, "default")
        streaming=True,
    )
        
    #agent, response = await strands_utils.process_streaming_response(agent, message)
    #full_plan, messages = state.get("full_plan", ""), state["messages"]


    full_plan = strands_utils.get_agent_state(owner_agent, "full_plan", "")
    messages = strands_utils.get_agent_state(owner_agent, "messages")
    message = '\n\n'.join([messages[-1]["content"][-1]["text"], FULL_PLAN_FORMAT.format(full_plan)])

    agent, response = asyncio.run(strands_utils.process_streaming_response(agent, message))
    logger.debug(f"\n{Colors.RED}Planner response:\n{pprint.pformat(response["text"], indent=2, width=100)}{Colors.END}")

    #full_plan, messages = state.get("full_plan", ""), state["messages"]
    #message = '\n\n'.join([messages[-1]["content"][-1]["text"], FULL_PLAN_FORMAT.format(full_plan)])
    #agent, response = asyncio.run(strands_utils.process_streaming_response(agent, message, agent_name="planner"))
    #logger.debug(f"\n{Colors.RED}Planner response:\n{pprint.pformat(response["text"], indent=2, width=100)}{Colors.END}")

    #goto = "supervisor"
    
    #agent = strands_utils.update_agent_state(agent, "messages", agent.messages)
    #agent = strands_utils.update_agent_state(agent, "goto", goto)
    history = strands_utils.get_agent_state(owner_agent, "history", [])
    history.append({"agent":"planner", "message": response["text"]})
    owner_agent = strands_utils.update_agent_state(owner_agent, "history", history)
    owner_agent = strands_utils.update_agent_state(owner_agent, "messages", [get_message_from_string(role="user", string=response["text"], imgs=[])])
    owner_agent = strands_utils.update_agent_state(owner_agent, "full_plan", response["text"])


    #history = state.get("history", [])
    #history.append({"agent":"planner", "message": response["text"]})
    logger.info(f"{Colors.GREEN}===== Planner completed task ====={Colors.END}")
    return f'Full plan: "{response["text"]}"'