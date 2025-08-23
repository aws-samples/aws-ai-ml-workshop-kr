import asyncio
from strands.models import BedrockModel
from src.utils.bedrock import bedrock_info
from src.utils.strands_sdk_utils import strands_utils
from src.prompts.template import apply_prompt_template


from strands.agent.agent_result import AgentResult
from strands.types.content import ContentBlock, Message
from strands.multiagent.base import MultiAgentBase, NodeResult, Status, MultiAgentResult

from strands.multiagent import GraphBuilder


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
            result = await self.func(task if isinstance(task, str) else str(task))
        else:
            result = self.func(task if isinstance(task, str) else str(task))

        agent_result = AgentResult(
            stop_reason="end_turn",
            message=Message(role="assistant", content=[ContentBlock(text=str(result))]),
            metrics={},
            state={}
        )

        # Return wrapped in MultiAgentResult
        return MultiAgentResult(
            status=Status.COMPLETED,
            results={self.name: NodeResult(result=agent_result)},
            # ... execution details
        )
    
async def coordinator_node(message: str):

    agent = strands_utils.get_agent(
        agent_name="coordinator",
        system_prompts=apply_prompt_template(prompt_name="coordinator", prompt_context={}), # apply_prompt_template(prompt_name="task_agent", prompt_context={"TEST": "sdsd"})
        agent_type="basic", #"reasoning", "basic"
        prompt_cache_info=(False, None), #(False, None), (True, "default")
        streaming=True,
    )
        
    agent, response = await strands_utils.process_streaming_response(agent, message)


    ## your logic here ##
    print ("여기서 무언가를 해도 되겠죠?")

    return agent, response

builder = GraphBuilder()
# Usage
coordinator = FunctionNode(func=coordinator_node, name="coordinator")
builder.add_node(coordinator, "coordinator")

# Set entry points (optional - will be auto-detected if not specified)
builder.set_entry_point("coordinator")

# Build the graph
graph = builder.build()

res = graph("안녕 나는 장동진이야")






