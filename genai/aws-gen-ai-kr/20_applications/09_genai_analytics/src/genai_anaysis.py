from textwrap import dedent
from utils.bedrock import bedrock_utils, bedrock_chain

class llm_call():

    def __init__(self, **kwargs):

        self.llm_sonnet=kwargs["llm_sonnet"]
        self.llm_haiku=kwargs["llm_haiku"]
        self.verbose = kwargs.get("verbose", False)
        self.chain = bedrock_chain(bedrock_utils.converse_api) | bedrock_chain(bedrock_utils.outputparser)


    def _message_format(self, role, message):

        if role == "user":
             message_format = {
                "role": "user",
                "content": [{"text": dedent(message)}]
            }
        elif role == "assistant":
            
            message_format = {
                "role": "assistant",
                'content': [{'text': message}]
            }

        return message_format
            
    def invoke(self, **kwargs):

        system_prompts = kwargs.get("system_prompts", None)
        messages = kwargs["messages"]
        llm_name = kwargs["llm_name"]

        if llm_name == "sonnet": llm = self.llm_sonnet
        else: llm = self.llm_haiku
    
        response = self.chain( ## pipeline의 제일 처음 func의 argument를 입력으로 한다. 여기서는 converse_api의 arg를 쓴다.
            llm=llm,
            system_prompts=system_prompts,
            messages=messages,
            verbose=self.verbose
        )
        
        ai_message = self._message_format(role="assistant", message=response["text"])
        messages.append(ai_message)
        return response, messages

