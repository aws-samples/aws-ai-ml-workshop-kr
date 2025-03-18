from textwrap import dedent
from utils.bedrock import bedrock_utils, bedrock_chain

class llm_call():

    def __init__(self, **kwargs):

        self.llm=kwargs["llm"]
        self.verbose = kwargs.get("verbose", False)
        self.chain = bedrock_chain(bedrock_utils.converse_api) | bedrock_chain(bedrock_utils.outputparser)

        self.origin_max_tokens = self.llm.inference_config["maxTokens"]
        self.origin_temperature = self.llm.inference_config["temperature"]

    def _message_format(self, role, message):

        if role == "user":
             message_format = {
                "role": "user",
                "content": [{"text": dedent(message)}]
            }
        elif role == "assistant":
            
            message_format = {
                "role": "assistant",
                'content': [{'text': dedent(message)}]
            }

        return message_format
            
    def invoke(self, **kwargs):

        system_prompts = kwargs.get("system_prompts", None)
        messages = kwargs["messages"]
        enable_reasoning = kwargs.get("enable_reasoning", False)
        reasoning_budget_tokens = kwargs.get("reasoning_budget_tokens", 1024)
       
        if enable_reasoning:
            reasoning_config = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": reasoning_budget_tokens
                }
            }

            # Ensure maxTokens is greater than reasoning_budget
            if self.llm.inference_config["maxTokens"] <= reasoning_budget_tokens:
                # Make it just one token more than the reasoning budget
                adjusted_max_tokens = reasoning_budget_tokens + 1
                print(f'Info: Extended Thinking enabled increasing maxTokens from {self.llm.inference_config["maxTokens"]} to {adjusted_max_tokens} to exceed reasoning budget')
                self.llm.inference_config["maxTokens"] = adjusted_max_tokens

            self.llm.additional_model_request_fields = reasoning_config
            self.llm.inference_config["temperature"] = 1.0

        #print ("self.llm.additional_model_request_fields", self.llm.additional_model_request_fields)
        #print ("self.llm.inference_config", self.llm.inference_config)
           
        response = self.chain( ## pipeline의 제일 처음 func의 argument를 입력으로 한다. 여기서는 converse_api의 arg를 쓴다.
            llm=self.llm,
            system_prompts=system_prompts,
            messages=messages,
            verbose=self.verbose
        )
        
        ai_message = self._message_format(role="assistant", message=response["text"])
        #messages.append(ai_message)

        # Reset
        if enable_reasoning:
            self.llm.additional_model_request_fields = None
            self.llm.inference_config["maxTokens"] = self.origin_max_tokens
            self.llm.inference_config["temperature"] = self.origin_temperature
            
        return response, ai_message