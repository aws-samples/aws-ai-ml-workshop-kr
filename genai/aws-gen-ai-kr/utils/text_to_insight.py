from textwrap import dedent
from utils.bedrock import bedrock_utils, bedrock_chain

class insight_extraction_chain():
    
    def __init__(self, **kwargs):

        self.llm = kwargs["llm"]
        self.system_prompts = kwargs["system_prompts"]
        self.user_prompts = kwargs["user_prompts"]
        self.messages = []
        self.multi_turn = kwargs.get("multi_turn", False)
        self.verbose = kwargs.get("verbose", False)
        self.tool_config = kwargs.get("tool_config", False)
        
        self.request_prompt = dedent(
            '''
            메시지 작성 시 요청사항 입니다.:
            <request>{request}</request>
            '''
        )
    
    def _get_analysis_points(self, **kwargs):
        
        get_analysis_point_pipeline = bedrock_chain(bedrock_utils.converse_api) | bedrock_chain(bedrock_utils.outputparser)
        
        context = kwargs["context"]
        
        if "request" not in context:
            query = self.user_prompts.format(**context)
        elif len(context) == 1 and "request" in context:
            query = self.request_prompt.format(**context)
        else:
            prompt = self.user_prompts + self.request_prompt
            query = prompt.format(**context)
        
        user_message = {
            "role": "user",
            "content": [{"text": dedent(query)}]
        }
        self.messages.append(user_message)
        
        response = get_analysis_point_pipeline( ## pipeline의 제일 처음 func의 argument를 입력으로 한다. 여기서는 converse_api의 arg를 쓴다.
            llm=self.llm,
            system_prompts=self.system_prompts,
            messages=self.messages,
            verbose=self.verbose
        )
        
        if self.verbose:
            print (f'Messages: {self.messages}')

        if self.multi_turn:
            if response["toolUse"] == None:
                ai_message = {
                    "role": "assistant",
                    'content': [{'text': response["text"]}]
                }
            else:
                ai_message = {
                    "role": "assistant",
                    'content': [{'toolUse': response["toolUse"]}]
                }
            self.messages.append(ai_message)
        else:
            self.messages = []
            
        return response
            
            
    def _parse_output(self, **kwargs):
        
        print ("_parse_output")
        
        
        response = kwargs["response"]
        messages = []

            
        prompt = dedent(
            '''
            You are an AI assistant specialized in converting unstructured responses into structured results.
            '''
        )
        system_prompt_parse_output = system_prompts = bedrock_utils.get_system_prompt(system_prompts=prompt,)
    
        query = dedent(
            '''
            Here is responses: <response>{response}</response>
            Please use the parse_output tool to do the task of structuring information from the responses.
            '''
        )
        query = query.format(**{"response":response["text"]})
        
        user_message = {
            "role": "user",
            "content": [{"text": dedent(query)}]
        }
        
        
        messages.append(user_message)
        print ("messages", messages)
        
        response = bedrock_utils.converse_api(
            llm=self.llm,
            system_prompts=system_prompt_parse_output,
            messages=messages,
            tool_config=self.tool_config,
            verbose=self.verbose
        )
        
        print ("response", response)
        
        

        

        
        if self.verbose:
            print (f'Messages: {messages}')

        if self.multi_turn:
            if response["toolUse"] == None:
                ai_message = {
                    "role": "assistant",
                    'content': [{'text': response["text"]}]
                }
            else:
                ai_message = {
                    "role": "assistant",
                    'content': [{'toolUse': response["toolUse"]}]
                }
            messages.append(ai_message)
        else:
            messages = []
        
        return response
        

        
        
        
    def invoke(self, **kwargs):
        
        response = self._get_analysis_points(**kwargs)
        
        #_ = self._parse_output(response=response)

        return response

    
    
class insight_extraction_tools():
    
    @classmethod
    def get_tool_list(cls, ):
    
        tool_list = [
            {
                "toolSpec": {
                    "name": "parse_output",
                    "description": "Use this tool to to do the task of structuring information from output responses",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "Empty string"
                                }
                            }
                        }
                    }
                }
            }
        ]
        
        tool_config={
            "tools": tool_list,
            "toolChoice": {
                "tool": {
                    "name": "parse_output"
                }
            }
        }
        
        return tool_config