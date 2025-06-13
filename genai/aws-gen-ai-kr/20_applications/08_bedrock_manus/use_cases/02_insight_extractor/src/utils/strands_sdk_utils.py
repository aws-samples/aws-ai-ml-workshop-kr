class strands_utils():

    @staticmethod
    def parsing_text_from_response(response):

        output = {}
        if len(response.message["content"]) == 2: ## reasoning
            output["reasoning"] = response.message["content"][0]["reasoningContent"]["reasoningText"]["text"]
            output["signature"] = response.message["content"][0]["reasoningContent"]["reasoningText"]["signature"]
        
        output["text"] = response.message["content"][-1]["text"]
    
        return output  