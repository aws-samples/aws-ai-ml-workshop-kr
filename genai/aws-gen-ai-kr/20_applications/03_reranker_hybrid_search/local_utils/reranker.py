import json
import numpy as np

class RunReranker:
    '''
    Given query and query pair, provide the highest similarity score
    '''
    def __init__(self, query, query_pair, boto3_client, endpoint_name, verbose):
        self.query = query
        self.query_pair = query_pair
        self.endpoint_name = endpoint_name
        self.boto3_client = boto3_client
        self.highest_intent = None
        self.verbose = verbose
        # Start function()
        self.get_the_highest_intent()

    def get_the_highest_intent(self):
        # print("########## get_the_highest_intent ################")
        payload = self.create_reranker_payload()
        response_reranker = self.run_payload(payload)
        highest_intent = self.get_highest_intent(response_reranker)    

        self.highest_intent = highest_intent

    def show_ranked_answers(self):
        print("#############################################")        
        print("Query: ", self.query)
        print("#############################################")        
        print("Re-Ranked Answers: ")
        payload = self.create_reranker_payload()
        response_reranker = self.run_payload(payload)
        self._sort_answer(response_reranker, verbose=False)

    def _sort_answer(self, response_reranker, verbose):
        score_list = []
        for label in response_reranker:
            score_list.append(label['score'])
        
        # Convert the list to a NumPy array
        sorted_score_list = sorted(score_list, reverse=True)
        np_numbers = np.array(score_list)
        sorted_indices = [index for index, value in sorted(enumerate(np_numbers), key=lambda x: x[1], reverse=True)]

        print("Product, Desc, Category, Similar Score:")
        for idx, score in zip(sorted_indices, sorted_score_list):
            query_pair = self.query_pair[idx]
            print(
                query_pair.metadata['product'],", ",
                query_pair.metadata['desc'],", ",
                query_pair.metadata['intent'],", ",
                round(score,6))            

        if verbose:
            print("### Reranker scorelist: \n", score_list)                    
            print("### Reranker sorted scorelist: \n", sorted_score_list)                                
            print("### Reranker scorelist sorted_indices: \n", sorted_indices)                                


        

    def create_reranker_payload(self):
        input_list = []
        for doc in self.query_pair:
            # print("doc: " , doc)
            # print("type of doc: ", type(doc))
            page_content = doc.page_content
            intent = doc.metadata['intent']
            single_input = {"text": self.query, "text_pair": page_content}
            input_list.append(single_input)
            
        payload = json.dumps(
            {
                "inputs": input_list
            },
             ensure_ascii=False,
             indent=4
        )
        if self.verbose:
            print("### Query: \n",self.query )
            print("### Reranker payload: \n", payload)

        return payload

    def run_payload(self, payload):
        response = self.boto3_client.invoke_endpoint(
            EndpointName= self.endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body= payload
        )

        out = json.loads(response['Body'].read().decode()) ## for json                        
        if self.verbose:
            print("### Reranker output: \n", out)        

        return out

    def get_highest_intent(self, response_reranker):
        score_list = []
        for label in response_reranker:
            score_list.append(label['score'])
        
        # Convert the list to a NumPy array
        np_numbers = np.array(score_list)

        # Use np.argmax() to find the index of the maximum value
        max_index = np.argmax(np_numbers)

        highest_query_pair = self.query_pair[max_index]
        highest_query = highest_query_pair.page_content
        highest_intent = highest_query_pair.metadata['intent']


        if self.verbose:
            print("### Reranker scorelist: \n", score_list)                    
            print(f"### The index of the maximum number is: {max_index}")            
            print("### Highest query: \n", highest_query)                                
            print("### Highest intent: \n", highest_intent)                                            

        return highest_intent
        
        
        

    def show_input(self):
        print(self.query)
        print(self.query_pair)