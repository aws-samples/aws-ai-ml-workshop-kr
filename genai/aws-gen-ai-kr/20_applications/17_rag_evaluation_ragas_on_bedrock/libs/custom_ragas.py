import json
from datasets import Dataset
import numpy as np
import boto3
from botocore.config import Config
import re

def evaluate(dataset, metrics, llm_id, emb_id, region, verbose=False):
    """
    Evaluate the dataset using the specified metrics.

    Args:
    dataset (List[Dict]): List of dictionaries containing 'user_input', 'response', and 'retrieved_contexts'.
    metrics (List[Type]): List of metric classes to use for evaluation.
    llm_id (str): ID of the LLM model to use.
    emb_id (str): ID of the embeddings model to use.
    region (str): AWS region to use for Bedrock.

    Returns:
    Dict: A dictionary containing the scores for each metric.
    """
    average_scores = {}
    detailed_results = []


    for i, row in enumerate(dataset):
        if verbose:
            print("## row\n", row)

        row_result = {'row': i+1}
        for metric_class in metrics:
            if issubclass(metric_class, AnswerRelevancy):
                metric = metric_class(llm_id=llm_id, emb_id=emb_id, region=region)
            elif issubclass(metric_class, (Faithfulness, ContextRecall, ContextPrecision)):
                metric = metric_class(llm_id=llm_id, region=region)
            else:
                raise ValueError(f"Unsupported metric class: {metric_class.__name__}")

            try:
                # score = metric.score(row)
                score = metric.score(row, verbose=verbose)
                print(f"{metric_class.__name__} - Row {i+1}: Score = {score}")
                row_result[metric_class.__name__] = score

                if metric_class.__name__ not in average_scores:
                    average_scores[metric_class.__name__] = []
                average_scores[metric_class.__name__].append(score)

            except Exception as e:
                print(f"Error processing row {i+1} for {metric_class.__name__}: {e}")
                row_result[metric_class.__name__] = None

        detailed_results.append(row_result)

    for metric, scores in average_scores.items():
        if scores:
            average_scores[metric] = sum(scores) / len(scores)
        else:
            average_scores[metric] = "No valid scores"

    return {
        'average_scores': average_scores,
        'detailed_results': detailed_results
    }

class AnswerRelevancy:
    def __init__(self, llm_id, emb_id, region, strictness=3):
        self.llm_id = llm_id
        self.emb_id = emb_id
        self.region = region
        self.strictness = strictness
        retry_config = Config(
            region_name=region,
            retries={"max_attempts": 10, "mode": "standard"}
        )
        self.boto3_client = boto3.client("bedrock-runtime", config=retry_config)
        self.tool_config = self._init_tool()

    def _init_tool(self):
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "QuestionGenerator",
                        "description": "Generates questions based on the given context and answer.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "question": {
                                        "type": "string",
                                        "description": "The generated question"
                                    },
                                    "noncommittal": {
                                        "type": "string",
                                        "description": "Give 'noncommittal' as 1 if the answer is noncommittal and 0 if the answer is committal. A noncommittal answer is one that is evasive, vague, or ambiguous. For example, 'I don't know' or 'I'm not sure' are noncommittal answers."
                                    }
                                },
                                "required": ["question", "answer"]
                            }
                        }
                    }
                }
            ]
        }
        return tool_config

    def _create_message_format(self, sys_template, user_template):
        sys_prompt = [{"text": sys_template}]
        usr_prompt = [{"role": "user", "content": [{"text": user_template}]}]
        return sys_prompt, usr_prompt

    def _converse_with_bedrock_tools(self, sys_prompt, usr_prompt):
        inference_config = {"temperature": 0.0, "topP": 0.1}
        response = self.boto3_client.converse(
            modelId=self.llm_id,
            messages=usr_prompt,
            system=sys_prompt,
            toolConfig=self.tool_config,
            inferenceConfig=inference_config
        )
        return response

    def _parse_tool_use(self, message):
        stop_reason = message['stopReason']

        if stop_reason == 'tool_use':
            tool_requests = message['output']['message']['content']
            for tool_request in tool_requests:
                if 'toolUse' in tool_request:
                    tool = tool_request['toolUse']

                    if tool['name'] == 'QuestionGenerator':
                        return tool['input']
        return None

    def _generate_questions(self, answer, context):
        sys_template = """
        Generate a question for the given answer based on the given context and identify if the answer is noncommittal. 
        """

        user_template = f"""
        Answer: {answer}
        Context: {context}
        Use 'QuestionGenerator' tool to generate a question.
        """

        questions = []
        noncommittals = []

        for _ in range(self.strictness):
            sys_prompt, user_prompt = self._create_message_format(sys_template, user_template)
            response = self._converse_with_bedrock_tools(sys_prompt, user_prompt)
            output = self._parse_tool_use(response)          
            
            question = output['question']
            noncommittal = int(output['noncommittal'])

            questions.append(question)
            noncommittals.append(noncommittal)
        
        return questions, noncommittals        

    def _get_embedding_vector(self, text):
        request = json.dumps({"inputText": text})
        response = self.boto3_client.invoke_model(modelId=self.emb_id, body=request)
        embedding = json.loads(response["body"].read())["embedding"]
        return embedding

    def score(self, row, verbose=False):
        user_input = row['user_input']
        answer = row['response']
        context = row['retrieved_contexts']
        context_str = '\n'.join(context)

        generated_questions, noncommittals = self._generate_questions(answer, context_str)

        user_input_vec = self._get_embedding_vector(user_input)
        generated_vectors = [self._get_embedding_vector(q) for q in generated_questions]
        similarities = [
            self._cosine_similarity(user_input_vec, vec) for vec in generated_vectors
        ]
        avg_similarity = np.mean(similarities)
        is_committal = all(not nc for nc in noncommittals)

        return avg_similarity * (1 if is_committal else 0)

    def _cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


class Faithfulness:
    def __init__(self, llm_id, region):
        self.llm_id = llm_id
        self.region = region
        retry_config = Config(
            region_name=region,
            retries={"max_attempts": 10, "mode": "standard"}
        )
        self.boto3_client = boto3.client("bedrock-runtime", config=retry_config)
        self.tool_config = self._init_tool()

    def _init_tool(self):
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "FaithfulnessChecker",
                        "description": "Checks the faithfulness of paragraphs based on a given context.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "verdicts": {
                                        "type": "array",
                                        "items": {
                                            "type": "integer",
                                            "enum": [0, 1]
                                        },
                                        "description": "Array of 0 (not faithful) or 1 (faithful) for each paragraph"
                                    }
                                },
                                "required": ["verdicts"]
                            }
                        }
                    }
                }
            ]
        }
        return tool_config

    def _create_message_format(self, sys_template, user_template):
        sys_prompt = [{"text": sys_template}]
        usr_prompt = [{"role": "user", "content": [{"text": user_template}]}]
        return sys_prompt, usr_prompt

    def _converse_with_bedrock_tools(self, sys_prompt, usr_prompt):
        inference_config = {"temperature": 0.0, "topP": 0.1}
        response = self.boto3_client.converse(
            modelId=self.llm_id,
            messages=usr_prompt,
            system=sys_prompt,
            toolConfig=self.tool_config,
            inferenceConfig=inference_config
        )
        return response

    def _parse_tool_use(self, message):
        stop_reason = message['stopReason']
        if stop_reason == 'tool_use':
            tool_requests = message['output']['message']['content']
            results = []
            for tool_request in tool_requests:
                if 'toolUse' in tool_request:
                    tool = tool_request['toolUse']
                    results.append(tool['input'])
            return results
        return None

    def _segment_paragraphs(self, text):
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs

    def _check_faithfulness(self, context, user_input, verbose=False):
        '''
        faithfulness 의 알고리즘은 아래의 1번, 2번을 Tool "FaithfulnessChecker" 에게 제공해서,  1 번과 2번이 Faithfull 하면 1, 그렇지 않으면 0 을 제공함.
        1. LLM 으로 부터의 답변을 paragraphs 으로 분리한다.
        2. 제공된 Context
        '''
        sys_template = """
        Your task is to judge the faithfulness of a series of paragraphs based on a given context. For each paragraph, determine if it can be directly inferred from the context..
        """
        paragraphs = self._segment_paragraphs(user_input) # generated_answer
        paragraphs_str = '\n\n'.join([f"Paragraph {i}:\n {p}" for i, p in enumerate(paragraphs)])

        user_template = f"""
        Context: {context}

        Paragraphs:
        {paragraphs_str}

        Use the FaithfulnessChecker tool to evaluate the given paragraphs.
        """
        sys_prompt, user_prompt = self._create_message_format(sys_template, user_template)
        response = self._converse_with_bedrock_tools(sys_prompt, user_prompt)
        output = self._parse_tool_use(response)

        if verbose:
            print("## _check_faithfulness")
            print("# sys_prompt: \n", sys_prompt)
            print("# user_prompt: \n", user_prompt)
            print("# total response from LLM: \n", response)
            print("# tool's output with FaithfulnessChecker: \n", output)

        if output and len(output) > 0:
            return output[0]['verdicts']
        return []

    def score(self, row, verbose=False):
        context = row['retrieved_contexts'] # retrieved_contexts
        user_input = row['response'] # generated_answert
        verdicts = self._check_faithfulness(context, user_input, verbose=False)
        # verdicts = self._check_faithfulness(context, user_input, verbose=True)

        if verbose:
            print("## retrieved_contexts: \n", context)
            print("## generated_answer: \n", user_input)            
            print("## verdicts: \n", verdicts)

        if not verdicts:
            return 0.0

        faithful_paragraphs = sum(verdicts)
        total_paragraphs = len(verdicts)
        score = faithful_paragraphs / total_paragraphs

        return score


class ContextRecall:
    def __init__(self, llm_id, region):
        self.llm_id = llm_id
        self.region = region
        retry_config = Config(
            region_name=region,
            retries={"max_attempts": 10, "mode": "standard"}
        )
        self.boto3_client = boto3.client("bedrock-runtime", config=retry_config)
        self.tool_config = self._init_tool()

    def _init_tool(self):
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "ContextRecallClassifier",
                        "description": "Classifies if a statement can be attributed to the given context.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "attributed": {
                                        "type": "array",
                                        "items": {
                                            "type": "integer",
                                            "enum": [0, 1]
                                        },
                                        "description": "Array of 0 (not attributed) or 1 (attributed) for each statement"
                                    }
                                },
                                "required": ["attributed"]
                            }
                        }
                    }
                }
            ]
        }
        return tool_config

    def _create_message_format(self, sys_template, user_template):
        sys_prompt = [{"text": sys_template}]
        usr_prompt = [{"role": "user", "content": [{"text": user_template}]}]
        return sys_prompt, usr_prompt

    def _converse_with_bedrock_tools(self, sys_prompt, usr_prompt):
        inference_config = {"temperature": 0.0, "topP": 0.1}
        response = self.boto3_client.converse(
            modelId=self.llm_id,
            messages=usr_prompt,
            system=sys_prompt,
            toolConfig=self.tool_config,
            inferenceConfig=inference_config
        )
        return response

    def _parse_tool_use(self, message):
        stop_reason = message['stopReason']
        if stop_reason == 'tool_use':
            tool_requests = message['output']['message']['content']
            results = []
            for tool_request in tool_requests:
                if 'toolUse' in tool_request:
                    tool = tool_request['toolUse']
                    results.append(tool['input'])
            return results
        return None

    def _segment_paragraphs(self, text):
        paragraphs = re.split(r'\n{2,}|\n', text.strip())
        return [p.strip() for p in paragraphs if p.strip()]

    def _check_context_recall(self, user_input, contexts, reference):
        sys_template = """
        Given multiple contexts and a reference answer, analyze each statement in the reference and classify if it can be attributed to any of the given contexts.
        """
        paragraphs = self._segment_paragraphs(reference)
        paragraphs_str = '\n\n'.join([f"Paragraph {i+1}: {p}" for i, p in enumerate(paragraphs)])
        contexts_str = '\n'.join([f"Context {i+1}: {c}" for i, c in enumerate(contexts)])
        user_template = f"""
        Question:
        {user_input}

        Contexts:
        {contexts_str}

        Reference paragraphs:
        {paragraphs_str}

        Use the ContextRecallClassifier tool to evaluate each paragraph in the reference.
        """
        sys_prompt, user_prompt = self._create_message_format(sys_template, user_template)
        response = self._converse_with_bedrock_tools(sys_prompt, user_prompt)
        output = self._parse_tool_use(response)

        if output and len(output) > 0:
            return output[0]['attributed']
        return []

    def score(self, row, verbose=False):
        user_input = row['user_input']
        contexts = row['retrieved_contexts']
        reference = row['reference']

        attributed = self._check_context_recall(user_input, contexts, reference)
        if not attributed:
            return 0.0

        total_paragraphs = len(attributed)
        attributed_paragraphs = sum(attributed)
        score = attributed_paragraphs / total_paragraphs if total_paragraphs > 0 else 0.0

        return score


class ContextPrecision:
    def __init__(self, llm_id: str, region: str):
        self.llm_id = llm_id
        self.region = region

        retry_config = Config(
            region_name=self.region,
            retries={"max_attempts": 10, "mode": "standard"}
        )
        self.boto3_client = boto3.client("bedrock-runtime", config=retry_config)
        self.tool_config = self._init_tool()

    def _init_tool(self):
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "ContextPrecisionClassifier",
                        "description": "Classifies if the given context was useful in arriving at the given answer.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "verdict": {
                                        "type": "integer",
                                        "enum": [0, 1],
                                        "description": "0 if not useful, 1 if useful"
                                    }
                                },
                                "required": ["verdict"]
                            }
                        }
                    }
                }
            ]
        }
        return tool_config

    def _create_message_format(self, sys_template, user_template):
        sys_prompt = [{"text": sys_template}]
        usr_prompt = [{"role": "user", "content": [{"text": user_template}]}]
        return sys_prompt, usr_prompt

    def _converse_with_bedrock_tools(self, sys_prompt, usr_prompt):
        inference_config = {"temperature": 0.0, "topP": 0.1}
        response = self.boto3_client.converse(
            modelId=self.llm_id,
            messages=usr_prompt,
            system=sys_prompt,
            toolConfig=self.tool_config,
            inferenceConfig=inference_config
        )
        return response

    def _parse_tool_use(self, message):
        stop_reason = message['stopReason']
        if stop_reason == 'tool_use':
            tool_requests = message['output']['message']['content']
            results = []
            for tool_request in tool_requests:
                if 'toolUse' in tool_request:
                    tool = tool_request['toolUse']
                    results.append(tool['input'])
            return results
        return None

    def _check_context_precision(self, question, context, answer):
        sys_template = """
        Given a question, answer, and context, verify if the context was useful in arriving at the given answer.
        """
        user_template = f"""
        Question: {question}
        Context: {context}
        Answer: {answer}

        Use the ContextPrecisionClassifier tool to evaluate if the context was useful for the answer.
        """
        sys_prompt, user_prompt = self._create_message_format(sys_template, user_template)
        response = self._converse_with_bedrock_tools(sys_prompt, user_prompt)
        output = self._parse_tool_use(response)

        if output and len(output) > 0:
            return output[0]['verdict']
        return 0

    def _calculate_average_precision(self, verifications):
        # verdict_list contains the usefulness judgement for each context (1 or 0)
        verdict_list = verifications
        # denominator is the total number of useful contexts
        # (a small value is added to prevent division by zero)
        denominator = sum(verdict_list) + 1e-10
        # numerator is the sum of precision at each position multiplied by its usefulness
        numerator = sum(
            [
                (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
                for i in range(len(verdict_list))
            ]
        )
        # The final score is calculated as numerator / denominator
        score = numerator / denominator

        if np.isnan(score):
            logger.warning(
                "Invalid response format. Expected a list of dictionaries with keys 'verdict'"
            )
        return score

    def score(self, row, verbose=False):
        question = row['user_input']
        contexts = row['retrieved_contexts']
        answer = row['reference']

        verifications = []
        for context in contexts:
            verdict = self._check_context_precision(question, context, answer)
            verifications.append(verdict)
        #print(verifications)

        score = self._calculate_average_precision(verifications)
        return score