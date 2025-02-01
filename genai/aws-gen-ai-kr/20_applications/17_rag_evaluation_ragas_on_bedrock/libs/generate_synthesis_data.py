import pdfplumber
from typing import List, Dict, Any

class PDFLoader:
    def __init__(self, file_path: str, start_page: int = None, end_page: int = None):
        self.file_path = file_path
        self.start_page = start_page
        self.end_page = end_page

    def load(self) -> Dict[str, Any]:
        combined_text = ""
        metadata = {}

        with pdfplumber.open(self.file_path) as pdf:
            total_pages = len(pdf.pages)

            start = (self.start_page or 1) - 1
            end = min(self.end_page or total_pages, total_pages)

            for page_num in range(start, end):
                page = pdf.pages[page_num]
                text = page.extract_text()
                combined_text += text + "\n"

            metadata = {
                "source": self.file_path,
                "filename": self.file_path,
                "total_pages": total_pages,
                "extracted_pages": f"{start + 1}-{end}"
            }

            for key, value in pdf.metadata.items():
                if isinstance(value, (str, int)):
                    metadata[key] = value

        return {
            "page_content": combined_text.strip(),
            "metadata": metadata
        }

        import re

from typing import List, Optional
import re

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100, separators: Optional[List[str]] = None) -> List[str]:
    separators = separators or ["\n\n", "\n", " ", ""]

    def _split_text_recursive(text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]

        separator = separators[0]
        splits = re.split(f"({re.escape(separator)})", text)
        splits = ["".join(splits[i:i+2]) for i in range(0, len(splits), 2)]

        final_chunks = []
        current_chunk = ""

        for split in splits:
            if len(current_chunk) + len(split) <= chunk_size:
                current_chunk += split
            else:
                if current_chunk:
                    final_chunks.append(current_chunk)
                if len(split) > chunk_size:
                    subsplits = _split_text_recursive(split, separators[1:])
                    final_chunks.extend(subsplits)
                else:
                    current_chunk = split

        if current_chunk:
            final_chunks.append(current_chunk)

        return final_chunks

    chunks = _split_text_recursive(text, separators)

    if chunk_overlap > 0:
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                overlap_text = chunks[i-1][-chunk_overlap:]
                overlapped_chunks.append(overlap_text + chunk)
        chunks = overlapped_chunks

    return chunks

def add_meta_into_chunk(chunks, docs):
    chunks_with_metadata = []
    for i, chunk in enumerate(chunks):
        chunks_with_metadata.append({
            'content': chunk,
            'metadata': {
                'chunk_id': i,
                'filename': docs['metadata'].get('filename', 'unknown')
            }
        })
    return chunks_with_metadata    

def create_prompt(sys_template, user_template):
    sys_prompt = [{"text": sys_template}]
    usr_prompt = [{"role": "user", "content": [{"text": user_template}]}]
    return sys_prompt, usr_prompt

def get_context_chunks(chunks_with_metadata, start_id):
    context_chunks = [
        chunks_with_metadata[start_id]['content'],
        chunks_with_metadata[start_id + 1]['content'],
        chunks_with_metadata[start_id + 2]['content']
    ]
    return " ".join(context_chunks)

import random
import json
from time import sleep


def converse_with_bedrock_tools(sys_prompt, usr_prompt, tool_config, model_id, boto3_client):
    temperature = 0.0
    top_p = 0.1
    top_k = 1
    inference_config = {"temperature": temperature, "topP": top_p}
    additional_model_fields = {"top_k": top_k}
    response = boto3_client.converse(
        modelId= model_id,
        messages=usr_prompt,
        system=sys_prompt,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields,
        toolConfig=tool_config
    )
    return response

def parse_tool_use(message):
    stop_reason = message['stopReason']

    if stop_reason == 'tool_use':
        tool_requests = message['output']['message']['content']
        for tool_request in tool_requests:
            if 'toolUse' in tool_request:
                tool = tool_request['toolUse']

                if tool['name'] == 'QuestionAnswerGenerator':
                    return tool['input']
    return None


tool_config = {
    "tools": [
        {
            "toolSpec": {
                "name": "QuestionAnswerGenerator",
                "description": "Generates questions and answers based on the given context.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The generated question"
                            },
                            "answer": {
                                "type": "string",
                                "description": "The answer to the generated question"
                            }
                        },
                        "required": ["question", "answer"]
                    }
                }
            }
        }
    ]
}

def generate_qa_dataset(model_id, chunks, num_pairs, boto3_client, output_file, verbose=False):
    total_chunks = len(chunks)
    dataset = []

    for i in range(num_pairs):
        start_id = random.randint(0, total_chunks - 3)
        context = get_context_chunks(chunks, start_id)

        if verbose:
            print("## context: \n", context)


        if i % 2 == 0: # question_type = "complex"
            sys_template = """
            You are an expert at generating practical questions based on given documentation.
            Your task is to generate complex, reasoning questions and answers in korean.

            Follow these rules:
            1. Generate questions that reflect real user information needs related to the document's subject matter (e.g., technical docs : feature availability, implementation details)
            2. Ensure questions are relevant, concise, preferably under 25 words, and fully answerable with the provided information
            3. Focus on extracting key information that users are likely to seek, while avoiding narrow or less important questions.
            4. When provided with code blocks, focus on understanding the overall functionality rather than the specific syntax or variables. Feel free to request examples of how to use key APIs or features.
            5. Do not use phrases like 'based on the provided context' or 'according to the context'.
            """
            question_type = "complex"
        else: # question_type = "simple"
            sys_template = """
            You are an expert at generating practical questions based on given documentation.
            Your task is to create simple, directly answerable questions from the given context in korean.

            Follow these rules:
            1. Generate questions that reflect real user information needs related to the document's subject matter (e.g., technical docs : feature availability, implementation details)
            2. Ensure questions are relevant, concise, preferably under 10 words, and fully answerable with the provided information
            3. Focus on extracting key information that users are likely to seek, while avoiding narrow or less important questions.
            4. When provided with code blocks, focus on understanding the overall functionality rather than the specific syntax or variables. Feel free to request examples of how to use key APIs or features.
            5. Do not use phrases like 'based on the provided context' or 'according to the context'.
            """
            question_type = "simple"

        user_template = f"""
        Generate a {question_type} question and its answer based on the following context:

        Context: {context}

        Use the QuestionAnswerGenerator tool to provide the output.
        """

        sys_prompt, user_prompt = create_prompt(sys_template, user_template)

        if verbose:
            print("## sys_prompt: \n", sys_prompt)
            print("## user_prompt: \n", user_prompt)

        response = converse_with_bedrock_tools(sys_prompt, user_prompt, tool_config, model_id, boto3_client)

        if verbose:        
            print("## response from LLM: \n", response)

        qa_data = parse_tool_use(response)

        if qa_data:
            qa_item = {
                "question": qa_data["question"],
                "ground_truth": qa_data["answer"],
                "question_type": question_type,
                "contexts": context
            }

            print(qa_item)

            with open(output_file, 'a') as f:
                json.dump(qa_item, f)
                f.write('\n')

            dataset.append(qa_item)

        sleep(1)
        # sleep(5)

    return dataset    