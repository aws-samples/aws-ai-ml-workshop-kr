import pdfplumber
import re

import json

from tqdm import tqdm
from typing import Dict, Any

class DocumentParser:
    # Load PDF into text
    @staticmethod
    def load_pdf(file_path, start_page=0, end_page=-1) -> str:
        full_text = ""
        
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            end_page = total_pages if end_page == -1 else min(end_page or total_pages, total_pages)
            
            for page_num in tqdm(range(start_page - 1, end_page)):
                text = pdf.pages[page_num].extract_text()
                text = re.sub(r'\s+', ' ', text).strip()
                full_text = "".join([full_text, text, " "])
                
        print(f"PDF Load: {file_path}")
        return full_text

    @staticmethod
    def load_pdf_with_metadata(file_path, start_page=0, end_page=-1) -> Dict[str, Any]:
        metadata = {}

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)

            start_page = (start_page or 1) - 1
            end_page = min(end_page or total_pages, total_pages)

            metadata = {
                "source": file_path,
                "filename": file_path,
                "total_pages": total_pages,
                "extracted_pages": f"{start_page + 1}-{end_page}"
            }

            for key, value in pdf.metadata.items():
                if isinstance(value, (str, int)):
                    metadata[key] = value

        return {
            "page_content": DocumentParser.load_pdf(file_path, start_page, end_page),
            "metadata": metadata
        }

    # process file
    @staticmethod
    def split(full_text, chunk_size, max_document_length=-1):
        documents = _split_into_chunks(full_text, max_document_length, 0) if max_document_length > 0 else [full_text]
        results = []

        for doc_id, document in enumerate(tqdm(documents)):
            result = {
                "doc_id": doc_id,
                "content": document,
                "chunks": []
            }

            for chunk_id, chunk in enumerate(_split_into_chunks(document, chunk_size, int(chunk_size * 0.08))):
                result["chunks"].append({
                    "chunk_id": chunk_id,
                    "content": chunk
                })

            results.append(result)    
        return results

    @staticmethod
    def situate_document(bedrock_service, index_name, document_file):
        # logger.info(f"Starting to situate {len(documents)} documents")
        total_token_usage = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
        documents_token_usage = {}

        sys_prompt = """
        You're an expert at providing a succinct context, targeted for specific text chunks.

        <instruction>
        - Offer 1-5 short sentences that explain what specific information this chunk provides within the document.
        - Focus on the unique content of this chunk, avoiding general statements about the overall document.
        - Clarify how this chunk's content relates to other parts of the document and its role in the document.
        - If there's essential information in the document that backs up this chunk's key points, mention the details.
        </instruction>
        """


        with open(document_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        for doc_id, document in enumerate(tqdm(documents)):
            doc_content = document['content']

            doc_token_usage = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
            
            for chunk in document:
                document_context_prompt = f"""
                <document>
                {doc_content}
                </document>
                """

                chunk_content = chunk['content']
                chunk_context_prompt = f"""
                Here is the chunk we want to situate within the whole document:

                <chunk>
                {chunk_content}
                </chunk>

                Skip the preamble and only provide the consise context.
                """
                usr_prompt = [{
                        "role": "user", 
                        "content": [
                            {"text": document_context_prompt},
                            {"text": chunk_context_prompt}
                        ]
                    }]

                temperature = 0.0
                top_p = 0.5
                inference_config = {"temperature": temperature, "topP": top_p}

                try:
                    response = bedrock_service.converse(
                        messages = usr_prompt, 
                        system_prompt = sys_prompt,
                    )
                    situated_context = response['output']['message']['content'][0]['text'].strip()
                    chunk['content'] = f"Context:\n{situated_context}\n\nChunk:\n{chunk['content']}"

                    if 'usage' in response:
                        usage = response['usage']
                        for key in ['inputTokens', 'outputTokens', 'totalTokens']:
                            doc_token_usage[key] += usage.get(key, 0)
                            total_token_usage[key] += usage.get(key, 0)

                except Exception as e:
                    print(f"Error generating context for chunk: {e}")

            documents_token_usage[f"document_{doc_id}"] = doc_token_usage
            print(f"Completed processing document {doc_id}/{len(documents)}")
            print(f"Document {doc_id} token usage - Input: {doc_token_usage['inputTokens']}, "
                        f"Output: {doc_token_usage['outputTokens']}, Total: {doc_token_usage['totalTokens']}")

            print(f"Total token usage - Input: {total_token_usage['inputTokens']}, "
                        f"Output: {total_token_usage['outputTokens']}, "
                        f"Total: {total_token_usage['totalTokens']}")

            token_usage_data = {
                "total_usage": total_token_usage,
                "documents_usage": documents_token_usage
            }

        with open(f"{index_name}_token_usage.json", 'w') as f:
            json.dump(token_usage_data, f, indent=4)
        print(f"Token usage saved to {index_name}_token_usage.json")

        return documents

def _split_into_chunks(text, chunk_size, overlap: int):
    # Initialize list and variables once
    chunks = []
    start = 0
    text_length = len(text)

    # Compile regex pattern once outside loop
    p = re.compile(r'[\.!?] |[\n$]')
    
    while start < text_length:
        end = start + chunk_size
        
        # Check text bounds first
        if end >= text_length:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Search for sentence boundary
        # Find sentence boundary and adjust end position in one step
        s = p.search(text, end)
        end = (s.span()[1] if s and s.span()[1] - end <= chunk_size else text.find(' ', end)) if s else end
            

        # Get chunk and append if non-empty
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Calculate next start position
        start = max(end - overlap, start + 1)

    return chunks

