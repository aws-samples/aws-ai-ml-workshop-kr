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

