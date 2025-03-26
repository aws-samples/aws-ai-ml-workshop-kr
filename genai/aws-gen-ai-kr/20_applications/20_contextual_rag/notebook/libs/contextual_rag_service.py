from datetime import datetime

from libs.bedrock_service import BedrockService
from libs.opensearch_service import OpensearchService
from libs.reranker import RerankerService

import logging
logger = logging.getLogger(__name__)

class ContextualRAGService:
    def __init__(self, bedrock_service: BedrockService, opensearch_service: OpensearchService, reranker_service: RerankerService):
        self.bedrock_service = bedrock_service
        self.opensearch_service = opensearch_service
        self.reranker_service = reranker_service

    def do(self, question: str, index_name: str = None, document_name: str = None, 
           chunk_size: str = None, use_hybrid: bool = True, use_contextual: bool = False, search_limit: int = 5):
        """
        Process a question using RAG approach.
        
        Args:
            question: Question to be answered
            index_name: Direct index name to use (if provided)
            document_name: Document name component for building index name
            chunk_size: Chunk size component for building index name
            use_hybrid: Whether to use hybrid search or just KNN
            use_contextual: Whether to use contextual index prefix
            search_limit: Number of results to return
            
        Returns:
            Dictionary containing the RAG results
        """
        start_dt = datetime.now()

        # Build index name if not directly provided
        if not index_name and document_name and chunk_size:
            index_name = f"{'contextual_' if use_contextual else ''}{document_name}_{chunk_size}"
        elif not index_name:
            raise ValueError("Either index_name or both document_name and chunk_size must be provided")

        # Generate embedding and search
        embedding = self.bedrock_service.embedding(question)
        
        if use_hybrid:
            knn_results = self.opensearch_service.search_by_knn(embedding, index_name, search_limit)
            bm25_results = self.opensearch_service.search_by_bm25(question, index_name, search_limit)
            search_results = self.reranker_service.rank_fusion(question, knn_results, bm25_results, final_reranked_results=search_limit)
        else:
            search_results = self.opensearch_service.search_by_knn(embedding, index_name, search_limit)

        # Prepare context
        docs = ""
        for result in search_results:
            docs += f"- {result['content']}\n\n"

        messages = [{
            'role': 'user',
            'content': [{'text': f"{question}\n\nAdditional Information:\n{docs}"}]
        }]

        processing_time = (datetime.now() - start_dt).microseconds // 1_000
        system_prompt = "You are a helpful AI assistant that provides accurate and concise information about Amazon Bedrock."

        response = self.bedrock_service.converse(
            messages=messages,
            system_prompt=system_prompt
        )

        result = {
            'timestamp': start_dt.isoformat(),
            'question': question,
            'answer': response['output']['message']['content'][0]['text'],
            'retrieved_contexts': response['output']['message']['content'],
            'usage': response['usage'],
            'latency': response['metrics']['latencyMs'],
            'elapsed_time': processing_time + response['metrics']['latencyMs']
        }

        return result
