import requests
import logging
import boto3
import botocore
logger = logging.getLogger(__name__)

class RerankerService:
    def __init__(self, aws_region: str, aws_profile: str, reranker_model_id: str, retries: int):
        self.aws_region = aws_region
        self.aws_profile = aws_profile
        self.reranker_model_id = reranker_model_id
        self.retries = retries
        self.reranker_client = self._init_reranker_client(aws_region, aws_profile, retries)

    def _init_reranker_client(self, aws_region: str, aws_profile: str, retries: int):
        retry_config = botocore.config.Config(
            retries={"max_attempts": retries, "mode": "standard"}
        )
        return boto3.Session(
            region_name=aws_region,
            profile_name=aws_profile
        ).client("bedrock-agent-runtime", config=retry_config)
        
    def rank_fusion(self, question, knn_results, bm25_results, hybrid_score_filter=40, final_reranked_results=20, knn_weight=0.6):
        bm25_weight = 1 - knn_weight

        def _normalize_and_weight_score(results, weight):
            if not results:
                return results
            min_score = min(r['score'] for r in results)
            max_score = max(r['score'] for r in results)
            score_range = max_score - min_score
            if score_range == 0:
                return results
            for r in results:
                r['normalized_score'] = ((r['score'] - min_score) / score_range) * weight
            return results

        knn_results = _normalize_and_weight_score(knn_results, knn_weight)
        bm25_results = _normalize_and_weight_score(bm25_results, bm25_weight)
        
        # Combine results and calculate hybrid score
        combined_results = {}
        for result in knn_results + bm25_results:
            chunk_id = result['metadata'].get('chunk_id', result['content']) 
            if chunk_id not in combined_results:
                combined_results[chunk_id] = result.copy()
                combined_results[chunk_id]['hybrid_score'] = result.get('normalized_score', 0)
                combined_results[chunk_id]['search_methods'] = [result['search_method']]
            else:
                combined_results[chunk_id]['hybrid_score'] += result.get('normalized_score', 0)
                if result['search_method'] not in combined_results[chunk_id]['search_methods']:
                    combined_results[chunk_id]['search_methods'].append(result['search_method'])

        # Convert back to list and sort by hybrid score
        results_list = list(combined_results.values())
        results_list.sort(key=lambda x: x['hybrid_score'], reverse=True)
        hybrid_results = results_list[:hybrid_score_filter]

        # Prepare documents for reranking
        documents_for_rerank = [
            {"content": doc['content'], "metadata": doc['metadata']} for doc in hybrid_results
        ]

        # Rerank the documents -> return ranked indices
        reranked_results = self._rerank_documents(question, documents_for_rerank, final_reranked_results)

        # Prepare final results
        if reranked_results and isinstance(reranked_results, dict) and 'results' in reranked_results:
            final_results = []
            for reranked_doc in reranked_results['results']:
                if isinstance(reranked_doc, dict) and 'index' in reranked_doc and 'relevance_score' in reranked_doc:
                    index = reranked_doc['index']
                    if 0 <= index < len(hybrid_results):
                        original_doc = hybrid_results[index]
                        final_doc = {
                            "content": original_doc["content"],
                            'metadata': original_doc['metadata'],
                            'score': reranked_doc['relevance_score'], 
                            'hybrid_score': original_doc['hybrid_score'],
                            'search_methods': original_doc['search_methods']
                        }   
                        final_results.append(final_doc)
                else:
                    logger.warning(f"Unexpected reranked document format: {reranked_doc}")

            final_results.sort(key=lambda x: x['score'], reverse=True)

        else:
            logger.warning("Reranking failed or returned unexpected format. Using hybrid results.")
            final_results = [{
                "content": doc["content"],
                'metadata': doc['metadata'],
                'score': doc['hybrid_score'],
                'hybrid_score': doc['hybrid_score'],
                'search_methods': doc['search_methods']
            } for doc in hybrid_results[:final_reranked_results]]
            
        return final_results
    
    def _rerank_documents(self, question, documents, top_k=20):
        text_sources = []
        for doc in documents:
            text_sources.append({
                "type": "INLINE",
                "inlineDocumentSource": {
                    "type": "TEXT",
                    "textDocument": {
                        "text": doc['content'],
                    }
                }
            })
        
        model_package_arn = f"arn:aws:bedrock:{self.aws_region}::foundation-model/{self.reranker_model_id}"
        response = self.reranker_client.rerank(
        queries=[
            {
                    "type": "TEXT",
                    "textQuery": {
                        "text": question
                    }
                }
            ],
            sources=text_sources,
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "numberOfResults": top_k,
                    "modelConfiguration": {
                        "modelArn": model_package_arn,
                    }
                }
            }
        )

        results = {
            "results": [
                {
                    "index": result['index'],
                    "relevance_score": result['relevanceScore']
                } for result in response['results']
            ]
        }
        return results

    