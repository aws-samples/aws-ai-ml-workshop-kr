import numpy as np
import time

def test_biencoder_pairs(predictor, query_doc_pairs):
    """
    BiEncoder를 사용하여 여러 쿼리-문서 쌍의 유사도를 계산합니다.
    
    Args:
        predictor: SageMaker predictor 객체
        query_doc_pairs: [(query, document), ...] 형태의 쿼리-문서 쌍 리스트
    """
    queries = [pair[0] for pair in query_doc_pairs]
    documents = [pair[1] for pair in query_doc_pairs]
    
    # BiEncoder 추론
    start = time.time()
    result = predictor.predict({
        "queries": queries,
        "documents": documents
    })
    encode_time = (time.time() - start) * 1000
    
    print(f"Encoding time: {encode_time:.1f}ms\n")
    
    # 유사도 계산
    query_embs = np.array(result['query_embeddings'])
    doc_embs = np.array(result['doc_embeddings'])
    
    # 각 쌍의 유사도 출력
    for i in range(len(queries)):
        similarity = np.dot(query_embs[i], doc_embs[i])
        
        if similarity >= 0.75:
            grade = "상"
        elif similarity >= 0.60:
            grade = "중"
        else:
            grade = "하"
        
        print(f"Pair {i+1}: [{similarity:.4f}] - {grade}")
        print(f"  Q: {queries[i]}")
        print(f"  D: {documents[i]}\n")