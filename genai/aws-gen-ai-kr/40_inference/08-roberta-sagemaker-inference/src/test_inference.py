#!/usr/bin/env python3

import json
import numpy as np
import inference

def test_inference():
    print("=" * 60)
    print("Testing SageMaker BiEncoder Inference Functions")
    print("=" * 60)

    print("\n1. Testing model_fn...")
    model = inference.model_fn(".")
    print("   ✓ BiEncoder model loaded")

    print("\n2. Testing input_fn...")
    input_data = {
        "queries": [
            "맛있는 한국 전통 음식 김치찌개",                    # Query 1 (상): 매우 구체적이고 관련성 높음
            "최신 기술 발전",                                  # Query 2 (중): 관련은 있으나 덜 구체적
            "색깔"                                            # Query 3 (하): 문서와 관련 없는 주제
        ],
        "documents": [
            "김치찌개와 된장찌개는 한국의 대표 전통 음식입니다.",  # Doc 1: 높은 관련성 (상)
            "인공지능 기술이 빠르게 발전하고 있습니다.",           # Doc 2: 중간 관련성 (중)
            "파리의 에펠탑은 프랑스의 상징입니다."                 # Doc 3: 낮은 관련성 (하)
        ]
    }
    processed = inference.input_fn(json.dumps(input_data), 'application/json')
    print(f"   ✓ Processed {len(processed['queries'])} queries and {len(processed['documents'])} documents")

    print("\n3. Testing predict_fn...")
    result = inference.predict_fn(processed, model)
    print(f"   ✓ Query embeddings shape: ({result['num_queries']}, {result['embedding_dim']})")
    print(f"   ✓ Document embeddings shape: ({result['num_documents']}, {result['embedding_dim']})")

    print("\n4. Testing output_fn...")
    output_json = inference.output_fn(result, 'application/json')
    output_data = json.loads(output_json)
    print(f"   ✓ Output keys: {list(output_data.keys())}")

    print("\n5. Computing cosine similarity for each pair...")
    query_embs = np.array(output_data['query_embeddings'])
    doc_embs = np.array(output_data['doc_embeddings'])

    # 각 쌍의 유사도 계산 (대각선만)
    print(f"\n   Pair-wise similarity scores:")
    for i in range(len(input_data['queries'])):
        query = input_data['queries'][i]
        doc = input_data['documents'][i]

        # 해당 쌍의 유사도 계산
        similarity = np.dot(query_embs[i], doc_embs[i])

        # 유사도 등급 결정
        if similarity >= 0.75:
            grade = "상 (High)"
        elif similarity >= 0.60:
            grade = "중 (Medium)"
        else:
            grade = "하 (Low)"

        print(f"\n   Pair {i+1}: [{similarity:.4f}] - {grade}")
        print(f"      Query: '{query}'")
        print(f"      Doc:   '{doc[:50]}...'")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_inference()