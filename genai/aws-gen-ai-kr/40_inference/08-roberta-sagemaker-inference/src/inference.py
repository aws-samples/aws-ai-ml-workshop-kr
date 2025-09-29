#!/usr/bin/env python3
"""
SageMaker Inference Script for KLUE RoBERTa BiEncoder
"""

import os
import json
import logging
import sys
import traceback
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class BiEncoder(nn.Module):
    """Dual Encoder for query and document encoding"""
    def __init__(self, model_path, device):
        super().__init__()
        self.query_encoder = AutoModel.from_pretrained(model_path)
        self.doc_encoder = AutoModel.from_pretrained(model_path)
        self.device = device

    def forward(self, query_inputs, doc_inputs):
        query_outputs = self.query_encoder(**query_inputs)
        doc_outputs = self.doc_encoder(**doc_inputs)

        # Mean pooling
        query_emb = self._mean_pooling(query_outputs.last_hidden_state, query_inputs['attention_mask'])
        doc_emb = self._mean_pooling(doc_outputs.last_hidden_state, doc_inputs['attention_mask'])

        # L2 정규화
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
        doc_emb = torch.nn.functional.normalize(doc_emb, p=2, dim=1)

        return query_emb, doc_emb

    def _mean_pooling(self, hidden_states, attention_mask):
        attention_mask_expanded = attention_mask.unsqueeze(-1)
        masked_hidden_states = hidden_states * attention_mask_expanded
        sum_hidden_states = torch.sum(masked_hidden_states, dim=1)
        sum_attention_mask = torch.sum(attention_mask_expanded, dim=1)
        return sum_hidden_states / sum_attention_mask


def model_fn(model_dir):
    logger.info(f"Loading BiEncoder model from {model_dir}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        local_model_dir = os.path.join(os.path.dirname(__file__), "model")

        if os.path.exists(os.path.join(local_model_dir, "config.json")):
            model_path = local_model_dir
            logger.info(f"Loading model from local directory: {local_model_dir}")
        elif os.path.exists(os.path.join(model_dir, "config.json")):
            model_path = model_dir
            logger.info(f"Loading model from model_dir: {model_dir}")
        else:
            model_path = "klue/roberta-base"
            logger.info(f"Downloading model from HuggingFace: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = BiEncoder(model_path, device)

        model = model.to(device)
        model.eval()

        model.tokenizer = tokenizer
        model._device = device

        logger.info("BiEncoder model loaded successfully")

        return model

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def input_fn(input_data, content_type):
    logger.info(f"Received content_type: {content_type}")

    try:
        if content_type == 'application/json':
            input_str = input_data
            if isinstance(input_data, bytes):
                input_str = input_data.decode('utf-8')

            data = json.loads(input_str)

            # 입력 형식 검증: queries와 documents 필요
            if 'queries' not in data or 'documents' not in data:
                raise ValueError("Input JSON must contain 'queries' and 'documents' fields")

            queries = data['queries']
            documents = data['documents']

            if not isinstance(queries, list):
                queries = [queries]
            if not isinstance(documents, list):
                documents = [documents]

            max_length = data.get('max_length', 128)

            return {
                'queries': queries,
                'documents': documents,
                'max_length': max_length
            }
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

    except Exception as e:
        logger.error(f"Error processing input: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def predict_fn(data, model):
    logger.info("Starting BiEncoder prediction")

    try:
        tokenizer = model.tokenizer
        device = model._device

        queries = data['queries']
        documents = data['documents']
        max_length = data['max_length']

        logger.info(f"Processing {len(queries)} query(s) and {len(documents)} document(s)")

        # 쿼리 토크나이징
        query_inputs = tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # 문서 토크나이징
        doc_inputs = tokenizer(
            documents,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # GPU로 이동
        query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
        doc_inputs = {k: v.to(device) for k, v in doc_inputs.items()}

        # BiEncoder 추론
        with torch.no_grad():
            query_embeddings, doc_embeddings = model(query_inputs, doc_inputs)

        result = {
            'query_embeddings': query_embeddings.cpu().numpy().tolist(),
            'doc_embeddings': doc_embeddings.cpu().numpy().tolist(),
            'embedding_dim': query_embeddings.shape[1],
            'num_queries': len(queries),
            'num_documents': len(documents)
        }

        logger.info("BiEncoder prediction completed successfully")
        return result

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def output_fn(prediction, accept):
    logger.info(f"Formatting output with accept: {accept}")

    try:
        if accept == 'application/json':
            return json.dumps(prediction)
        else:
            raise ValueError(f"Unsupported accept type: {accept}")

    except Exception as e:
        logger.error(f"Error formatting output: {str(e)}")
        logger.error(traceback.format_exc())
        raise