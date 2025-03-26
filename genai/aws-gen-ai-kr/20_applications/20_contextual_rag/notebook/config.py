from dataclasses import dataclass
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(override=True)

class Config:
    @dataclass
    class AWSConfig:
        region: str = os.getenv('AWS_REGION', 'us-west-2')
        profile: str = os.getenv('AWS_PROFILE', 'default')

    @dataclass
    class BedrockConfig:
        model_id: str = os.getenv('BEDROCK_MODEL_ID')
        embed_model_id: str = os.getenv('EMBED_MODEL_ID')
        retries: int = int(os.getenv('BEDROCK_RETRIES', 10))

    @dataclass
    class ModelConfig:
        max_tokens: int = int(os.getenv('MAX_TOKENS', 4096))
        temperature: float = float(os.getenv('TEMPERATURE', 0))
        top_p: float = float(os.getenv('TOP_P', 0.7))

    @dataclass
    class OpenSearchConfig:
        domain_name: None = os.getenv('OPENSEARCH_DOMAIN_NAME')
        document_name: str = os.getenv('OPENSEARCH_DOCUMENT_NAME')
        prefix: str = os.getenv('OPENSEARCH_PREFIX')
        user: None = os.getenv('OPENSEARCH_USER')
        password: None = os.getenv('OPENSEARCH_PASSWORD')

    @dataclass
    class RerankerConfig:
        reranker_model_id: str = os.getenv('RERANKER_MODEL_ID')
        aws_region: str = os.getenv('RERANKER_AWS_REGION', 'us-west-2')
        aws_profile: str = os.getenv('RERANKER_AWS_PROFILE')

    @dataclass
    class RankFusionConfig:
        top_k: int = int(os.getenv('RERANK_TOP_K', 20))
        hybrid_score_filter: int = int(os.getenv('HYBRID_SCORE_FILTER', 40))
        final_reranked_results: int = int(os.getenv('FINAL_RERANKED_RESULTS', 20))
        knn_weight: float = float(os.getenv('KNN_WEIGHT', 0.6))

    @dataclass
    class AppConfig:
        chunk_size: str = os.getenv('CHUNK_SIZE', '1000')
        rate_limit_delay: str = int(os.getenv('RATE_LIMIT_DELAY', 60))

    def __init__(self):        
        self.aws = self.AWSConfig()
        self.bedrock = self.BedrockConfig()
        self.model = self.ModelConfig()
        self.opensearch = self.OpenSearchConfig()
        self.reranker = self.RerankerConfig()
        self.rank_fusion = self.RankFusionConfig()
        self.app = self.AppConfig()

    @classmethod
    def load(cls) -> 'Config':
        return cls()