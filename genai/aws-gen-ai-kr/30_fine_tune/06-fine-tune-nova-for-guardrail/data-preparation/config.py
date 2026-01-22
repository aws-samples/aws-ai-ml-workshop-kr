"""
Configuration settings for Guard Rail Fine-tuning.
"""
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class AWSConfig:
    """AWS service configuration."""
    region: str = os.getenv("AWS_REGION", "us-east-1")
    s3_bucket_name: str = os.getenv("S3_BUCKET_NAME", "guard-rail-fine-tuning-data")
    bedrock_model_id: str = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0:300k")
    bedrock_custom_model_name: str = os.getenv("BEDROCK_CUSTOM_MODEL_NAME", "guard-rail-nova-pro")
    bedrock_role_arn: str = os.getenv("BEDROCK_ROLE_ARN", "")


@dataclass
class DataConfig:
    """Dataset configuration."""
    base_dir: str = os.path.dirname(os.path.dirname(__file__))
    dataset_dir: str = os.path.join(base_dir, "dataset")
    output_dir: str = os.path.join(base_dir, "data")

    # Input file
    train_json_file: str = "train.json"

    # Output files (JSONL format for Nova)
    training_data_filename: str = "training_data.jsonl"
    validation_data_filename: str = "validation_data.jsonl"


@dataclass
class ModelConfig:
    """Model training configuration."""
    # Fine-tuning hyperparameters for Nova Pro
    epochs: int = 3
    batch_size: int = 1
    learning_rate: float = 1e-5
    learning_rate_warmup_steps: int = 0

    # Data split ratios
    train_ratio: float = 0.9
    validation_ratio: float = 0.1


@dataclass
class AppConfig:
    """Main application configuration."""
    aws: AWSConfig
    data: DataConfig
    model: ModelConfig

    @classmethod
    def load(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        return cls(
            aws=AWSConfig(),
            data=DataConfig(),
            model=ModelConfig()
        )


# Global configuration instance
config = AppConfig.load()
