"""
Data converter for Amazon Nova fine-tuning format.

Converts the guard-rail dataset into the JSONL format required by
Amazon Bedrock Nova models for supervised fine-tuning.
"""
import json
import os
import random
from typing import Optional

from config import config


# System prompt for the guard-rail model
SYSTEM_PROMPT = (
    "당신은 무신사 고객 서비스 AI 에이전트입니다. "
    "고객의 문의를 분석하여 정당한 요청과 부정한 의도(사기, 워드로빙, 허위 불량 신고 등)를 구분합니다. "
    "부정한 의도가 감지되면 정중하지만 단호하게 거부하고, 정당한 문의에는 도움을 제공합니다. "
    "무신사 반품 정책: 제품 불량, 배송 손상, 오배송의 경우에만 반품이 가능합니다."
)


def load_dataset(filepath: str) -> list[dict]:
    """Load the JSON dataset."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_to_nova_format(record: dict) -> dict:
    """
    Convert a single record to Nova fine-tuning format.

    Args:
        record: Dictionary containing the original record with messages.

    Returns:
        Dictionary in Nova JSONL format.
    """
    messages = record.get("messages", [])

    nova_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        nova_messages.append({
            "role": role,
            "content": [{"text": content}]
        })

    return {
        "schemaVersion": "bedrock-conversation-2024",
        "system": [{"text": SYSTEM_PROMPT}],
        "messages": nova_messages
    }


def split_dataset(
    data: list[dict],
    train_ratio: float = 0.9,
    seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """
    Split dataset into training and validation sets.

    Args:
        data: Full dataset.
        train_ratio: Ratio for training data.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (training_data, validation_data).
    """
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def save_jsonl(data: list[dict], filepath: str) -> None:
    """Save data as JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(data)} records to {filepath}")


def validate_jsonl(filepath: str) -> dict:
    """
    Validate JSONL file format for Nova fine-tuning.

    Args:
        filepath: Path to JSONL file.

    Returns:
        Dictionary with validation results.
    """
    errors = []
    valid_count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())

                # Check schema version
                if record.get("schemaVersion") != "bedrock-conversation-2024":
                    errors.append(f"Line {line_num}: Invalid schemaVersion")
                    continue

                # Check messages structure
                messages = record.get("messages", [])
                if not messages:
                    errors.append(f"Line {line_num}: Empty messages array")
                    continue

                # Check first message is user
                if messages[0].get("role") != "user":
                    errors.append(f"Line {line_num}: First message must be from user")
                    continue

                # Check last message is assistant
                if messages[-1].get("role") != "assistant":
                    errors.append(f"Line {line_num}: Last message must be from assistant")
                    continue

                # Check content format
                for msg in messages:
                    content = msg.get("content", [])
                    if not isinstance(content, list) or not content:
                        errors.append(f"Line {line_num}: Content must be non-empty list")
                        break
                    if "text" not in content[0]:
                        errors.append(f"Line {line_num}: Content must have 'text' field")
                        break
                else:
                    valid_count += 1

            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: JSON parse error - {e}")

    total_lines = line_num if 'line_num' in dir() else 0

    return {
        "filepath": filepath,
        "total_lines": total_lines,
        "valid_count": valid_count,
        "error_count": len(errors),
        "errors": errors[:10],
        "is_valid": len(errors) == 0
    }


def main():
    """Main conversion function."""
    # Paths
    input_file = os.path.join(config.data.dataset_dir, config.data.train_json_file)
    output_dir = config.data.output_dir
    training_file = os.path.join(output_dir, config.data.training_data_filename)
    validation_file = os.path.join(output_dir, config.data.validation_data_filename)

    print(f"Loading dataset from {input_file}...")
    data = load_dataset(input_file)
    print(f"Loaded {len(data)} records")

    # Print label distribution
    labels = {}
    for record in data:
        label = record.get("label", "Unknown")
        labels[label] = labels.get(label, 0) + 1
    print(f"Label distribution: {labels}")

    # Convert to Nova format
    print("\nConverting to Nova format...")
    nova_data = [convert_to_nova_format(record) for record in data]

    # Split dataset
    print(f"\nSplitting dataset (train: {config.model.train_ratio}, val: {config.model.validation_ratio})...")
    train_data, val_data = split_dataset(nova_data, config.model.train_ratio)
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Save JSONL files
    print("\nSaving JSONL files...")
    save_jsonl(train_data, training_file)
    save_jsonl(val_data, validation_file)

    # Validate output files
    print("\nValidating output files...")
    train_validation = validate_jsonl(training_file)
    val_validation = validate_jsonl(validation_file)

    print(f"\nTraining file validation: {'✓ Valid' if train_validation['is_valid'] else '✗ Invalid'}")
    print(f"  - Total: {train_validation['total_lines']}, Valid: {train_validation['valid_count']}")
    if train_validation['errors']:
        print(f"  - Errors: {train_validation['errors'][:3]}")

    print(f"\nValidation file validation: {'✓ Valid' if val_validation['is_valid'] else '✗ Invalid'}")
    print(f"  - Total: {val_validation['total_lines']}, Valid: {val_validation['valid_count']}")
    if val_validation['errors']:
        print(f"  - Errors: {val_validation['errors'][:3]}")

    print("\n✓ Conversion complete!")
    print(f"  Training file: {training_file}")
    print(f"  Validation file: {validation_file}")

    return training_file, validation_file


if __name__ == "__main__":
    main()
