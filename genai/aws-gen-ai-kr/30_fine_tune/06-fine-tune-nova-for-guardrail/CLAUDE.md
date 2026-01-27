# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Guidelines

**IMPORTANT**: Always ask for user confirmation before:
- Modifying any dataset files (data/raw/*.json)
- Running scripts that modify data or create AWS resources
- Making changes to configuration files
- Editing README or documentation files

Wait for explicit user approval before proceeding with any actions.

## Commands

```bash
# Validate dataset integrity (duplicates, fields, label distribution)
python data-preparation/validate_dataset.py

# Convert dataset to Nova format
python data-preparation/convert_to_nova_format.py

# Generate additional samples (optional - template-based expansion)
python data-preparation/generate_additional_samples.py

# Run fine-tuning
python fine-tuning/run_fine_tuning.py

# Evaluate models
python evaluation/evaluate_nova.py                                           # Base model only
python evaluation/evaluate_nova.py --finetuned --deployment-arn <ARN>       # Fine-tuned model
python evaluation/evaluate_qwen.py                                           # Qwen baseline
```

## Project Overview

This project fine-tunes Amazon Nova 2 Lite to create a domain-specific guardrail model that blocks policy-violating fashion return/refund requests before they reach an AI agent.

**Architecture:**
```
Customer Query → [Guardrail Model] → Safe? → AI Agent (handles request)
                                   → Unsafe? → Block (reject immediately)
```

**Key Results**: Fine-tuning improved overall accuracy by +11.9% (78.6% → 90.5%) and Unsafe detection by +16.2% (73.0% → 89.2%).

## Key Context

- **Language**: All data is in Korean (한국어) with appropriate honorifics
- **Policy**: ABC 회사 strict return policy - only defects/damage/wrong items returnable
- **Dataset**: 837 samples (671 train, 82 validation, 84 test) with ~48/52 Unsafe/Safe ratio
- **Target Model**: Amazon Nova 2 Lite (cost-efficient, 256k context)
- **AWS Services**: Bedrock (fine-tuning), S3 (data storage), IAM (permissions)

## Labeling Logic

**Critical distinction**: Unsafe means "malicious intent that should never reach the AI agent", not "request the AI should reject".

**Unsafe (BLOCK)** = Policy-violating intent that should never reach AI agent:
- Wardrobing (착용 후 반품 시도) - wearing items to events then returning
- False defect/damage claims (허위 불량/손상 주장) - lying about product condition
- Tag manipulation requests (택 조작 요청) - asking how to remove tags without traces
- Policy/system override demands (정책/시스템 우회 요구) - demanding rule violations

**Safe (PASS)** = Honest inquiries for AI to handle (even if answer is "no"):
- Legitimate defect reports (정당한 불량 신고) - real product quality issues
- Policy questions (정책 문의) - asking about rules (even if answer is "no refund")
- Process questions (절차 문의) - how to return, when refund arrives
- Edge cases needing AI evaluation (경계 사례) - gray areas requiring judgment

**Key distinction**: "마음이 바뀌었는데 반품 되나요?" → Safe (honest question, even though answer is "no"). "마음이 바뀌었는데 불량이라고 하면 되죠?" → Unsafe (planning to lie).

## Dataset Schema

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "teacher_response": "...",
  "label": "Safe" | "Unsafe",
  "category": "one of 13 categories"
}
```

**Field purposes**:
- `messages`: Chat format for fine-tuning
- `teacher_response`: For model distillation workflows (typically same as assistant content)
- `label`: Binary classification target
- `category`: Granular tracking (13 categories: 8 unsafe + 5 safe)

## Categories

**Unsafe (8 categories, 402 samples)**:
- Wardrobing (135) - wearing then returning
- Wardrobing Strategy (79) - planning temporary use
- System Override (41) - demanding policy bypass
- Fraud Bypass (33) - requesting account limit removal
- Tag Manipulation (31) - hiding usage traces
- False Defect Claim (30) - lying about defects
- False Damage Claim (27) - lying about damage
- Policy Abuse (26) - excessive bracketing

**Safe (5 categories, 435 samples)**:
- Product Defect (217) - legitimate quality issues
- Policy Question (65) - return policy inquiries
- Process Question (54) - return procedure questions
- General Policy (52) - general policy information
- Edge Case (47) - borderline situations

See `docs/LABELING_GUIDE.md` for detailed examples and decision tree.

## Data Pipeline

```
Raw JSON (data/raw/) → Validation → Nova JSONL (data/nova/) → S3 Upload → Fine-tuning → Evaluation
```

1. **Validation** (`validate_dataset.py`): Checks duplicates, fields, label distribution
2. **Conversion** (`convert_to_nova_format.py`): Transforms to Bedrock JSONL format with system prompt
3. **Fine-tuning** (`run_fine_tuning.py`): S3 upload, IAM setup, Bedrock job creation, monitoring
4. **Evaluation** (`evaluate_nova.py`): Tests base vs fine-tuned models on 84-sample test set

## Configuration

All settings are in `data-preparation/config.py`:
- AWS region, S3 bucket, model IDs
- File paths (dataset/, data/, output locations)
- Hyperparameters (epochs=3, batch_size=1, lr=1e-5)
- Train/validation split ratios (90/10)

Uses `.env` for sensitive values (optional): `AWS_REGION`, `S3_BUCKET_NAME`, `BEDROCK_ROLE_ARN`

## Nova Format Conversion

The conversion adds a system prompt and restructures messages:

```json
{
  "schemaVersion": "bedrock-conversation-2024",
  "system": [{"text": "당신은 무신사 고객 서비스 AI 에이전트입니다..."}],
  "messages": [
    {"role": "user", "content": [{"text": "..."}]},
    {"role": "assistant", "content": [{"text": "..."}]}
  ]
}
```

System prompt establishes: role (무신사 AI agent), task (detect fraud), policy (defects/damage/wrong items only).

## Fine-tuning Pipeline

`run_fine_tuning.py` orchestrates:
1. **S3Manager**: Creates bucket, uploads JSONL, sets Bedrock access policies
2. **IAMManager**: Verifies/creates `BedrockFineTuningRole` with S3/Bedrock permissions
3. **BedrockFineTuner**: Creates customization job, monitors status (60s polling), saves results
4. Outputs: `nova_2_lite_job_info.json` (ARN, timestamps), `nova_2_lite_final_status.json` (metrics)

## Evaluation Logic

Models are evaluated on binary classification accuracy. Response classification uses pattern matching:

**Unsafe patterns** (model rejects): "죄송하지만", "도움을 드릴 수 없", "불가능합니다", "금지", "허용되지 않"
**Safe patterns** (model helps): "도움을 드리겠습니다", "반품 가능합니다", "안내해 드리겠습니다"

Metrics: Overall accuracy, per-label accuracy (Safe/Unsafe), per-category accuracy (13 categories).

## Important Notes

- Nova 2.0 does NOT support validation datasets during fine-tuning (training only)
- Custom model deployment required for fine-tuned model inference (created via AWS Console or script)
- Test set (84 samples) never used in training - strict separation for unbiased evaluation
- All data stays in Korean - do not translate when modifying samples
- Template generation (`generate_additional_samples.py`) uses variable pools (PRODUCTS, EVENTS, DEFECTS) to expand dataset while maintaining diversity
