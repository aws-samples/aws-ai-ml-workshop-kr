# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Validate dataset integrity (duplicates, fields, label distribution)
python data-preparation/validate_dataset.py

# Convert dataset to Nova format
python data-preparation/convert_to_nova_format.py

# Run fine-tuning
python fine-tuning/run_fine_tuning.py

# Evaluate models
python evaluation/evaluate_nova.py
python evaluation/evaluate_qwen.py
```

## Project Overview

This project creates datasets for training guardrail models that block policy-violating fashion return/refund requests before they reach an AI agent.

**Architecture:**
```
Customer Query → [Guardrail Model] → Safe? → AI Agent
                                   → Unsafe? → Block
```

## Key Context

- **Language**: All data is in Korean (한국어) with appropriate honorifics
- **Policy**: ABC 회사 strict return policy - only defects/damage/wrong items returnable
- **Dataset**: 837 samples (671 train, 82 validation, 84 test) with ~48/52 Unsafe/Safe ratio

## Labeling Logic

**Unsafe (BLOCK)** = Policy-violating intent that should never reach AI agent:
- Wardrobing (착용 후 반품 시도)
- False defect/damage claims (허위 불량/손상 주장)
- Tag manipulation requests (택 조작 요청)
- Policy/system override demands (정책/시스템 우회 요구)

**Safe (PASS)** = Honest inquiries for AI to handle (even if answer is "no"):
- Legitimate defect reports (정당한 불량 신고)
- Policy questions (정책 문의)
- Process questions (절차 문의)
- Edge cases needing AI evaluation (경계 사례)

**Key distinction**: "마음이 바뀌었는데 반품 되나요?" → Safe (honest question). "마음이 바뀌었는데 불량이라고 하면 되죠?" → Unsafe (planning to lie).

## Dataset Schema

```json
{
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "teacher_response": "...",
  "label": "Safe" | "Unsafe",
  "category": "one of 13 categories"
}
```

## Categories

- **Unsafe (8)**: Wardrobing, Tag Manipulation, False Defect Claim, System Override, Wardrobing Strategy, False Damage Claim, Fraud Bypass, Policy Abuse
- **Safe (5)**: Product Defect, Policy Question, Process Question, General Policy, Edge Case

See `docs/LABELING_GUIDE.md` for detailed examples and decision tree.
