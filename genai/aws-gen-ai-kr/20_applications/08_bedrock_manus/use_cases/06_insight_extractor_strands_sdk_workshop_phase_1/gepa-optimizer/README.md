# GEPA Optimizer

A systematic framework for prompt optimization using the **GEPA methodology** (Generate, Evaluate, Produce, Assess).

Designed for use with **Strands SDK** and **Amazon Bedrock**, but adaptable to other LLM frameworks.

## Quick Start

### 1. Copy to Your Project

```bash
# Simply copy the gepa-optimizer directory to your project
cp -r gepa-optimizer /path/to/your/project/
```

### 2. Install Dependencies

```bash
cd gepa-optimizer
pip install -r requirements.txt
```

### 3. Use It

```python
from gepa import GEPAOptimizer, LLMEvaluator, TestCaseLoader

# Load test cases
loader = TestCaseLoader()
tests = loader.load("test_cases/sample_tests.yaml")

# Create evaluator
async def my_evaluator(result):
    evaluator = LLMEvaluator(
        criteria={"clarity": "Clear and concise"},
        guidelines=["Be specific"]
    )
    return await evaluator.evaluate(result)

# Run optimization
optimizer = GEPAOptimizer(
    base_prompt="Your prompt here",
    evaluation_fn=my_evaluator,
    verbose=True
)

best = await optimizer.optimize(tests, max_iterations=5, target_score=0.9)
print(f"Optimized prompt score: {best.score}")
```

## What is GEPA?

GEPA is a structured 4-phase methodology for prompt optimization:

1. **Generate**: Run tests with your current prompt
2. **Evaluate**: Assess output quality using LLM judge or custom criteria
3. **Produce**: Create an improved prompt based on feedback
4. **Assess**: Validate the improvement and decide whether to continue

## Features

- 🤖 **LLM-as-Judge**: Automatic evaluation using Claude as evaluator
- 📝 **Test Management**: YAML/JSON-based test case organization
- 🎯 **Custom Evaluators**: Define domain-specific evaluation criteria
- 📊 **Progress Tracking**: Version history and score visualization
- 🔧 **Iterative Improvement**: Systematic optimization through multiple cycles
- 🚀 **Portable**: Single directory - copy and use anywhere

## Directory Structure

```
gepa-optimizer/
├── gepa/                    # Core framework
│   ├── optimizer.py         # GEPA engine
│   ├── evaluator.py         # LLM & rule-based evaluators
│   └── test_loader.py       # Test case management
│
├── examples/                # Usage examples
│   ├── optimize_coordinator.py
│   └── validate_new_prompt.py
│
├── test_cases/              # Test case repository
│   ├── coordinator_tests.yaml
│   ├── planner_tests.yaml
│   └── sample_tests.yaml
│
├── tutorials/               # Learning resources
│   └── gepa_optimization.ipynb
│
└── docs/                    # Documentation
    ├── ARCHITECTURE.md
    ├── QUICK_START.md
    └── INTEGRATION_GUIDE.md
```

## Examples

### Optimize an Existing Prompt

```bash
python examples/optimize_coordinator.py
```

### Validate a New Prompt

```bash
python examples/validate_new_prompt.py \
  --prompt path/to/prompt.md \
  --tests test_cases/my_tests.yaml
```

### Interactive Tutorial

```bash
jupyter lab tutorials/gepa_optimization.ipynb
```

## Requirements

### Minimal

- Python 3.12+
- `pyyaml` (for test case loading)

### For Full Functionality

This framework is designed for projects using:
- **Strands SDK** (`strands-agents>=1.12.0`)
- **Amazon Bedrock** (via `boto3`)
- A module at `src/utils/strands_sdk_utils` with:
  - `get_agent()` - Agent creation
  - `process_streaming_response_yield()` - Streaming response handling

If your project uses different LLM infrastructure, you'll need to adapt the agent creation calls in `gepa/optimizer.py` and `gepa/evaluator.py`.

## Documentation

- **[Quick Start](docs/QUICK_START.md)** - Detailed getting started guide
- **[Architecture](docs/ARCHITECTURE.md)** - Implementation details and design
- **[Integration Guide](docs/INTEGRATION_GUIDE.md)** - How to integrate with your project
- **[Examples README](examples/README.md)** - Usage patterns and examples

## Use Cases

- **Optimize existing agent prompts** - Improve coordinator, planner, or tool agent prompts
- **Validate new prompts** - Quality check before deployment
- **A/B test prompts** - Compare different prompt versions
- **Continuous improvement** - Iteratively refine prompts based on new test cases

## How It Works

1. **Define test cases** (YAML/JSON format)
2. **Create an evaluator** (LLM-based or rule-based)
3. **Run GEPA optimization** (3-5 iterations typical)
4. **Get improved prompt** with version history

The optimizer automatically:
- Tests your prompt against all cases
- Evaluates outputs for quality
- Generates improved versions
- Tracks progress and scores
- Saves optimization history

## License

MIT License - See LICENSE file

## Contributing

Contributions welcome! This is a standalone component extracted from the Deep Insight project.

---

**Built for Strands SDK + Amazon Bedrock**
Portable, systematic prompt optimization for agentic AI systems
