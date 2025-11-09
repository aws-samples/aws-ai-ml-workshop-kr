# GEPA Optimizer - Quick Start Guide

## Installation

### Step 1: Copy to Your Project

Simply copy the `gepa-optimizer/` directory to your project:

```bash
cp -r gepa-optimizer /path/to/your/project/
```

### Step 2: Install Dependencies

```bash
cd gepa-optimizer
pip install -r requirements.txt
```

That's it! The only required dependency is `pyyaml`.

## Prerequisites

This framework is designed for projects using:

- **Strands SDK** (`strands-agents>=1.12.0`)
- **Amazon Bedrock** (via `boto3`)
- A module at `src/utils/strands_sdk_utils` with:
  - `get_agent()` function
  - `process_streaming_response_yield()` function

If your project already has these, you're ready to go!

## Your First Optimization

### 1. Create Test Cases

Create a file `test_cases/my_tests.yaml`:

```yaml
test_cases:
  - input: "Hello!"
    expected: "Should greet warmly"
    metadata:
      category: "greeting"
      priority: "medium"

  - input: "Explain AI"
    expected: "Should explain clearly and concisely"
    metadata:
      category: "explanation"
      priority: "high"
```

### 2. Run the Basic Example

```bash
python examples/basic_usage.py
```

This will:
- Load sample test cases
- Run GEPA optimization
- Show you the improved prompt
- Let you save the results

### 3. Customize for Your Needs

```python
import asyncio
from gepa import GEPAOptimizer, LLMEvaluator, TestCaseLoader

async def my_evaluator(result):
    evaluator = LLMEvaluator(
        criteria={
            "accuracy": "Response must be factually correct",
            "tone": "Professional but friendly",
            "format": "Structured with clear sections"
        },
        guidelines=["Be specific", "Use examples"]
    )
    return await evaluator.evaluate(result)

async def main():
    # Load test cases
    loader = TestCaseLoader()
    tests = loader.load("test_cases/my_tests.yaml")

    # Create optimizer
    optimizer = GEPAOptimizer(
        base_prompt="Your current prompt here",
        evaluation_fn=my_evaluator,
        guidelines=["Use XML tags", "Provide examples"],
        verbose=True
    )

    # Optimize!
    best = await optimizer.optimize(tests, max_iterations=5)
    print(f"Final score: {best.score:.2f}")

asyncio.run(main())
```

## Common Use Cases

### Use Case 1: Optimize Existing Prompt

```bash
python examples/optimize_coordinator.py
```

This example shows how to:
- Load an existing prompt from your project
- Create a custom evaluator with domain-specific logic
- Run full GEPA optimization
- Save the improved version

### Use Case 2: Validate New Prompt

```bash
python examples/validate_new_prompt.py \
  --prompt path/to/new_prompt.md \
  --tests test_cases/my_tests.yaml
```

This will:
- Load your new prompt
- Run evaluation (no optimization)
- Give you immediate feedback
- Tell you if it's ready for deployment

### Use Case 3: Interactive Mode

```bash
python examples/validate_new_prompt.py --interactive
```

- Paste your prompt
- Create test cases on the fly
- Get instant validation results

## Understanding the Output

When you run GEPA, you'll see:

```
=== Iteration 1/5 ===
[GENERATE] Running 8 test cases...
  Test 1/8: Hello...
  Test 2/8: Explain AI...

[EVALUATE] Evaluating 8 results...
  Test 1: Score = 0.85
  Test 2: Score = 0.72
  Average Score: 0.78
  Pass Rate: 62.5% (5/8)

[PRODUCE] Generating improved prompt...
  Generated improved prompt (3421 chars)

[ASSESS] Validating improved prompt...
  Score change: 0.78 → 0.85 (+0.07)
  ✅ Improvement confirmed - using new prompt
```

**Score Interpretation:**
- **0.9-1.0**: Excellent - ready for production
- **0.8-0.9**: Good - minor improvements possible
- **0.6-0.8**: Okay - needs optimization
- **<0.6**: Needs work - review test cases or prompt

## Tips for Success

### 1. Create Good Test Cases

✅ **Good:**
```yaml
- input: "Analyze sales data from Q4"
  expected: "Should create detailed plan with data loading, analysis, and visualization steps"
  metadata:
    category: "complex_task"
    priority: "high"
```

❌ **Bad:**
```yaml
- input: "Do something"
  expected: "Works"
```

### 2. Set Realistic Goals

- Start with 3-5 iterations
- Target score: 0.85-0.9 (not 1.0)
- Min improvement: 0.03-0.05

### 3. Review Results Manually

- Don't blindly accept optimized prompts
- Test with real use cases
- Compare to original behavior

### 4. Iterate Gradually

- Start with a few test cases
- Add edge cases as you find them
- Build up comprehensive test suite over time

## Next Steps

1. **Read the Architecture docs** - Understand how GEPA works: `docs/ARCHITECTURE.md`
2. **Check the Examples** - See usage patterns: `examples/README.md`
3. **Integration Guide** - Adapt for your project: `docs/INTEGRATION_GUIDE.md`
4. **Try the Tutorial** - Interactive Jupyter notebook: `tutorials/gepa_optimization.ipynb`

## Troubleshooting

### Error: "No module named 'gepa'"

**Solution:** Make sure you're adding gepa-optimizer to your Python path:

```python
import sys
from pathlib import Path
gepa_root = Path(__file__).parent.parent  # Adjust as needed
sys.path.insert(0, str(gepa_root))
```

### Error: "No module named 'src.utils.strands_sdk_utils'"

**Solution:** Run from your project root where `src/` exists, not from inside `gepa-optimizer/`.

### Low Scores (<0.5)

**Check:**
1. Are test cases too strict?
2. Is the base prompt completely wrong?
3. Are evaluation criteria realistic?

**Try:**
- Start with SimpleEvaluator for structural checks
- Review failed test outputs manually
- Adjust evaluation criteria

### No Improvement After Iterations

**Try:**
1. Increase max_iterations (7-10)
2. Review improvement_context
3. Check if target_score is too ambitious
4. Add more specific guidelines

## Support

- **Issues**: Check `docs/ARCHITECTURE.md` for implementation details
- **Examples**: See `examples/README.md` for patterns
- **Questions**: Review the integration guide

Happy optimizing! 🚀
