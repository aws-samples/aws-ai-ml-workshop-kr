# GEPA Optimization Examples

This directory contains example scripts demonstrating how to use the GEPA (Generate, Evaluate, Produce, Assess) optimizer for prompt engineering.

## Available Scripts

### 1. optimize_coordinator.py

**Purpose**: Optimize the existing Coordinator agent prompt using GEPA methodology.

**Usage**:
```bash
# From project root
python examples/optimize_coordinator.py
```

**What it does**:
- Loads the current coordinator.md prompt
- Runs test cases from data/test_cases/coordinator_tests.yaml
- Uses LLM judge to evaluate responses
- Iteratively improves the prompt through GEPA cycles
- Saves optimized prompt to src/prompts/coordinator_optimized.md
- Saves optimization history to artifacts/coordinator_optimization_history.json

**Key Features**:
- Custom evaluator that checks handoff logic
- Integration with skills/system-prompt-writer guidelines
- Configurable iterations and target scores
- Detailed progress reporting

**Example Output**:
```
GEPA OPTIMIZATION START
==================================================================
Base prompt length: 3421 chars
Test cases: 8
Max iterations: 3
Target score: 0.9

=== Iteration 1/3 ===
[GENERATE] Running 8 test cases...
[EVALUATE] Evaluating 8 results...
  Average Score: 0.75
  Pass Rate: 62.5% (5/8)
[PRODUCE] Generating improved prompt...
[ASSESS] Validating improved prompt...
  Score change: 0.75 → 0.85 (+0.10)
  ✅ Improvement confirmed

...

OPTIMIZATION COMPLETE
Best version: 2
Best score: 0.87
```

---

### 2. validate_new_prompt.py

**Purpose**: Validate a newly created prompt without full optimization - quick quality check.

**Usage**:

**Interactive Mode**:
```bash
python examples/validate_new_prompt.py --interactive
```
Enter your prompt, create or load test cases interactively.

**File Mode**:
```bash
python examples/validate_new_prompt.py \
  --prompt src/prompts/my_new_prompt.md \
  --tests data/test_cases/my_tests.yaml
```

**What it does**:
- Loads your new prompt
- Runs evaluation (no optimization)
- Provides immediate feedback on quality
- Suggests improvements if needed
- Optionally saves validation report

**Example Output**:
```
VALIDATION RESULTS
==================================================================
Average Score: 0.88
Pass Rate: 87.5% (7/8)

Score Breakdown:
  ✓ Test 1: 0.92
  ✓ Test 2: 0.85
  ✗ Test 3: 0.65
     → Missing required keyword handling

RECOMMENDATIONS
==================================================================
👍 Good! Minor improvements recommended:
   - Review failed test cases
   - Consider running GEPA optimization for refinement
```

---

## Quick Start Guide

### Optimizing an Existing Prompt

1. **Prepare test cases**:
   ```python
   from src.optimizer.test_loader import TestCaseBuilder

   builder = TestCaseBuilder()
   builder.add("Hello", expected="Should greet warmly")
   builder.add("Analyze data", expected="Should plan analysis")
   builder.save("my_agent_tests.yaml")
   ```

2. **Create custom evaluator** (optional):
   ```python
   async def my_evaluator(result):
       from src.optimizer import LLMEvaluator

       evaluator = LLMEvaluator(
           criteria={"format": "JSON output", "tone": "Professional"},
           guidelines=["Be specific", "Use examples"]
       )
       return await evaluator.evaluate(result)
   ```

3. **Run optimization**:
   ```python
   from src.optimizer import GEPAOptimizer
   from src.prompts.template import apply_prompt_template

   current_prompt = apply_prompt_template("my_agent")

   optimizer = GEPAOptimizer(
       base_prompt=current_prompt,
       evaluation_fn=my_evaluator,
       verbose=True
   )

   best = await optimizer.optimize(test_cases, max_iterations=5)
   ```

### Validating a New Prompt

1. **Create your prompt** in `src/prompts/my_new_agent.md`

2. **Create test cases** in `data/test_cases/my_new_agent_tests.yaml`

3. **Run validation**:
   ```bash
   python examples/validate_new_prompt.py \
     --prompt src/prompts/my_new_agent.md \
     --tests data/test_cases/my_new_agent_tests.yaml
   ```

4. **Review results** and iterate if needed

---

## Common Patterns

### Pattern 1: Quick Validation Before Deployment

```python
# validate.py
from examples.validate_new_prompt import validate_prompt_from_file

score = await validate_prompt_from_file(
    "src/prompts/new_feature.md",
    "data/test_cases/new_feature_tests.yaml"
)

if score >= 0.8:
    print("✅ Prompt ready for deployment")
else:
    print("❌ Needs improvement - run GEPA optimization")
```

### Pattern 2: Continuous Optimization

```python
# optimize_all.py
agents = ["coordinator", "planner", "supervisor"]

for agent in agents:
    optimizer = create_optimizer_for_agent(agent)
    best = await optimizer.optimize(test_cases, max_iterations=3)
    save_prompt(agent, best.prompt)
```

### Pattern 3: A/B Testing Prompts

```python
# compare_prompts.py
from src.optimizer import GEPAOptimizer

prompts = {
    "version_a": load_prompt("coordinator.md"),
    "version_b": load_prompt("coordinator_v2.md")
}

results = {}
for name, prompt in prompts.items():
    optimizer = GEPAOptimizer(base_prompt=prompt, ...)
    test_results = await optimizer.generate(test_cases)
    eval_results = await optimizer.evaluate(test_results)
    results[name] = eval_results['avg_score']

winner = max(results, key=results.get)
print(f"🏆 Winner: {winner} (score: {results[winner]:.2f})")
```

---

## Best Practices

1. **Test Case Quality**:
   - Cover edge cases, not just happy paths
   - Include both simple and complex scenarios
   - Use realistic user inputs
   - Define clear expected behaviors

2. **Evaluation Criteria**:
   - Be specific about what "good" looks like
   - Balance structural and quality checks
   - Include domain-specific requirements
   - Use weighted scoring for priorities

3. **Optimization Settings**:
   - Start with 3-5 iterations
   - Set realistic target scores (0.85-0.9)
   - Use min_improvement to avoid diminishing returns
   - Enable verbose mode for debugging

4. **Integration**:
   - Validate before deploying to production
   - Keep optimization history for auditing
   - Review LLM judge feedback manually
   - Run regression tests on old prompts

---

## Troubleshooting

**Issue**: Optimization not improving scores

**Solutions**:
- Check if test cases are too easy/hard
- Review evaluation criteria clarity
- Increase max_iterations
- Manually review failing test outputs

**Issue**: LLM evaluator gives inconsistent scores

**Solutions**:
- Make evaluation criteria more specific
- Add concrete examples to guidelines
- Use SimpleEvaluator for structural checks first
- Increase evaluation samples

**Issue**: Optimized prompt is too long

**Solutions**:
- Add length constraint to guidelines
- Manually edit and re-validate
- Use more concise examples in improvement_context

---

## Additional Resources

- GEPA Methodology: See `tutorials/gepa_optimization.ipynb`
- Test Case Format: See `data/test_cases/README.md`
- Evaluator API: See `src/optimizer/llm_evaluator.py` docstrings
- Optimizer API: See `src/optimizer/gepa_optimizer.py` docstrings
