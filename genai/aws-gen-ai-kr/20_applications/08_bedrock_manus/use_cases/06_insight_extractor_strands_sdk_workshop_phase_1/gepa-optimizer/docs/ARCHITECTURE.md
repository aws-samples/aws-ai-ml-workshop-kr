# GEPA Optimizer Implementation Guide

## Overview

This document provides a comprehensive overview of the GEPA (Generate, Evaluate, Produce, Assess) optimizer implementation for systematic prompt engineering in Deep Insight.

**Implementation Date**: November 2025
**Version**: 1.0
**Framework**: Built on Strands SDK and Amazon Bedrock

---

## Architecture

### Directory Structure

```
src/optimizer/              # Core GEPA implementation
├── __init__.py            # Module exports
├── gepa_optimizer.py      # Main GEPA engine
├── llm_evaluator.py       # LLM-based evaluation
└── test_loader.py         # Test case management

data/test_cases/           # Test case repository
├── README.md              # Test case documentation
├── coordinator_tests.yaml # Coordinator agent tests
└── planner_tests.yaml     # Planner agent tests

examples/                  # Usage examples
├── README.md              # Examples documentation
├── optimize_coordinator.py
└── validate_new_prompt.py

tutorials/                 # Learning resources
└── gepa_optimization.ipynb
```

---

## Core Components

### 1. GEPAOptimizer (`src/optimizer/gepa_optimizer.py`)

**Purpose**: Main orchestration engine for GEPA optimization workflow.

**Key Classes**:
- `PromptVersion`: Data class storing prompt versions with scores and metadata
- `GEPAOptimizer`: Main optimizer implementing the 4-phase GEPA cycle

**Key Methods**:
```python
async def generate(test_cases: List[Dict]) -> List[Dict]
    # Phase 1: Run tests with current prompt

async def evaluate(results: List[Dict]) -> Dict[str, Any]
    # Phase 2: Assess quality of outputs

async def produce(evaluation: Dict) -> str
    # Phase 3: Create improved prompt

async def assess(new_prompt: str, test_cases: List[Dict]) -> tuple[bool, Dict]
    # Phase 4: Validate improvement

async def optimize(test_cases: List[Dict], max_iterations: int = 5, ...) -> PromptVersion
    # Full GEPA cycle orchestration
```

**Integration Points**:
- Uses `strands_utils.get_agent()` for agent creation
- Uses `strands_utils.process_streaming_response_yield()` for streaming
- Integrates with `skills/system-prompt-writer` guidelines via `improvement_context`

**Configuration**:
```python
optimizer = GEPAOptimizer(
    base_prompt=str,              # Initial prompt to optimize
    evaluation_fn=callable,       # Sync or async evaluator
    agent_type="claude-sonnet-4", # Model for testing
    guidelines=List[str],         # Optimization guidelines
    improvement_context=str,      # Additional context (e.g., from skills/)
    enable_reasoning=bool,        # Extended thinking toggle
    verbose=bool                  # Progress output
)
```

---

### 2. LLMEvaluator (`src/optimizer/llm_evaluator.py`)

**Purpose**: Automatic evaluation using Claude as judge.

**Key Classes**:
- `LLMEvaluator`: Main evaluator using LLM judge
- `SimpleEvaluator`: Rule-based evaluator for structural validation

**Evaluation Criteria** (default weights):
- **Structure** (30%): Format and structural correctness
- **Quality** (40%): Content accuracy, relevance, completeness
- **Guidelines** (30%): Adherence to specified guidelines

**Example Usage**:
```python
evaluator = LLMEvaluator(
    criteria={
        "format": "JSON with 'analysis' field",
        "tone": "Professional and concise",
        "length": "Between 200-500 words"
    },
    guidelines=["Be specific", "Use examples"],
    judge_model="claude-sonnet-4",
    weights={"structure": 0.3, "quality": 0.4, "guidelines": 0.3}
)

result = await evaluator.evaluate({
    "input": "...",
    "output": "...",
    "expected": "..."
})

# Returns: {"score": 0.85, "feedback": "...", "detailed_scores": {...}}
```

**SimpleEvaluator** (for quick structural checks):
```python
evaluator = SimpleEvaluator(
    required_keywords=["analysis", "recommendation"],
    forbidden_keywords=["sorry", "cannot"],
    min_length=100,
    max_length=1000,
    expected_format="json"
)

result = evaluator.evaluate({"input": "...", "output": "..."})
```

---

### 3. TestCaseLoader (`src/optimizer/test_loader.py`)

**Purpose**: Manage test cases from YAML/JSON files.

**Key Classes**:
- `TestCaseLoader`: Load and filter test cases
- `TestCaseBuilder`: Programmatically create test cases

**Test Case Format** (YAML):
```yaml
test_cases:
  - input: "User query or prompt"
    expected: "Expected behavior description"
    metadata:
      category: "test_type"
      priority: "high/medium/low"
      custom_field: "additional context"
```

**Usage**:
```python
# Load test cases
loader = TestCaseLoader()
cases = loader.load("coordinator_tests.yaml")

# Filter by metadata
high_priority = loader.filter_by_metadata(cases, priority="high")
complex_tasks = loader.filter_by_metadata(cases, category="complex_task")

# Build test cases programmatically
builder = TestCaseBuilder()
builder.add("Hello", expected="Should greet warmly")
builder.add("Analyze data", expected="Should plan analysis",
           metadata={"priority": "high"})
builder.save("my_tests.yaml")
```

---

## Usage Patterns

### Pattern 1: Optimize Existing Prompt

```python
from src.optimizer import GEPAOptimizer, LLMEvaluator, TestCaseLoader
from src.prompts.template import apply_prompt_template

# 1. Load current prompt
current_prompt = apply_prompt_template("coordinator", {...})

# 2. Load test cases
loader = TestCaseLoader()
tests = loader.load("coordinator_tests.yaml")

# 3. Define evaluator
async def evaluator(result):
    llm_eval = LLMEvaluator(
        criteria={"handoff": "Correct decision", "tone": "Professional"},
        guidelines=["Simple tasks direct", "Complex tasks handoff"]
    )
    return await llm_eval.evaluate(result)

# 4. Run optimization
optimizer = GEPAOptimizer(
    base_prompt=current_prompt,
    evaluation_fn=evaluator,
    guidelines=["Use XML tags", "Provide examples"]
)

best = await optimizer.optimize(tests, max_iterations=5, target_score=0.9)

# 5. Save optimized prompt
with open("coordinator_optimized.md", "w") as f:
    f.write(best.prompt)
```

### Pattern 2: Validate New Prompt

```python
from src.optimizer import GEPAOptimizer, LLMEvaluator

# 1. Load new prompt
with open("new_agent_prompt.md", "r") as f:
    new_prompt = f.read()

# 2. Create optimizer (no optimization, just validation)
optimizer = GEPAOptimizer(
    base_prompt=new_prompt,
    evaluation_fn=generic_evaluator,
    verbose=True
)

# 3. Run single evaluation
results = await optimizer.generate(test_cases)
evaluation = await optimizer.evaluate(results)

# 4. Check quality
if evaluation['avg_score'] >= 0.8:
    print("✅ Prompt ready for deployment")
else:
    print("⚠️ Needs improvement")
```

### Pattern 3: Custom Evaluator with Domain Logic

```python
async def coordinator_evaluator(result):
    """Custom evaluator checking handoff logic."""
    output = result.get("output", "")
    metadata = result.get("metadata", {})

    # Structural validation
    has_handoff_marker = "handoff_to_planner" in output.lower()
    should_handoff = metadata.get("expected_action") == "handoff_to_planner"

    # LLM quality check
    llm_eval = LLMEvaluator(
        criteria={"decision": "Correct handoff decision", "quality": "Good response"},
        guidelines=["Clear logic", "Professional tone"]
    )
    eval_result = await llm_eval.evaluate(result)

    # Adjust score based on structural correctness
    if has_handoff_marker == should_handoff:
        eval_result["score"] = min(1.0, eval_result["score"] + 0.1)  # Bonus
    else:
        eval_result["score"] = max(0.0, eval_result["score"] - 0.3)  # Penalty
        eval_result["feedback"] = f"[WRONG DECISION] {eval_result['feedback']}"

    return eval_result
```

---

## Integration with Project

### Strands SDK Integration

The optimizer seamlessly integrates with existing Strands SDK utilities:

```python
# Agent creation uses existing infrastructure
agent = strands_utils.get_agent(
    agent_name="gepa_test_agent",
    system_prompts=prompt_to_test,
    agent_type="claude-sonnet-4",
    enable_reasoning=False,
    prompt_cache_info=(False, "default"),
    streaming=True
)

# Streaming response processing
async for event in strands_utils.process_streaming_response_yield(
    agent, user_input, agent_name="gepa_test_agent", source="gepa_optimizer"
):
    if event.get("event_type") == "text_chunk":
        full_text += event.get("data", "")
```

### Skills Integration

The optimizer can automatically incorporate guidelines from `skills/`:

```python
# Load system-prompt-writer guidelines
guideline_path = Path("skills/system-prompt-writer")
if guideline_path.exists():
    guide_files = list(guideline_path.glob("*.md"))
    with open(guide_files[0], 'r') as f:
        improvement_context = f.read()

optimizer = GEPAOptimizer(
    base_prompt=current_prompt,
    improvement_context=improvement_context,  # Injected into Produce phase
    ...
)
```

---

## Best Practices

### Test Case Design

1. **Coverage**: Include edge cases, not just happy paths
2. **Realism**: Use actual user queries from production/logs
3. **Balance**: Mix simple and complex scenarios
4. **Metadata**: Tag tests for filtering and analysis

**Good Example**:
```yaml
test_cases:
  - input: "안녕하세요"
    expected: "Should respond directly with greeting"
    metadata:
      category: "greeting"
      priority: "medium"
      expected_action: "respond_to_user"
```

### Evaluation Criteria

1. **Specificity**: Clear, measurable criteria
2. **Priority**: Weight important aspects more heavily
3. **Domain Knowledge**: Include domain-specific requirements
4. **Consistency**: Use same criteria across iterations

**Good Example**:
```python
criteria = {
    "handoff_logic": "Must correctly decide when to handoff (critical)",
    "response_quality": "Clear, helpful, and appropriate",
    "language_matching": "Must use same language as user",
    "tone": "Friendly but professional"
}
```

### Optimization Settings

1. **Iterations**: Start with 3-5 (more may have diminishing returns)
2. **Target Score**: Set realistic goals (0.85-0.9 typically)
3. **Min Improvement**: Use 0.03-0.05 to avoid tiny changes
4. **Verbose Mode**: Enable during development for debugging

**Good Example**:
```python
best = await optimizer.optimize(
    test_cases=tests,
    max_iterations=5,        # Reasonable limit
    target_score=0.88,       # Achievable goal
    min_improvement=0.04     # Meaningful improvement threshold
)
```

---

## Troubleshooting

### Issue: Low Initial Scores

**Symptoms**: Base prompt scores < 0.5

**Solutions**:
1. Review test cases - are they too strict?
2. Check evaluation criteria - are they realistic?
3. Manually inspect failed outputs
4. Start with SimpleEvaluator to check structural issues

### Issue: No Improvement After Iterations

**Symptoms**: Scores plateau or decrease

**Solutions**:
1. Increase `max_iterations` (try 7-10)
2. Review `improvement_context` - is it clear?
3. Check if target score is too ambitious
4. Manually review LLM judge feedback
5. Try different `guidelines`

### Issue: Inconsistent Evaluation

**Symptoms**: Similar outputs get very different scores

**Solutions**:
1. Make evaluation criteria more specific
2. Add concrete examples to `criteria`
3. Use SimpleEvaluator for structural validation first
4. Increase evaluation samples (run multiple times)

### Issue: Optimized Prompt Too Long

**Symptoms**: Prompt becomes overly verbose

**Solutions**:
1. Add "Be concise" to `guidelines`
2. Add length constraint to `criteria`
3. Manually edit and re-validate
4. Use shorter examples in `improvement_context`

---

## Performance Considerations

### Token Usage

- **Generate Phase**: ~1K-3K tokens per test case (input + output)
- **Evaluate Phase**: ~2K-5K tokens per evaluation (judge prompt + response)
- **Produce Phase**: ~5K-10K tokens (improvement generation)
- **Total per iteration**: ~20K-50K tokens (depends on test count)

**Optimization Tips**:
- Filter test cases to high-priority for initial iterations
- Use SimpleEvaluator for structural checks (no tokens)
- Cache evaluation results when repeating tests
- Use `claude-sonnet-4` (fast) for testing, not `opus`

### Time Estimates

- **Single test execution**: 3-8 seconds
- **Evaluation (LLM judge)**: 5-10 seconds per test
- **Full iteration** (8 tests): 2-5 minutes
- **Complete optimization** (5 iterations): 10-25 minutes

---

## Future Enhancements

### Planned Features

1. **Parallel Test Execution**: Run multiple tests concurrently
2. **A/B Testing**: Compare multiple prompt versions
3. **Regression Detection**: Ensure new versions don't break old tests
4. **Prompt Library**: Reusable prompt components
5. **Batch Optimization**: Optimize multiple agents together
6. **Metrics Dashboard**: Visualize optimization history

### Integration Opportunities

1. **CI/CD Pipeline**: Automated prompt validation on commits
2. **Monitoring**: Track production prompt performance
3. **Human-in-the-Loop**: Manual review before deploying
4. **Fine-tuning Data**: Convert successful examples to training data

---

## References

### Documentation

- **Examples**: `examples/README.md`
- **Test Cases**: `data/test_cases/README.md`
- **Tutorial**: `tutorials/gepa_optimization.ipynb`
- **API Docs**: Docstrings in `src/optimizer/`

### External Resources

- **GEPA Methodology**: Systematic prompt engineering approach
- **LLM-as-Judge**: Using LLMs for evaluation (Anthropic, OpenAI research)
- **Prompt Engineering Guide**: Best practices and patterns
- **Strands SDK**: Agent framework documentation

---

## Contact & Support

For questions or issues related to GEPA optimizer:

1. **GitHub Issues**: Report bugs or feature requests
2. **Documentation**: Check `examples/README.md` and tutorial
3. **Contributors**: See main README for contact information

---

**Last Updated**: November 2025
**Maintainers**: AWS Korea SA Team
**License**: MIT
