# Integration Guide

This guide shows how to integrate GEPA optimizer into your project.

## Prerequisites

Your project must have:

1. **Strands SDK** installed and configured
2. **src/utils/strands_sdk_utils.py** module with:
   - `get_agent(agent_name, system_prompts, agent_type, enable_reasoning, prompt_cache_info, streaming, tools=None)` function
   - `process_streaming_response_yield(agent, message, agent_name, source)` async generator

## Integration Steps

### Step 1: Copy GEPA Optimizer

```bash
# Copy to your project root
cp -r /path/to/gepa-optimizer /path/to/your/project/
```

Your project structure will look like:

```
your-project/
├── src/
│   └── utils/
│       └── strands_sdk_utils.py  # Required
├── gepa-optimizer/               # Copied
│   ├── gepa/
│   ├── examples/
│   └── test_cases/
└── ... your other files
```

### Step 2: Install Dependencies

```bash
cd gepa-optimizer
pip install -r requirements.txt
```

### Step 3: Use in Your Code

```python
import sys
from pathlib import Path

# Add gepa-optimizer to path
project_root = Path(__file__).parent  # Adjust to your project root
gepa_path = project_root / "gepa-optimizer"
sys.path.insert(0, str(gepa_path))

# Now import
from gepa import GEPAOptimizer, LLMEvaluator, TestCaseLoader
```

## Example: Optimize a Custom Agent

```python
import asyncio
import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
gepa_path = project_root / "gepa-optimizer"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(gepa_path))

from gepa import GEPAOptimizer, LLMEvaluator, TestCaseLoader


async def custom_evaluator(result):
    """Evaluator for my custom agent."""
    output = result.get("output", "")
    expected = result.get("expected", "")

    # Custom validation logic
    has_required_format = "analysis:" in output.lower()

    # Use LLM judge for quality
    llm_eval = LLMEvaluator(
        criteria={
            "format": "Must include 'Analysis:' section",
            "quality": "Analysis must be accurate and complete",
            "tone": "Professional and objective"
        },
        guidelines=[
            "Use structured format",
            "Provide evidence for claims",
            "Be concise"
        ]
    )

    eval_result = await llm_eval.evaluate(result)

    # Adjust score based on format
    if not has_required_format:
        eval_result["score"] = max(0, eval_result["score"] - 0.3)
        eval_result["feedback"] = f"[MISSING FORMAT] {eval_result['feedback']}"

    return eval_result


async def main():
    # Load your current prompt
    with open("prompts/my_agent.md", "r") as f:
        current_prompt = f.read()

    # Create test cases
    loader = TestCaseLoader()
    tests = loader.load("gepa-optimizer/test_cases/my_agent_tests.yaml")

    # Run optimization
    optimizer = GEPAOptimizer(
        base_prompt=current_prompt,
        evaluation_fn=custom_evaluator,
        agent_type="claude-sonnet-4",
        guidelines=[
            "Use clear section headers",
            "Provide examples",
            "Define success criteria"
        ],
        verbose=True
    )

    best = await optimizer.optimize(
        test_cases=tests,
        max_iterations=5,
        target_score=0.9
    )

    # Save optimized prompt
    if best:
        with open("prompts/my_agent_optimized.md", "w") as f:
            f.write(best.prompt)
        print(f"✓ Saved optimized prompt (score: {best.score:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
```

## Creating Test Cases for Your Agent

### Example: API Agent Test Cases

```yaml
# gepa-optimizer/test_cases/api_agent_tests.yaml
test_cases:
  - input: "Create a REST API endpoint for user registration"
    expected: "Should generate complete endpoint with validation, error handling, and documentation"
    metadata:
      category: "endpoint_creation"
      priority: "high"
      expected_elements:
        - "request validation"
        - "error responses"
        - "success response"

  - input: "Add rate limiting to the API"
    expected: "Should include rate limit implementation with configurable limits"
    metadata:
      category: "feature_addition"
      priority: "high"

  - input: "Fix the login endpoint"
    expected: "Should ask for details about what needs fixing"
    metadata:
      category: "maintenance"
      priority: "medium"
```

## Working with Project-Specific Prompts

If your prompts use templates:

```python
from src.prompts.template import apply_prompt_template
from datetime import datetime

# Load and render template
current_prompt = apply_prompt_template(
    prompt_name="my_agent",
    prompt_context={"CURRENT_TIME": datetime.now().isoformat()}
)

# Use in optimizer
optimizer = GEPAOptimizer(
    base_prompt=current_prompt,
    evaluation_fn=my_evaluator,
    ...
)
```

## Adapting to Different LLM Frameworks

If you're NOT using Strands SDK, you need to modify:

### Option 1: Create Wrapper Functions

```python
# In your project
# my_project/utils/llm_wrapper.py

async def get_agent(agent_name, system_prompts, agent_type, **kwargs):
    """Wrapper for your LLM framework."""
    # Your implementation here
    # Return an agent object
    pass

async def process_streaming_response_yield(agent, message, **kwargs):
    """Wrapper for streaming responses."""
    # Your implementation here
    # Yield events in expected format
    pass
```

Then import as:

```python
from my_project.utils import llm_wrapper as strands_utils
```

### Option 2: Modify GEPA Source

Edit `gepa-optimizer/gepa/optimizer.py` and `gepa-optimizer/gepa/evaluator.py`:

Replace:
```python
from src.utils import strands_utils
```

With:
```python
from my_framework import my_llm_utils as strands_utils
```

## Best Practices

### 1. Separate Test Cases by Agent Type

```
gepa-optimizer/test_cases/
├── coordinator_tests.yaml
├── planner_tests.yaml
├── api_agent_tests.yaml
└── custom_agent_tests.yaml
```

### 2. Version Control Integration

Add to `.gitignore`:
```
# GEPA optimization artifacts
gepa-optimizer/optimized_prompts/
gepa-optimizer/*.json
*_optimized.md
```

Keep in version control:
```
gepa-optimizer/gepa/          # Core code
gepa-optimizer/test_cases/    # Test definitions
gepa-optimizer/examples/      # Usage examples
gepa-optimizer/docs/          # Documentation
```

### 3. CI/CD Integration

```yaml
# .github/workflows/validate-prompts.yml
name: Validate Prompts

on: [pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r gepa-optimizer/requirements.txt
      - name: Validate prompts
        run: |
          python gepa-optimizer/examples/validate_all_prompts.py
```

### 4. Prompt Version Management

```python
# Save with version info
version_info = {
    "version": "2.1",
    "date": datetime.now().isoformat(),
    "score": best.score,
    "iterations": len(optimizer.versions)
}

with open("prompts/my_agent_v2.1.md", "w") as f:
    f.write(f"<!-- Version: {json.dumps(version_info)} -->\n")
    f.write(best.prompt)
```

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'gepa'`

**Solution:**
```python
# Make sure path is correct
import sys
from pathlib import Path

# If running from project root:
sys.path.insert(0, "gepa-optimizer")

# If running from subdirectory:
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "gepa-optimizer"))
```

### Agent Creation Errors

**Problem:** `strands_utils.get_agent()` not found

**Solution:**
- Ensure `src/utils/strands_sdk_utils.py` exists
- Run from project root (where `src/` is visible)
- Check Strands SDK is installed

### Path Issues in Examples

**Problem:** Examples can't find project files

**Solution:**
The examples assume this structure:
```
project-root/
├── src/
├── gepa-optimizer/
│   └── examples/
│       └── optimize_coordinator.py  # Runs from here
```

If different, adjust paths in examples:
```python
# In example script
project_root = Path(__file__).parent.parent.parent  # Adjust as needed
```

## Advanced: Custom Adapter Pattern

For maximum flexibility, create an adapter:

```python
# my_project/gepa_adapter.py

from abc import ABC, abstractmethod

class AgentAdapter(ABC):
    @abstractmethod
    async def run_agent(self, prompt: str, user_input: str) -> str:
        pass

class MyProjectAdapter(AgentAdapter):
    async def run_agent(self, prompt: str, user_input: str) -> str:
        # Your implementation using your LLM framework
        agent = my_framework.create_agent(prompt)
        response = await agent.run(user_input)
        return response.text

# Use in GEPA
optimizer = GEPAOptimizer(
    base_prompt=prompt,
    evaluation_fn=evaluator,
    agent_adapter=MyProjectAdapter()  # Pass your adapter
)
```

Then modify `gepa/optimizer.py` to accept and use the adapter.

## Support

For integration issues:

1. Check `docs/QUICK_START.md` for basics
2. Review `examples/` for working code
3. See `docs/ARCHITECTURE.md` for implementation details
4. Check your `src/utils/strands_sdk_utils.py` implementation

## Summary

**Minimum steps to integrate:**

1. Copy `gepa-optimizer/` to your project
2. Install `pip install -r gepa-optimizer/requirements.txt`
3. Ensure `src/utils/strands_sdk_utils.py` exists with required functions
4. Import and use: `from gepa import GEPAOptimizer, LLMEvaluator, TestCaseLoader`

That's it! The framework is designed to be drop-in ready for Strands SDK projects.
