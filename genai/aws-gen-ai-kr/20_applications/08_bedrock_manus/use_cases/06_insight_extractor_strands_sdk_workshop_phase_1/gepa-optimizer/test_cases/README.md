# Test Cases Directory

This directory contains test cases for prompt optimization using the GEPA framework.

## File Format

Test cases are stored in YAML or JSON format with the following structure:

```yaml
test_cases:
  - input: "User query or prompt to test"
    expected: "Expected behavior or output description"
    metadata:
      category: "test category"
      priority: "high/medium/low"
      custom_field: "any additional context"
```

## Available Test Files

- **coordinator_tests.yaml**: Test cases for Coordinator agent
  - Tests handoff logic (simple queries vs complex tasks)
  - 8 test cases covering greetings, simple queries, and complex multi-step tasks

- **planner_tests.yaml**: Test cases for Planner agent
  - Tests plan creation for various task complexities
  - 5 test cases from simple to complex ML pipelines

## Creating New Test Files

### Using TestCaseBuilder

```python
from src.optimizer.test_loader import TestCaseBuilder

builder = TestCaseBuilder()
builder.add(
    "Your test input",
    expected="Expected behavior",
    metadata={"category": "test", "priority": "high"}
)
builder.save("my_tests.yaml")
```

### Using create_sample_test_file

```python
from src.optimizer.test_loader import create_sample_test_file

create_sample_test_file("coordinator")  # Creates coordinator_tests.yaml
create_sample_test_file("planner")      # Creates planner_tests.yaml
```

## Loading Test Cases

```python
from src.optimizer.test_loader import TestCaseLoader

loader = TestCaseLoader()

# Load all test cases
cases = loader.load("coordinator_tests.yaml")

# Filter by metadata
high_priority = loader.filter_by_metadata(cases, priority="high")
complex_tasks = loader.filter_by_metadata(cases, category="complex_task")
```

## Best Practices

1. **Input**: Write realistic user queries that match actual use cases
2. **Expected**: Describe the desired behavior clearly and specifically
3. **Metadata**: Add useful tags for filtering and analysis
   - `category`: Type of test (greeting, complex_task, simple, etc.)
   - `priority`: Importance level (high, medium, low)
   - `expected_action`: Specific action expected (handoff_to_planner, respond_to_user)
4. **Coverage**: Include both positive and edge cases
5. **Clarity**: Make expectations measurable and verifiable
