"""
Test Case Loader - Load and manage test cases from YAML/JSON files

Supports loading test cases for prompt optimization from structured files.
"""

import yaml
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class TestCaseLoader:
    """
    Loader for test cases from YAML or JSON files.

    Test case format:
    ```yaml
    test_cases:
      - input: "User query or prompt"
        expected: "Expected behavior or output"
        metadata:
          category: "type of test"
          priority: "high/medium/low"

      - input: "Another query"
        expected: "Expected behavior"
    ```

    Example:
        >>> loader = TestCaseLoader()
        >>> cases = loader.load("data/test_cases/coordinator_tests.yaml")
        >>> print(len(cases))
        5
    """

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize TestCaseLoader.

        Args:
            base_path: Base directory for test case files (default: gepa-optimizer/test_cases/)
        """
        if base_path:
            self.base_path = Path(base_path)
        else:
            # Default to gepa-optimizer/test_cases/ (same directory as gepa module)
            self.base_path = Path(__file__).parent.parent / "test_cases"

    def load(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load test cases from a file.

        Args:
            filename: Filename (with or without path). If no path provided,
                     uses base_path. Supports .yaml, .yml, and .json extensions.

        Returns:
            List of test case dictionaries

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        filepath = Path(filename)

        # If filename has no directory, use base_path
        if not filepath.parent or filepath.parent == Path("."):
            filepath = self.base_path / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Test case file not found: {filepath}")

        # Load based on extension
        if filepath.suffix in ['.yaml', '.yml']:
            return self._load_yaml(filepath)
        elif filepath.suffix == '.json':
            return self._load_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}. Use .yaml, .yml, or .json")

    def _load_yaml(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load test cases from YAML file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return self._validate_and_extract(data, filepath)

    def _load_json(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load test cases from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return self._validate_and_extract(data, filepath)

    def _validate_and_extract(self, data: Any, filepath: Path) -> List[Dict[str, Any]]:
        """Validate and extract test cases from loaded data."""
        if not isinstance(data, dict):
            raise ValueError(f"Invalid format in {filepath}: Expected dict at root level")

        # Support multiple formats
        if "test_cases" in data:
            test_cases = data["test_cases"]
        elif "tests" in data:
            test_cases = data["tests"]
        elif "cases" in data:
            test_cases = data["cases"]
        else:
            raise ValueError(
                f"Invalid format in {filepath}: Expected 'test_cases', 'tests', or 'cases' key"
            )

        if not isinstance(test_cases, list):
            raise ValueError(f"Invalid format in {filepath}: test_cases must be a list")

        # Validate each test case
        for i, case in enumerate(test_cases):
            if not isinstance(case, dict):
                raise ValueError(f"Invalid test case #{i+1} in {filepath}: Expected dict")
            if "input" not in case:
                raise ValueError(f"Invalid test case #{i+1} in {filepath}: Missing 'input' field")

        return test_cases

    def save(self, test_cases: List[Dict[str, Any]], filename: str, format: str = "yaml"):
        """
        Save test cases to a file.

        Args:
            test_cases: List of test case dictionaries
            filename: Filename to save to
            format: File format ('yaml' or 'json')
        """
        filepath = Path(filename)

        # If filename has no directory, use base_path
        if not filepath.parent or filepath.parent == Path("."):
            filepath = self.base_path / filename

        # Ensure correct extension
        if format == "yaml" and filepath.suffix not in ['.yaml', '.yml']:
            filepath = filepath.with_suffix('.yaml')
        elif format == "json" and filepath.suffix != '.json':
            filepath = filepath.with_suffix('.json')

        data = {"test_cases": test_cases}

        if format == "yaml":
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        elif format == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")

        print(f"💾 Saved {len(test_cases)} test cases to {filepath}")

    def filter_by_metadata(
        self,
        test_cases: List[Dict[str, Any]],
        **filters
    ) -> List[Dict[str, Any]]:
        """
        Filter test cases by metadata fields.

        Example:
            >>> cases = loader.load("tests.yaml")
            >>> high_priority = loader.filter_by_metadata(cases, priority="high")
            >>> regression = loader.filter_by_metadata(cases, category="regression")

        Args:
            test_cases: List of test cases
            **filters: Metadata key-value pairs to filter by

        Returns:
            Filtered list of test cases
        """
        if not filters:
            return test_cases

        filtered = []
        for case in test_cases:
            metadata = case.get("metadata", {})
            if all(metadata.get(k) == v for k, v in filters.items()):
                filtered.append(case)

        return filtered

    def list_files(self) -> List[str]:
        """List all test case files in the base directory."""
        files = []
        for ext in ['.yaml', '.yml', '.json']:
            files.extend([f.name for f in self.base_path.glob(f'*{ext}')])
        return sorted(files)


class TestCaseBuilder:
    """
    Builder for creating test cases programmatically.

    Example:
        >>> builder = TestCaseBuilder()
        >>> builder.add("Simple greeting", expected="Should respond politely")
        >>> builder.add("Complex analysis", expected="Should create detailed report",
        ...            metadata={"category": "analysis", "priority": "high"})
        >>> cases = builder.build()
        >>> builder.save("my_tests.yaml")
    """

    def __init__(self):
        """Initialize TestCaseBuilder."""
        self.test_cases: List[Dict[str, Any]] = []

    def add(
        self,
        input_text: str,
        expected: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'TestCaseBuilder':
        """
        Add a test case.

        Args:
            input_text: Input text for the test
            expected: Expected behavior or output (optional)
            metadata: Additional metadata (optional)

        Returns:
            Self for method chaining
        """
        test_case = {"input": input_text}

        if expected:
            test_case["expected"] = expected

        if metadata:
            test_case["metadata"] = metadata

        self.test_cases.append(test_case)
        return self

    def build(self) -> List[Dict[str, Any]]:
        """Build and return the test cases list."""
        return self.test_cases

    def save(self, filename: str, format: str = "yaml"):
        """
        Save test cases to file.

        Args:
            filename: Filename to save to
            format: File format ('yaml' or 'json')
        """
        loader = TestCaseLoader()
        loader.save(self.test_cases, filename, format)

    def clear(self):
        """Clear all test cases."""
        self.test_cases = []


def create_sample_test_file(agent_name: str, output_path: Optional[str] = None):
    """
    Create a sample test case file for a given agent.

    Args:
        agent_name: Name of agent (coordinator, planner, etc.)
        output_path: Where to save the file (default: data/test_cases/{agent_name}_tests.yaml)
    """
    builder = TestCaseBuilder()

    if agent_name == "coordinator":
        builder.add(
            "안녕하세요",
            expected="Should respond directly without handoff to planner",
            metadata={"category": "greeting", "priority": "medium"}
        )
        builder.add(
            "서울시 아파트 매매 데이터를 분석해서 보고서를 작성해주세요",
            expected="Should handoff to planner for complex task",
            metadata={"category": "complex_task", "priority": "high"}
        )
        builder.add(
            "고마워요",
            expected="Should respond directly with acknowledgment",
            metadata={"category": "simple", "priority": "low"}
        )

    elif agent_name == "planner":
        builder.add(
            "Python 데이터 분석 보고서 작성",
            expected="Should create detailed plan with data loading, analysis, visualization, and reporting steps",
            metadata={"category": "data_analysis", "priority": "high"}
        )
        builder.add(
            "간단한 계산",
            expected="Should create concise plan for simple task",
            metadata={"category": "simple", "priority": "medium"}
        )

    else:
        # Generic template
        builder.add(
            "Sample input 1",
            expected="Expected behavior 1",
            metadata={"category": "sample", "priority": "medium"}
        )
        builder.add(
            "Sample input 2",
            expected="Expected behavior 2",
            metadata={"category": "sample", "priority": "low"}
        )

    if output_path is None:
        output_path = f"{agent_name}_tests.yaml"

    builder.save(output_path)
    print(f"✅ Created sample test file: {output_path}")
