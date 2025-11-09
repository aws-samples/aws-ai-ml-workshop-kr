"""
GEPA Optimizer Module

A systematic framework for prompt optimization using the GEPA methodology:
- Generate: Run tests with current prompt
- Evaluate: Assess quality using LLM judge
- Produce: Create improved version based on feedback
- Assess: Validate improvement and iterate

This package is designed to work with Strands SDK and Amazon Bedrock,
but can be adapted to other LLM frameworks.

Usage:
    from gepa import GEPAOptimizer, LLMEvaluator, TestCaseLoader

    # Load test cases
    loader = TestCaseLoader()
    tests = loader.load("test_cases/sample_tests.yaml")

    # Create optimizer
    optimizer = GEPAOptimizer(
        base_prompt="Your prompt here",
        evaluation_fn=my_evaluator,
        verbose=True
    )

    # Run optimization
    best = await optimizer.optimize(tests, max_iterations=5)
"""

from .optimizer import GEPAOptimizer, PromptVersion
from .evaluator import LLMEvaluator, SimpleEvaluator
from .test_loader import TestCaseLoader, TestCaseBuilder, create_sample_test_file

__version__ = "1.0.0"

__all__ = [
    'GEPAOptimizer',
    'PromptVersion',
    'LLMEvaluator',
    'SimpleEvaluator',
    'TestCaseLoader',
    'TestCaseBuilder',
    'create_sample_test_file',
]
