"""
Basic Usage Example

This is a minimal example showing how to use GEPA optimizer
in any project with Strands SDK.
"""

import asyncio
import sys
from pathlib import Path

# Add gepa-optimizer to path
gepa_root = Path(__file__).parent.parent
project_root = gepa_root.parent
sys.path.insert(0, str(gepa_root))
sys.path.insert(0, str(project_root))

from gepa import GEPAOptimizer, LLMEvaluator, TestCaseLoader


async def simple_evaluator(result):
    """Simple evaluator using LLM judge."""
    evaluator = LLMEvaluator(
        criteria={
            "clarity": "Response is clear and easy to understand",
            "helpfulness": "Response is helpful and appropriate",
            "conciseness": "Response is concise without unnecessary verbosity"
        },
        guidelines=[
            "Be friendly and professional",
            "Stay on topic",
            "Provide accurate information"
        ]
    )
    return await evaluator.evaluate(result)


async def main():
    print("=" * 60)
    print("GEPA Optimizer - Basic Usage Example")
    print("=" * 60)

    # 1. Define a simple prompt to optimize
    base_prompt = """
You are a helpful AI assistant. Answer user questions clearly and concisely.
Be friendly but professional in your responses.
"""

    print(f"\nBase prompt ({len(base_prompt)} chars):")
    print("-" * 60)
    print(base_prompt)
    print("-" * 60)

    # 2. Load test cases
    loader = TestCaseLoader()
    test_cases = loader.load("sample_tests.yaml")  # Uses default gepa-optimizer/test_cases/
    print(f"\nLoaded {len(test_cases)} test cases")

    # 3. Create optimizer
    optimizer = GEPAOptimizer(
        base_prompt=base_prompt,
        evaluation_fn=simple_evaluator,
        agent_type="claude-sonnet-4",
        guidelines=[
            "Use clear, simple language",
            "Be concise but complete",
            "Maintain friendly tone"
        ],
        verbose=True
    )

    # 4. Run optimization (just 2 iterations for demo)
    print("\nStarting optimization...\n")
    best_version = await optimizer.optimize(
        test_cases=test_cases,
        max_iterations=2,
        target_score=0.9,
        min_improvement=0.05
    )

    # 5. Show results
    if best_version:
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Best version: {best_version.version}")
        print(f"Final score: {best_version.score:.3f}")
        print(f"Pass rate: {best_version.metadata.get('pass_rate', 0):.1%}")

        print(f"\nOptimized prompt:")
        print("-" * 60)
        print(best_version.prompt[:300] + "..." if len(best_version.prompt) > 300 else best_version.prompt)
        print("-" * 60)

        # Auto-save to output directory
        output_dir = gepa_root / "output"
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / "basic_usage_optimized.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(best_version.prompt)
        print(f"\n✓ Saved optimized prompt to: {output_file}")

        # Save optimization history
        history_file = output_dir / "basic_usage_history.json"
        optimizer.save_history(str(history_file))
    else:
        print("\n⚠ No optimization performed")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
