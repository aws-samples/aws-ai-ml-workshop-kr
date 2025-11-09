"""
Example: Validate New Prompt using GEPA

This script demonstrates how to use GEPA to validate a newly created prompt
without full optimization - just running evaluation to ensure it meets quality standards.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root and gepa-optimizer to path
project_root = Path(__file__).parent.parent.parent  # Go to project root
gepa_root = Path(__file__).parent.parent  # gepa-optimizer directory
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(gepa_root))

from gepa import GEPAOptimizer, LLMEvaluator, TestCaseLoader, TestCaseBuilder


async def generic_evaluator(result: Dict[str, Any]) -> Dict[str, Any]:
    """Generic evaluator for any prompt."""
    criteria = {
        "clarity": "Response should be clear and easy to understand",
        "completeness": "Response should address the user's request fully",
        "correctness": "Response should be accurate and appropriate",
        "tone": "Response should match expected tone and style"
    }

    guidelines = [
        "Responses should be helpful and actionable",
        "Avoid unnecessary verbosity",
        "Match the user's language",
        "Maintain consistency with defined role"
    ]

    llm_eval = LLMEvaluator(
        criteria=criteria,
        guidelines=guidelines,
        judge_model="claude-sonnet-4"
    )

    return await llm_eval.evaluate(result)


async def validate_prompt_interactive():
    """Interactive prompt validation."""
    print("=" * 70)
    print("GEPA Validator - New Prompt Validation")
    print("=" * 70)

    # Get prompt from user
    print("\n[1] Enter your prompt to validate")
    print("    (Type or paste your prompt, then press Ctrl+D or Ctrl+Z when done)")
    print("-" * 70)

    prompt_lines = []
    try:
        while True:
            line = input()
            prompt_lines.append(line)
    except EOFError:
        pass

    new_prompt = "\n".join(prompt_lines)

    if not new_prompt.strip():
        print("\n❌ No prompt provided. Exiting.")
        return

    print(f"\n   ✓ Prompt loaded ({len(new_prompt)} characters)")

    # Get test cases
    print("\n[2] Test cases")
    print("    Choose an option:")
    print("    1. Use existing test file")
    print("    2. Create test cases interactively")

    choice = input("    Enter choice (1 or 2): ").strip()

    if choice == "1":
        # Load existing test file
        loader = TestCaseLoader()
        available_files = loader.list_files()

        print("\n    Available test files:")
        for i, filename in enumerate(available_files, 1):
            print(f"    {i}. {filename}")

        file_idx = int(input(f"    Select file (1-{len(available_files)}): ")) - 1
        test_cases = loader.load(available_files[file_idx])
        print(f"   ✓ Loaded {len(test_cases)} test cases from {available_files[file_idx]}")

    else:
        # Create test cases interactively
        builder = TestCaseBuilder()
        print("\n    Enter test cases (press Ctrl+D when done)")

        while True:
            try:
                print("\n    Test input: ", end="")
                test_input = input()
                print("    Expected (optional): ", end="")
                expected = input()

                builder.add(
                    test_input,
                    expected=expected if expected.strip() else None
                )
                print("    ✓ Added")

            except EOFError:
                break

        test_cases = builder.build()
        print(f"\n   ✓ Created {len(test_cases)} test cases")

    if not test_cases:
        print("\n❌ No test cases provided. Exiting.")
        return

    # Run validation (single iteration, no optimization)
    print("\n[3] Running validation...")
    print("=" * 70)

    optimizer = GEPAOptimizer(
        base_prompt=new_prompt,
        evaluation_fn=generic_evaluator,
        agent_type="claude-sonnet-4",
        enable_reasoning=False,
        verbose=True
    )

    # Just run generate + evaluate (no optimization)
    results = await optimizer.generate(test_cases)
    evaluation = await optimizer.evaluate(results)

    # Display results
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"Average Score: {evaluation['avg_score']:.3f}")
    print(f"Pass Rate: {evaluation['pass_rate']:.1%} ({evaluation['passed']}/{evaluation['total']})")
    print(f"\nScore Breakdown:")
    for i, score in enumerate(evaluation['scores'], 1):
        status = "✓" if score >= 0.8 else "✗"
        print(f"  {status} Test {i}: {score:.2f}")
        if evaluation['feedback'][i-1]:
            print(f"     → {evaluation['feedback'][i-1]}")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    if evaluation['avg_score'] >= 0.9:
        print("✅ Excellent! Your prompt is ready to use.")
    elif evaluation['avg_score'] >= 0.8:
        print("👍 Good! Minor improvements recommended:")
        print("   - Review failed test cases")
        print("   - Consider running GEPA optimization for refinement")
    elif evaluation['avg_score'] >= 0.6:
        print("⚠️ Needs improvement:")
        print("   - Review feedback for failed tests")
        print("   - Run GEPA optimization to improve the prompt")
        print("   - Consider adding more specific instructions or examples")
    else:
        print("❌ Significant issues detected:")
        print("   - Prompt may need restructuring")
        print("   - Run GEPA optimization with more iterations")
        print("   - Review test cases to ensure they're appropriate")

    # Auto-save validation report
    gepa_root = project_root / "gepa-optimizer"
    output_dir = gepa_root / "output"
    output_dir.mkdir(exist_ok=True)

    report_path = output_dir / "validation_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
            f.write("PROMPT VALIDATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Score: {evaluation['avg_score']:.3f}\n")
            f.write(f"Pass Rate: {evaluation['pass_rate']:.1%}\n\n")
            f.write("TESTED PROMPT:\n")
            f.write("-" * 70 + "\n")
            f.write(new_prompt + "\n")
            f.write("-" * 70 + "\n\n")
            f.write("TEST RESULTS:\n")
            for i, (result, score) in enumerate(zip(results, evaluation['scores']), 1):
                f.write(f"\nTest {i}: {score:.2f}\n")
                f.write(f"Input: {result['input']}\n")
                f.write(f"Output: {result['output'][:200]}...\n")
                if evaluation['feedback'][i-1]:
                    f.write(f"Feedback: {evaluation['feedback'][i-1]}\n")

    print(f"\n✓ Validation report saved to: {report_path}")
    print("\n" + "=" * 70)


async def validate_prompt_from_file(prompt_file: str, test_file: str):
    """Validate a prompt from a file."""
    print("=" * 70)
    print("GEPA Validator - File-based Validation")
    print("=" * 70)

    # Load prompt
    print(f"\n[1] Loading prompt from {prompt_file}...")
    prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        prompt_path = project_root / prompt_file

    with open(prompt_path, 'r', encoding='utf-8') as f:
        new_prompt = f.read()

    print(f"   ✓ Loaded prompt ({len(new_prompt)} characters)")

    # Load test cases
    print(f"\n[2] Loading test cases from {test_file}...")
    loader = TestCaseLoader()
    test_cases = loader.load(test_file)
    print(f"   ✓ Loaded {len(test_cases)} test cases")

    # Run validation
    print("\n[3] Running validation...")
    print("=" * 70)

    optimizer = GEPAOptimizer(
        base_prompt=new_prompt,
        evaluation_fn=generic_evaluator,
        agent_type="claude-sonnet-4",
        verbose=True
    )

    results = await optimizer.generate(test_cases)
    evaluation = await optimizer.evaluate(results)

    # Display results
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"Average Score: {evaluation['avg_score']:.3f}")
    print(f"Pass Rate: {evaluation['pass_rate']:.1%}")

    return evaluation['avg_score'] >= 0.8


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate a new prompt using GEPA")
    parser.add_argument("--prompt", type=str, help="Path to prompt file")
    parser.add_argument("--tests", type=str, help="Path to test cases file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    if args.interactive or (not args.prompt and not args.tests):
        await validate_prompt_interactive()
    elif args.prompt and args.tests:
        success = await validate_prompt_from_file(args.prompt, args.tests)
        sys.exit(0 if success else 1)
    else:
        print("Error: Provide both --prompt and --tests, or use --interactive")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
