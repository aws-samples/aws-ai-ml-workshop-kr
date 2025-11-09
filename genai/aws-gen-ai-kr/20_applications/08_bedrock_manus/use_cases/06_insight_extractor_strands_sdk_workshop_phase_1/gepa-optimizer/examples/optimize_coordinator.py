"""
Example: Optimize Coordinator Prompt using GEPA

This script demonstrates how to use the GEPA optimizer to improve
the Coordinator agent's prompt based on test cases.
"""

import asyncio
import sys
from pathlib import Path

# Add project root and gepa-optimizer to path
project_root = Path(__file__).parent.parent.parent  # Go to project root
gepa_root = Path(__file__).parent.parent  # gepa-optimizer directory
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(gepa_root))

from gepa import GEPAOptimizer, LLMEvaluator, TestCaseLoader
from src.prompts.template import apply_prompt_template
from datetime import datetime


async def coordinator_evaluator(result: dict) -> dict:
    """
    Custom evaluator for Coordinator agent outputs.

    Checks:
    1. Correct handoff decision (handoff vs direct response)
    2. Response quality and appropriateness
    3. Language matching and tone
    """
    # Extract data
    output = result.get("output", "")
    expected = result.get("expected", "")
    metadata = result.get("metadata", {})

    # Prepare criteria for LLM evaluator
    criteria = {
        "handoff_decision": "Should correctly decide between 'handoff_to_planner' and direct response",
        "response_quality": "Response should be appropriate, clear, and helpful",
        "language": "Should match user's language (Korean/English)",
        "tone": "Should be friendly but professional"
    }

    # Add specific handoff expectation if available
    expected_action = metadata.get("expected_action")
    if expected_action:
        criteria["expected_action"] = f"Must use '{expected_action}' approach"

    guidelines = [
        "Coordinator should NOT attempt to solve complex tasks",
        "Simple greetings should be handled directly",
        "Complex multi-step tasks should be handed off with 'handoff_to_planner:' marker",
        "Responses should be warm and welcoming",
        "Must preserve user's language"
    ]

    # Use LLM evaluator
    llm_eval = LLMEvaluator(
        criteria=criteria,
        guidelines=guidelines,
        judge_model="claude-sonnet-4"
    )

    eval_result = await llm_eval.evaluate(result)

    # Add structural validation bonus/penalty
    has_handoff_marker = "handoff_to_planner" in output.lower()
    should_handoff = expected_action == "handoff_to_planner" if expected_action else "handoff" in expected.lower()

    # Adjust score based on structural correctness
    if has_handoff_marker == should_handoff:
        eval_result["score"] = min(1.0, eval_result["score"] + 0.1)  # Bonus for correct decision
    else:
        eval_result["score"] = max(0.0, eval_result["score"] - 0.3)  # Penalty for wrong decision
        eval_result["feedback"] = f"[WRONG DECISION: {'has' if has_handoff_marker else 'missing'} handoff marker] " + eval_result["feedback"]

    return eval_result


async def main():
    print("=" * 70)
    print("GEPA Optimizer - Coordinator Prompt Optimization")
    print("=" * 70)

    # 1. Load current coordinator prompt
    print("\n[1] Loading current coordinator prompt...")
    current_prompt = apply_prompt_template(
        prompt_name="coordinator",
        prompt_context={"CURRENT_TIME": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    )
    print(f"   ✓ Loaded prompt ({len(current_prompt)} characters)")

    # 2. Load test cases
    print("\n[2] Loading test cases...")
    loader = TestCaseLoader()
    test_cases = loader.load("coordinator_tests.yaml")
    print(f"   ✓ Loaded {len(test_cases)} test cases")

    # Optional: Filter to high-priority cases for faster iteration
    # test_cases = loader.filter_by_metadata(test_cases, priority="high")
    # print(f"   → Filtered to {len(test_cases)} high-priority cases")

    # 3. Load system-prompt-writer guidelines (optional)
    print("\n[3] Loading guidelines...")
    guidelines = [
        "Follow skills/system-prompt-writer best practices",
        "Use clear XML tags for structure",
        "Provide concrete examples",
        "Be specific about success criteria",
        "Define clear constraints"
    ]

    # Try to load actual guideline content
    guideline_path = project_root / "skills" / "system-prompt-writer"
    improvement_context = None
    if guideline_path.exists():
        try:
            # Read guideline files if they exist
            guide_files = list(guideline_path.glob("*.md"))
            if guide_files:
                with open(guide_files[0], 'r', encoding='utf-8') as f:
                    improvement_context = f.read()
                print(f"   ✓ Loaded guidelines from {guide_files[0].name}")
        except Exception as e:
            print(f"   ⚠ Could not load guideline files: {e}")

    if not improvement_context:
        improvement_context = """
**System Prompt Writing Guidelines:**
- Use clear structural tags (e.g., <role>, <instructions>, <constraints>)
- Provide concrete examples for each scenario
- Define measurable success criteria
- Be specific about what to do AND what not to do
- Keep language clear and direct
"""
        print("   ✓ Using default guidelines")

    # 4. Create GEPA optimizer
    print("\n[4] Creating GEPA optimizer...")
    optimizer = GEPAOptimizer(
        base_prompt=current_prompt,
        evaluation_fn=coordinator_evaluator,
        agent_type="claude-sonnet-4",
        guidelines=guidelines,
        improvement_context=improvement_context,
        enable_reasoning=False,  # Coordinator doesn't use reasoning
        verbose=True
    )
    print("   ✓ Optimizer ready")

    # 5. Run optimization
    print("\n[5] Starting GEPA optimization...")
    print("=" * 70)

    best_version = await optimizer.optimize(
        test_cases=test_cases,
        max_iterations=3,  # Limit iterations for this example
        target_score=0.9,
        min_improvement=0.05
    )

    # 6. Save results
    print("\n[6] Saving results...")

    if best_version:
        # Create output directory
        output_dir = gepa_root / "output"
        output_dir.mkdir(exist_ok=True)

        # Save optimized prompt
        output_path = output_dir / "coordinator_optimized.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(best_version.prompt)
        print(f"   ✓ Saved optimized prompt to: {output_path}")

        # Save optimization history
        history_path = output_dir / "coordinator_optimization_history.json"
        optimizer.save_history(str(history_path))

        # Print summary
        print("\n" + "=" * 70)
        print("OPTIMIZATION SUMMARY")
        print("=" * 70)
        print(f"Versions created: {len(optimizer.versions)}")
        print(f"Best version: {best_version.version}")
        print(f"Best score: {best_version.score:.3f}")
        print(f"\nTest Results:")
        print(f"  Pass rate: {best_version.metadata.get('pass_rate', 0):.1%}")
        print(f"  Passed: {best_version.metadata.get('passed', 0)}/{best_version.metadata.get('total', 0)}")

        print("\n💡 Next steps:")
        print("   1. Review the optimized prompt at:", output_path)
        print("   2. Test it manually with various inputs")
        print("   3. If satisfied, replace src/prompts/coordinator.md")
        print("   4. Run full test suite to ensure no regressions")
    else:
        print("   ⚠ No optimization performed (no versions created)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
