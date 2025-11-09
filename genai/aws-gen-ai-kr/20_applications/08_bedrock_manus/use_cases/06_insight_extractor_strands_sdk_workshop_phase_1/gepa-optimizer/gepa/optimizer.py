"""
GEPA Optimizer - Core Implementation

Implements the GEPA (Generate, Evaluate, Produce, Assess) methodology for prompt optimization.
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Any
from datetime import datetime
import json

from src.utils.strands_sdk_utils import strands_utils


@dataclass
class PromptVersion:
    """Represents a version of an optimized prompt with its evaluation results."""
    version: int
    prompt: str
    test_results: List[Dict[str, Any]]
    score: float
    metadata: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "prompt": self.prompt,
            "score": self.score,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "test_results_count": len(self.test_results)
        }


class GEPAOptimizer:
    """
    GEPA (Generate, Evaluate, Produce, Assess) Optimizer

    A systematic approach to prompt optimization that iteratively improves prompts
    through four phases:
    1. Generate: Run tests with current prompt
    2. Evaluate: Assess the quality of outputs
    3. Produce: Create improved prompt based on feedback
    4. Assess: Validate improvement and decide whether to continue

    Example:
        >>> optimizer = GEPAOptimizer(
        ...     base_prompt="You are a helpful assistant",
        ...     evaluation_fn=my_evaluator,
        ...     guidelines=["Be concise", "Use examples"]
        ... )
        >>> await optimizer.optimize(test_cases, max_iterations=5)
        >>> best = optimizer.get_best_prompt()
    """

    def __init__(
        self,
        base_prompt: str,
        evaluation_fn: Callable[[Dict], Any],
        agent_type: str = "claude-sonnet-4",
        guidelines: Optional[List[str]] = None,
        improvement_context: Optional[str] = None,
        enable_reasoning: bool = False,
        verbose: bool = True
    ):
        """
        Initialize GEPA Optimizer.

        Args:
            base_prompt: Initial prompt to optimize
            evaluation_fn: Function that evaluates agent outputs (can be sync or async)
            agent_type: Model to use for testing (default: claude-sonnet-4)
            guidelines: List of guidelines to follow during optimization
            improvement_context: Additional context for prompt improvement (e.g., from skills/)
            enable_reasoning: Whether to enable extended thinking for the test agent
            verbose: Whether to print progress information
        """
        self.base_prompt = base_prompt
        self.evaluation_fn = evaluation_fn
        self.agent_type = agent_type
        self.guidelines = guidelines or []
        self.improvement_context = improvement_context
        self.enable_reasoning = enable_reasoning
        self.verbose = verbose

        self.versions: List[PromptVersion] = []
        self.current_prompt = base_prompt

    async def _run_agent(self, prompt: str, test_input: str) -> str:
        """Run agent with given prompt and input."""
        # Create agent with the prompt to be tested
        agent = strands_utils.get_agent(
            agent_name="gepa_test_agent",
            system_prompts=prompt,
            agent_type=self.agent_type,
            enable_reasoning=self.enable_reasoning,
            prompt_cache_info=(False, "default"),
            streaming=True
        )

        # Process streaming response
        full_text = ""
        async for event in strands_utils.process_streaming_response_yield(
            agent, test_input, agent_name="gepa_test_agent", source="gepa_optimizer"
        ):
            if event.get("event_type") == "text_chunk":
                full_text += event.get("data", "")

        return full_text

    async def generate(self, test_cases: List[Dict]) -> List[Dict]:
        """
        Phase 1: Generate - Run tests with current prompt.

        Args:
            test_cases: List of test cases with 'input' and optional 'expected' fields

        Returns:
            List of test results with input, output, and expected values
        """
        if self.verbose:
            print(f"\n[GENERATE] Running {len(test_cases)} test cases...")

        results = []
        for i, test_case in enumerate(test_cases):
            if self.verbose:
                print(f"  Test {i+1}/{len(test_cases)}: {test_case.get('input', '')[:50]}...")

            output = await self._run_agent(self.current_prompt, test_case["input"])

            results.append({
                "input": test_case["input"],
                "output": output,
                "expected": test_case.get("expected"),
                "metadata": test_case.get("metadata", {})
            })

        return results

    async def evaluate(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Phase 2: Evaluate - Assess quality of outputs.

        Args:
            results: Test results from generate() phase

        Returns:
            Evaluation summary with scores and feedback
        """
        if self.verbose:
            print(f"\n[EVALUATE] Evaluating {len(results)} results...")

        scores = []
        feedback_list = []

        for i, result in enumerate(results):
            # Call evaluation function (handle both sync and async)
            if asyncio.iscoroutinefunction(self.evaluation_fn):
                eval_result = await self.evaluation_fn(result)
            else:
                eval_result = self.evaluation_fn(result)

            scores.append(eval_result["score"])
            feedback_list.append(eval_result.get("feedback", ""))

            if self.verbose:
                print(f"  Test {i+1}: Score = {eval_result['score']:.2f}")

        avg_score = sum(scores) / len(scores) if scores else 0.0
        passed_count = sum(1 for s in scores if s >= 0.8)

        evaluation = {
            "avg_score": avg_score,
            "scores": scores,
            "feedback": feedback_list,
            "passed": passed_count,
            "total": len(results),
            "pass_rate": passed_count / len(results) if results else 0.0
        }

        if self.verbose:
            print(f"  Average Score: {avg_score:.2f}")
            print(f"  Pass Rate: {evaluation['pass_rate']:.1%} ({passed_count}/{len(results)})")

        return evaluation

    async def produce(self, evaluation: Dict[str, Any]) -> str:
        """
        Phase 3: Produce - Create improved prompt based on evaluation.

        Args:
            evaluation: Evaluation results from evaluate() phase

        Returns:
            Improved prompt text
        """
        if self.verbose:
            print(f"\n[PRODUCE] Generating improved prompt...")

        # Build improvement prompt
        guidelines_text = "\n".join(f"- {g}" for g in self.guidelines) if self.guidelines else "N/A"

        context_section = ""
        if self.improvement_context:
            context_section = f"""
**Guidelines and Context:**
{self.improvement_context}
"""

        feedback_summary = "\n".join(
            f"- Test {i+1} (Score: {evaluation['scores'][i]:.2f}): {fb}"
            for i, fb in enumerate(evaluation['feedback']) if fb
        )

        improvement_prompt = f"""You are an expert prompt engineer. Your task is to improve the following system prompt based on evaluation results.

**Current Prompt:**
```
{self.current_prompt}
```

**Evaluation Results:**
- Average Score: {evaluation['avg_score']:.2f}
- Pass Rate: {evaluation['pass_rate']:.1%} ({evaluation['passed']}/{evaluation['total']})

**Detailed Feedback:**
{feedback_summary if feedback_summary else "No specific feedback provided."}

**Guidelines to Follow:**
{guidelines_text}
{context_section}

**Instructions:**
1. Analyze the evaluation results and identify weaknesses in the current prompt
2. Create an improved version that addresses the identified issues
3. Ensure the improved prompt is clear, specific, and actionable
4. Follow all provided guidelines and context
5. Maintain the core intent of the original prompt while improving effectiveness

**Output Format:**
Provide ONLY the improved prompt text without any explanations or metadata.
"""

        # Use a powerful model for improvement generation
        agent = strands_utils.get_agent(
            agent_name="gepa_improver",
            system_prompts="You are an expert prompt engineer who creates high-quality, effective prompts.",
            agent_type="claude-sonnet-4",
            enable_reasoning=False,
            prompt_cache_info=(False, "default"),
            streaming=True
        )

        # Get improved prompt
        improved_prompt = ""
        async for event in strands_utils.process_streaming_response_yield(
            agent, improvement_prompt, agent_name="gepa_improver", source="gepa_produce"
        ):
            if event.get("event_type") == "text_chunk":
                improved_prompt += event.get("data", "")

        # Clean up the response (remove markdown code blocks if present)
        improved_prompt = improved_prompt.strip()
        if improved_prompt.startswith("```"):
            lines = improved_prompt.split("\n")
            improved_prompt = "\n".join(lines[1:-1]).strip()

        if self.verbose:
            print(f"  Generated improved prompt ({len(improved_prompt)} chars)")

        return improved_prompt

    async def assess(self, new_prompt: str, test_cases: List[Dict]) -> tuple[bool, Dict[str, Any]]:
        """
        Phase 4: Assess - Validate improvement.

        Args:
            new_prompt: The improved prompt to test
            test_cases: Test cases to validate with

        Returns:
            Tuple of (improved: bool, evaluation: Dict)
        """
        if self.verbose:
            print(f"\n[ASSESS] Validating improved prompt...")

        # Temporarily switch to new prompt
        old_prompt = self.current_prompt
        self.current_prompt = new_prompt

        # Run tests with new prompt
        results = await self.generate(test_cases)
        evaluation = await self.evaluate(results)

        # Check if improved
        if self.versions:
            last_score = self.versions[-1].score
            improved = evaluation["avg_score"] > last_score
            if self.verbose:
                delta = evaluation["avg_score"] - last_score
                print(f"  Score change: {last_score:.2f} → {evaluation['avg_score']:.2f} ({delta:+.2f})")
        else:
            # First iteration - consider improved if score is decent
            improved = evaluation["avg_score"] >= 0.7
            if self.verbose:
                print(f"  Initial score: {evaluation['avg_score']:.2f}")

        # Save version
        version = PromptVersion(
            version=len(self.versions) + 1,
            prompt=new_prompt,
            test_results=results,
            score=evaluation["avg_score"],
            metadata=evaluation
        )
        self.versions.append(version)

        if not improved:
            # Revert to old prompt if not improved
            self.current_prompt = old_prompt
            if self.verbose:
                print("  ❌ No improvement - reverting to previous prompt")
        else:
            if self.verbose:
                print("  ✅ Improvement confirmed - using new prompt")

        return improved, evaluation

    async def optimize(
        self,
        test_cases: List[Dict],
        max_iterations: int = 5,
        target_score: float = 0.9,
        min_improvement: float = 0.01
    ) -> PromptVersion:
        """
        Run full GEPA optimization cycle.

        Args:
            test_cases: Test cases for optimization
            max_iterations: Maximum number of optimization iterations
            target_score: Target score to achieve (stops early if reached)
            min_improvement: Minimum improvement required to continue

        Returns:
            Best prompt version found
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"GEPA OPTIMIZATION START")
            print(f"{'='*60}")
            print(f"Base prompt length: {len(self.base_prompt)} chars")
            print(f"Test cases: {len(test_cases)}")
            print(f"Max iterations: {max_iterations}")
            print(f"Target score: {target_score}")

        for iteration in range(max_iterations):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration + 1}/{max_iterations}")
                print(f"{'='*60}")

            # Generate: Run tests
            results = await self.generate(test_cases)

            # Evaluate: Assess results
            evaluation = await self.evaluate(results)

            # Check if target reached (save current version first)
            if evaluation["avg_score"] >= target_score:
                # Save the current prompt as a version before breaking
                version = PromptVersion(
                    version=len(self.versions) + 1,
                    prompt=self.current_prompt,
                    test_results=results,
                    score=evaluation["avg_score"],
                    metadata=evaluation
                )
                self.versions.append(version)

                if self.verbose:
                    print(f"\n🎉 Target score {target_score} reached!")
                break

            # Produce: Create improved prompt
            new_prompt = await self.produce(evaluation)

            # Assess: Validate improvement
            improved, new_evaluation = await self.assess(new_prompt, test_cases)

            if not improved:
                if self.verbose:
                    print(f"\n⚠️ No improvement in iteration {iteration + 1}, stopping.")
                break

            # Check if improvement is too small
            if len(self.versions) > 1:
                improvement = self.versions[-1].score - self.versions[-2].score
                if improvement < min_improvement:
                    if self.verbose:
                        print(f"\n⚠️ Improvement too small ({improvement:.3f} < {min_improvement}), stopping.")
                    break

        best = self.get_best_prompt()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"OPTIMIZATION COMPLETE")
            print(f"{'='*60}")
            print(f"Total versions: {len(self.versions)}")
            if best:
                print(f"Best version: {best.version}")
                print(f"Best score: {best.score:.2f}")
                if len(self.versions) > 1:
                    initial_score = self.versions[0].score
                    print(f"Improvement: {best.score - initial_score:.2f}")
            else:
                print("No versions created (optimization may have completed on first iteration)")

        return best

    def get_best_prompt(self) -> Optional[PromptVersion]:
        """Get the best performing prompt version."""
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.score)

    def save_history(self, filepath: str):
        """Save optimization history to JSON file."""
        history = {
            "base_prompt": self.base_prompt,
            "agent_type": self.agent_type,
            "guidelines": self.guidelines,
            "versions": [v.to_dict() for v in self.versions]
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"💾 History saved to {filepath}")
