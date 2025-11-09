"""
LLM Evaluator - Automatic evaluation using LLM as judge

Uses Claude as a judge to automatically evaluate prompt outputs based on
configurable criteria.
"""

import asyncio
from typing import Dict, List, Any, Optional
from src.utils.strands_sdk_utils import strands_utils


class LLMEvaluator:
    """
    LLM-based evaluator that uses Claude as a judge to assess prompt outputs.

    Evaluation is based on three main criteria:
    - Structural correctness (30%): Does output follow expected format?
    - Quality (40%): Is the content accurate, relevant, and well-structured?
    - Guideline adherence (30%): Does it follow specified guidelines?

    Example:
        >>> evaluator = LLMEvaluator(
        ...     criteria={
        ...         "format": "JSON with 'analysis' and 'recommendations' fields",
        ...         "tone": "Professional and concise",
        ...         "length": "Between 200-500 words"
        ...     },
        ...     guidelines=["Be specific", "Use examples"]
        ... )
        >>> result = await evaluator.evaluate({
        ...     "input": "Analyze this data",
        ...     "output": "...",
        ...     "expected": "..."
        ... })
        >>> print(result["score"])  # 0.0 to 1.0
    """

    def __init__(
        self,
        criteria: Optional[Dict[str, str]] = None,
        guidelines: Optional[List[str]] = None,
        judge_model: str = "claude-sonnet-4",
        weights: Optional[Dict[str, float]] = None,
        verbose: bool = False
    ):
        """
        Initialize LLM Evaluator.

        Args:
            criteria: Specific evaluation criteria (e.g., format, tone, length)
            guidelines: List of guidelines that outputs should follow
            judge_model: Model to use as judge (default: claude-sonnet-4)
            weights: Custom weights for scoring (structure, quality, guidelines)
            verbose: Whether to print evaluation details
        """
        self.criteria = criteria or {}
        self.guidelines = guidelines or []
        self.judge_model = judge_model
        self.verbose = verbose

        # Default weights
        self.weights = weights or {
            "structure": 0.30,
            "quality": 0.40,
            "guidelines": 0.30
        }

    async def evaluate(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single test result.

        Args:
            result: Dictionary with 'input', 'output', and optional 'expected' keys

        Returns:
            Dictionary with 'score' (0.0-1.0) and 'feedback' (str)
        """
        input_text = result.get("input", "")
        output_text = result.get("output", "")
        expected = result.get("expected", "")
        metadata = result.get("metadata", {})

        # Build evaluation prompt
        criteria_text = "\n".join(
            f"- **{key}**: {value}"
            for key, value in self.criteria.items()
        ) if self.criteria else "No specific criteria provided."

        guidelines_text = "\n".join(
            f"- {g}" for g in self.guidelines
        ) if self.guidelines else "No specific guidelines provided."

        expected_text = f"""
**Expected Behavior:**
{expected}
""" if expected else ""

        metadata_text = ""
        if metadata:
            metadata_text = f"""
**Additional Context:**
{chr(10).join(f"- {k}: {v}" for k, v in metadata.items())}
"""

        evaluation_prompt = f"""You are an expert evaluator for AI prompt outputs. Your task is to assess the quality of an AI agent's response based on specific criteria.

**Input Given to Agent:**
{input_text}

**Agent's Output:**
{output_text}
{expected_text}{metadata_text}

**Evaluation Criteria:**
{criteria_text}

**Guidelines to Check:**
{guidelines_text}

**Your Task:**
Evaluate the output on three dimensions:

1. **Structural Correctness (0-10)**: Does the output follow the expected format and structure?
2. **Quality (0-10)**: Is the content accurate, relevant, complete, and well-organized?
3. **Guideline Adherence (0-10)**: Does it follow the specified guidelines?

**Output Format (JSON):**
```json
{{
  "structure_score": <0-10>,
  "quality_score": <0-10>,
  "guidelines_score": <0-10>,
  "feedback": {{
    "structure": "Brief explanation of structural assessment",
    "quality": "Brief explanation of quality assessment",
    "guidelines": "Brief explanation of guideline adherence"
  }},
  "overall_feedback": "1-2 sentence summary of strengths and areas for improvement"
}}
```

Provide ONLY the JSON output, no additional text.
"""

        # Create judge agent
        agent = strands_utils.get_agent(
            agent_name="llm_judge",
            system_prompts="You are an expert evaluator who provides objective, constructive feedback on AI outputs.",
            agent_type=self.judge_model,
            enable_reasoning=False,
            prompt_cache_info=(False, "default"),
            streaming=True
        )

        # Get evaluation
        eval_response = ""
        async for event in strands_utils.process_streaming_response_yield(
            agent, evaluation_prompt, agent_name="llm_judge", source="llm_evaluator"
        ):
            if event.get("event_type") == "text_chunk":
                eval_response += event.get("data", "")

        # Parse JSON response
        eval_data = self._parse_evaluation_response(eval_response)

        # Calculate weighted score
        final_score = (
            (eval_data["structure_score"] / 10.0) * self.weights["structure"] +
            (eval_data["quality_score"] / 10.0) * self.weights["quality"] +
            (eval_data["guidelines_score"] / 10.0) * self.weights["guidelines"]
        )

        if self.verbose:
            print(f"\n  Evaluation Scores:")
            print(f"    Structure: {eval_data['structure_score']}/10")
            print(f"    Quality: {eval_data['quality_score']}/10")
            print(f"    Guidelines: {eval_data['guidelines_score']}/10")
            print(f"    Final: {final_score:.2f}")

        return {
            "score": final_score,
            "feedback": eval_data.get("overall_feedback", ""),
            "detailed_scores": {
                "structure": eval_data["structure_score"],
                "quality": eval_data["quality_score"],
                "guidelines": eval_data["guidelines_score"]
            },
            "detailed_feedback": eval_data.get("feedback", {})
        }

    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from judge, with fallback for malformed responses."""
        import json
        import re

        # Try to extract JSON from response
        try:
            # Remove markdown code blocks if present
            cleaned = response.strip()
            if "```json" in cleaned:
                match = re.search(r'```json\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
            elif "```" in cleaned:
                match = re.search(r'```\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)

            # Parse JSON
            data = json.loads(cleaned)

            # Validate required fields
            required = ["structure_score", "quality_score", "guidelines_score"]
            for field in required:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")

            return data

        except Exception as e:
            if self.verbose:
                print(f"  Warning: Failed to parse evaluation response: {e}")
                print(f"  Response: {response[:200]}...")

            # Fallback: Return default scores
            return {
                "structure_score": 5,
                "quality_score": 5,
                "guidelines_score": 5,
                "feedback": {
                    "structure": "Could not evaluate",
                    "quality": "Could not evaluate",
                    "guidelines": "Could not evaluate"
                },
                "overall_feedback": "Evaluation failed - using default scores"
            }

    async def evaluate_batch(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple results in parallel.

        Args:
            results: List of test results to evaluate

        Returns:
            List of evaluation results
        """
        tasks = [self.evaluate(result) for result in results]
        return await asyncio.gather(*tasks)


class SimpleEvaluator:
    """
    Simple rule-based evaluator for basic validation.

    Useful for quick structural checks without LLM overhead.

    Example:
        >>> evaluator = SimpleEvaluator(
        ...     required_keywords=["analysis", "recommendation"],
        ...     min_length=100,
        ...     max_length=1000
        ... )
        >>> result = evaluator.evaluate({
        ...     "input": "...",
        ...     "output": "This is an analysis with recommendations"
        ... })
    """

    def __init__(
        self,
        required_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        expected_format: Optional[str] = None  # 'json', 'markdown', etc.
    ):
        """
        Initialize Simple Evaluator.

        Args:
            required_keywords: Keywords that must appear in output
            forbidden_keywords: Keywords that must NOT appear in output
            min_length: Minimum output length (characters)
            max_length: Maximum output length (characters)
            expected_format: Expected format ('json', 'markdown', etc.)
        """
        self.required_keywords = [k.lower() for k in (required_keywords or [])]
        self.forbidden_keywords = [k.lower() for k in (forbidden_keywords or [])]
        self.min_length = min_length
        self.max_length = max_length
        self.expected_format = expected_format

    def evaluate(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate result using simple rules."""
        output = result.get("output", "").lower()
        score = 1.0
        feedback_parts = []

        # Check required keywords
        if self.required_keywords:
            missing = [k for k in self.required_keywords if k not in output]
            if missing:
                score -= 0.3
                feedback_parts.append(f"Missing keywords: {', '.join(missing)}")

        # Check forbidden keywords
        if self.forbidden_keywords:
            found = [k for k in self.forbidden_keywords if k in output]
            if found:
                score -= 0.3
                feedback_parts.append(f"Contains forbidden keywords: {', '.join(found)}")

        # Check length
        output_len = len(result.get("output", ""))
        if self.min_length and output_len < self.min_length:
            score -= 0.2
            feedback_parts.append(f"Too short ({output_len} < {self.min_length})")
        if self.max_length and output_len > self.max_length:
            score -= 0.2
            feedback_parts.append(f"Too long ({output_len} > {self.max_length})")

        # Check format
        if self.expected_format:
            if self.expected_format == 'json':
                import json
                try:
                    json.loads(result.get("output", ""))
                except:
                    score -= 0.3
                    feedback_parts.append("Invalid JSON format")

        score = max(0.0, score)
        feedback = "; ".join(feedback_parts) if feedback_parts else "All checks passed"

        return {
            "score": score,
            "feedback": feedback
        }
