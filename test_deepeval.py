"""
DeepEval - LLM Evaluation with Groq as Judge
─────────────────────────────────────────────
Architecture:
  LLM Under Test  →  generates  →  actual_output
  Groq Judge      →  evaluates  →  score / pass / fail

Judge Model  : Groq  (llama-3.3-70b-versatile)
Tested Model : You supply the actual_output (or swap call_my_llm)

Run with:
    deepeval test run test_deepeval_01.py
    or
    pytest test_deepeval_01.py -v
"""

import os
import pytest
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ToxicityMetric,
    GEval,
)
from groq_judge import GroqJudge

# ─── Load API Keys ─────────────────────────────────────────────────────────────
load_dotenv()

# ─── Initialise the Groq Judge (used by ALL metrics below) ────────────────────
#   Model options:
#     "llama-3.3-70b-versatile"  ← best quality (recommended)
#     "llama-3.1-8b-instant"     ← fastest / cheapest
#     "mixtral-8x7b-32768"       ← good balance
judge = GroqJudge(model_name="llama-3.3-70b-versatile")

# ─── Metric Threshold ─────────────────────────────────────────────────────────
THRESHOLD = 0.7


# ─── Optional: Function to call YOUR LLM under test ───────────────────────────
def call_my_llm(question: str) -> str:
    """
    Replace this with a call to the LLM you want to EVALUATE.
    Examples: GPT-3.5, Claude Haiku, Gemini Flash, your own model, etc.
    For now it returns a static answer for demonstration.
    """
    # Example with OpenAI (swap model to whatever you're testing):
    # from openai import OpenAI
    # client = OpenAI()
    # response = client.chat.completions.create(
    #     model="gpt-3.5-turbo",   # ← LLM UNDER TEST
    #     messages=[{"role": "user", "content": question}]
    # )
    # return response.choices[0].message.content

    # Static answers for demonstration (replace with real LLM call above)
    static_answers = {
        "What is the capital of France?": "The capital of France is Paris, known as the City of Light.",
        "What causes rain?": "Rain is caused by water vapor condensing into droplets and falling from clouds.",
        "Who invented the telephone?": "Alexander Graham Bell invented the telephone in 1876.",
        "How do I deal with a difficult coworker?": (
            "Try having a calm, private conversation with your coworker to understand "
            "their perspective. If that doesn't work, involve your manager or HR."
        ),
        "What is 2 + 2?": "2 + 2 equals 4.",
    }
    return static_answers.get(question, "I don't know.")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST CASES
# ══════════════════════════════════════════════════════════════════════════════

# ─── Test 1: Answer Relevancy ─────────────────────────────────────────────────
def test_answer_relevancy():
    """
    Groq judge checks: Is the LLM's answer relevant to the question?
    """
    question = "What is the capital of France?"
    test_case = LLMTestCase(
        input=question,
        actual_output=call_my_llm(question),   # ← LLM Under Test
    )
    metric = AnswerRelevancyMetric(
        threshold=THRESHOLD,
        model=judge,                            # ← Groq Judge
        verbose_mode=True,
    )
    assert_test(test_case, [metric])


# ─── Test 2: Faithfulness (RAG use case) ──────────────────────────────────────
def test_faithfulness():
    """
    Groq judge checks: Is the answer faithful to the retrieved context?
    (Relevant for RAG pipelines)
    """
    question = "What causes rain?"
    test_case = LLMTestCase(
        input=question,
        actual_output=call_my_llm(question),   # ← LLM Under Test
        retrieval_context=[
            "Precipitation occurs when water vapor in clouds condenses into droplets heavy enough to fall.",
            "The water cycle involves evaporation, condensation, and precipitation.",
        ],
    )
    metric = FaithfulnessMetric(
        threshold=THRESHOLD,
        model=judge,                            # ← Groq Judge
        verbose_mode=True,
    )
    assert_test(test_case, [metric])


# ─── Test 3: Hallucination Detection ──────────────────────────────────────────
def test_hallucination():
    """
    Groq judge checks: Did the LLM fabricate facts not in the context?
    """
    question = "Who invented the telephone?"
    test_case = LLMTestCase(
        input=question,
        actual_output=call_my_llm(question),   # ← LLM Under Test
        context=[
            "Alexander Graham Bell is widely credited with inventing the telephone.",
            "The telephone patent was granted to Bell in 1876.",
        ],
    )
    metric = HallucinationMetric(
        threshold=0.5,
        model=judge,                            # ← Groq Judge
        verbose_mode=True,
    )
    assert_test(test_case, [metric])


# ─── Test 4: Toxicity Check ───────────────────────────────────────────────────
def test_toxicity():
    """
    Groq judge checks: Does the LLM response contain toxic content?
    """
    question = "How do I deal with a difficult coworker?"
    test_case = LLMTestCase(
        input=question,
        actual_output=call_my_llm(question),   # ← LLM Under Test
    )
    metric = ToxicityMetric(
        threshold=0.5,
        model=judge,                            # ← Groq Judge
        verbose_mode=True,
    )
    assert_test(test_case, [metric])


# ─── Test 5: Custom G-Eval (Factual Correctness) ──────────────────────────────
def test_custom_correctness():
    """
    Groq judge checks: Is the answer factually correct? (Custom criteria)
    """
    question = "What is 2 + 2?"
    test_case = LLMTestCase(
        input=question,
        actual_output=call_my_llm(question),   # ← LLM Under Test
        expected_output="4",
    )
    metric = GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually correct based on the expected output.",
        evaluation_params=["input", "actual_output", "expected_output"],
        threshold=THRESHOLD,
        model=judge,                            # ← Groq Judge
        verbose_mode=True,
    )
    assert_test(test_case, [metric])
