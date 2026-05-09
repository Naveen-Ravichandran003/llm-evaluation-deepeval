from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric
from groq_judge import GroqJudge
import pytest

def test_deepeval():
    judge = GroqJudge(model_name="llama-3.3-70b-versatile")

    test = LLMTestCase(
        name="test_deepeval",
        input="What is the capital of Indiadeepeval test run?",
        actual_output="The capital of India is New Delhi.",
        expected_output="New Delhi",
        context=["New Delhi is the capital and largest city of India."],
    )

    metrics = [
        AnswerRelevancyMetric(threshold=0.5, model=judge),
        HallucinationMetric(threshold=0.5, model=judge),
    ]

    assert_test(test, metrics)