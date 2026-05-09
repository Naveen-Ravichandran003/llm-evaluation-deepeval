"""
groq_judge.py
─────────────
Custom DeepEval judge wrapper for Groq models.
Implements the DeepEvalBaseLLM interface so any Groq model
can be used as an evaluator/judge inside DeepEval metrics.

Supported Groq Models (fast & free-tier available):
  - llama-3.3-70b-versatile   (recommended judge — best quality)
  - llama-3.1-8b-instant      (fastest)
  - mixtral-8x7b-32768        (good balance)
  - gemma2-9b-it              (Google Gemma via Groq)
"""

import os
from typing import Optional
from groq import Groq
from deepeval.models.base_model import DeepEvalBaseLLM


class GroqJudge(DeepEvalBaseLLM):
    """
    Wraps a Groq-hosted LLM as a DeepEval judge.

    Usage:
        judge = GroqJudge(model_name="llama-3.3-70b-versatile")
        metric = AnswerRelevancyMetric(threshold=0.7, model=judge)
    """

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        super().__init__()

    def load_model(self):
        """Returns the Groq client (already initialised in __init__)."""
        return self._client

    def generate(self, prompt: str, *args, **kwargs) -> str:
        """Synchronous generation — called by DeepEval metrics."""
        client = self.load_model()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        """Async generation — called by DeepEval async metrics."""
        # Groq SDK is sync-only; run in thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt)

    def get_model_name(self) -> str:
        return f"Groq/{self.model_name}"
