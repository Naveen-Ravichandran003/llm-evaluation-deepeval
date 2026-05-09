# 🧪 DeepEval — LLM Evaluation with Groq as Judge

> A lightweight, production-ready framework for evaluating Large Language Model outputs using [DeepEval](https://github.com/confident-ai/deepeval) metrics with **Groq-hosted models** as the evaluator/judge — delivering fast, free-tier evaluation without needing an OpenAI API key.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     DeepEval Framework                  │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │  LLM Under   │───▶│  Test Case   │───▶│  DeepEval │  │
│  │    Test       │    │  (input /    │    │  Metrics  │  │
│  │ (any model)   │    │   output)    │    │           │  │
│  └──────────────┘    └──────────────┘    └─────┬─────┘  │
│                                                │        │
│                                          ┌─────▼─────┐  │
│                                          │   Groq     │  │
│                                          │   Judge    │  │
│                                          │ (LLaMA 3.3)│  │
│                                          └─────┬─────┘  │
│                                                │        │
│                                          ┌─────▼─────┐  │
│                                          │  Score /   │  │
│                                          │ Pass/Fail  │  │
│                                          └───────────┘  │
└─────────────────────────────────────────────────────────┘
```

**How it works:**

1. **LLM Under Test** generates an `actual_output` for a given input question
2. **DeepEval Metrics** define the evaluation criteria (relevancy, faithfulness, hallucination, etc.)
3. **Groq Judge** (`GroqJudge` — a custom `DeepEvalBaseLLM` wrapper) scores the output using a Groq-hosted LLM
4. **Pass / Fail** is determined by comparing the score against a configurable threshold

---

## 🧠 What is LLM Evaluation?

LLM Evaluation is the systematic process of assessing the performance, reliability, and safety of Large Language Models. Unlike traditional software where unit tests expect deterministic outputs (e.g., `2 + 2 = 4`), LLMs generate probabilistic natural language. This requires a paradigm shift from **exact-match assertions** to **LLM-as-a-Judge evaluations**.

Instead of writing complex regex or string-matching rules to grade an output, we use another high-quality LLM (the **Judge**) to evaluate the generated text against specific criteria (like relevancy, factual consistency, or toxicity). In this project, we use **Groq's LLaMA 3.3 70B** as the judge to evaluate the outputs.

---

## 🎯 DeepEval Evaluation Metrics in Detail

This framework implements five core metrics provided by DeepEval, each designed to evaluate a specific dimension of LLM performance:

### 1. Answer Relevancy
* **What it measures:** Does the LLM actually answer the user's question, or does it ramble and provide irrelevant information?
* **How it works:** The judge evaluates the `actual_output` against the `input` to penalize incomplete answers or redundant information.
* **Best for:** General Q&A bots, customer support assistants.

### 2. Faithfulness
* **What it measures:** Is the LLM's response strictly derived from the provided context?
* **How it works:** Crucial for Retrieval-Augmented Generation (RAG). The judge compares the `actual_output` against the `retrieval_context` to ensure the model isn't making up information outside the bounds of the provided documents.
* **Best for:** Document Q&A, enterprise search, RAG pipelines.

### 3. Hallucination
* **What it measures:** Did the LLM fabricate facts or hallucinate details?
* **How it works:** Similar to faithfulness, but specifically looks for contradictions between the `actual_output` and the provided `context`.
* **Best for:** Fact-checking, summarization, medical/legal chatbots.

### 4. Toxicity
* **What it measures:** Is the response safe for users?
* **How it works:** The judge analyzes the `actual_output` for harmful, offensive, biased, or toxic content.
* **Best for:** Public-facing applications, social media bots, safety guardrails.

### 5. G-Eval (Custom Criteria)
* **What it measures:** Anything you want! G-Eval is a flexible, criteria-based metric.
* **How it works:** You define a natural language rubric (e.g., "Determine whether the actual output is factually correct based on the expected output"). The judge follows these exact instructions to generate a score.
* **Best for:** Domain-specific evaluations, tone/style checks, formatting compliance.

---

## 🗂️ Project Structure

```
DeepEval/
├── groq_judge.py            # Custom Groq judge (DeepEvalBaseLLM wrapper)
├── test_deepeval.py          # Full test suite — 5 metrics with Groq judge
├── test_deepeval_01.py       # Minimal test — Answer Relevancy + Hallucination
├── test_basic.py             # Sanity check (Hello World)
├── .env                      # API keys (GROQ_API_KEY, OPENAI_API_KEY, etc.)
├── .env.local                # DeepEval Confident AI dashboard keys
├── .deepeval/                # DeepEval cache & test run history
│   ├── .deepeval-cache.json
│   └── .latest_test_run.json
└── venv/                     # Python virtual environment
```

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Evaluation Framework** | [DeepEval](https://github.com/confident-ai/deepeval) `v3.9.7` |
| **Judge Model** | [Groq](https://groq.com/) — `llama-3.3-70b-versatile` |
| **Judge SDK** | `groq` Python SDK `v0.25.0` |
| **Test Runner** | `pytest` `v9.0.2` |
| **Environment** | `python-dotenv` `v1.2.2` |
| **Language** | Python 3.x |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/keys) (free tier available)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/DeepEval.git
cd DeepEval
```

### 2. Create & Activate Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install deepeval groq python-dotenv pytest
```

### 4. Configure API Keys

Create a `.env` file in the project root:

```env
# Required: Groq API key (used as the judge/evaluator)
GROQ_API_KEY=your-groq-api-key-here

# Optional: OpenAI (if testing OpenAI models)
OPENAI_API_KEY=your-openai-api-key-here
```

### 5. Run Tests

```bash
# Using DeepEval CLI (recommended — generates dashboard report)
deepeval test run test_deepeval.py

# Using pytest directly
pytest test_deepeval.py -v

# Run the minimal test
deepeval test run test_deepeval_01.py
```

---

## 🔧 Custom Groq Judge — `groq_judge.py`

The `GroqJudge` class wraps any Groq-hosted model as a DeepEval-compatible evaluator by implementing the `DeepEvalBaseLLM` interface:

```python
from groq_judge import GroqJudge

judge = GroqJudge(model_name="llama-3.3-70b-versatile")
metric = AnswerRelevancyMetric(threshold=0.7, model=judge)
```

### Supported Groq Models

| Model | Strengths |
|-------|-----------|
| `llama-3.3-70b-versatile` | Best quality — **recommended for judging** |
| `llama-3.1-8b-instant` | Fastest / cheapest |
| `mixtral-8x7b-32768` | Good balance of speed & quality |
| `gemma2-9b-it` | Google Gemma via Groq |

---

## 📊 Sample Test Results

From the latest test run (`test_deepeval_01.py`):

| Metric | Score | Threshold | Result |
|--------|-------|-----------|--------|
| Answer Relevancy | **1.00** | 0.5 | ✅ Pass |
| Hallucination | **0.00** | 0.5 | ✅ Pass |

> **Judge Model:** `Groq/llama-3.3-70b-versatile`
> **Run Duration:** ~42 seconds (2 metrics, 1 test case)

---

## 🧩 Extending the Framework

### Evaluate Your Own LLM

Edit the `call_my_llm()` function in `test_deepeval.py` to call your model:

```python
def call_my_llm(question: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",             # ← Your LLM under test
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content
```

### Add New Metrics

DeepEval supports many more metrics — simply import and add them:

```python
from deepeval.metrics import BiasMetric, ContextualRelevancyMetric

metric = BiasMetric(threshold=0.5, model=judge)
```

---

## 📄 License

This project is open-source. Feel free to use and modify it for your own LLM evaluation needs.

---

## 👤 Author

**Naveen Ravichandran**

---

<p align="center">
  Built with ❤️ using <a href="https://github.com/confident-ai/deepeval">DeepEval</a> + <a href="https://groq.com/">Groq</a>
</p>
