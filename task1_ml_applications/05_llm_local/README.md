# Local LLM Deployment

## 1. Library / Solution Advantages

**Tool:** Ollama
**Model:** `llama3.2:1b`

Advantages:
- Runs **100% locally** — no data sent to third-party servers.
- Simple installation and model management (`ollama pull`, `ollama run`).
- Exposes a REST API and an official Python client.
- Works on CPU without a GPU.

## 2. Model Principle and Selection Reasoning

**How Llama 3.2 1B works:**
- Decoder-only Transformer with 1.3 billion parameters.
- Uses grouped-query attention (GQA) and rotary positional embeddings.
- Token-by-token autoregressive generation with sampling (temperature, top-p).
- Quantized (4-bit) version runs comfortably on CPU.

**Why Llama 3.2 1B:**
- Small enough to run on a laptop with no NVIDIA GPU.
- Meta's instruction-tuned model — good at chat and simple reasoning.
- Alternatives: Phi-3-mini (3.8B, larger), Gemma 2B (also good).

## 3. Dataset Structure

**For inference:**
- Input: a list of chat messages (`[{"role": "user", "content": "..."}]`).
- Tokenization handled internally by Ollama.
- Output: model response as a string.

**For training (background):**
- Llama 3.2 was trained on a large multilingual corpus (trillions of tokens).
- We use the released weights; no fine-tuning performed.

## 4. Quality Metrics

- **Perplexity** (lower is better) — standard LM metric.
- **Latency** per token (ms/token) — critical for UX.
- **Throughput** (tokens/second).
- **Task-specific evaluation:** MMLU, HumanEval, etc. (published by Meta).
- In production: user satisfaction, session length, hallucination rate.

## 5. Implementation Features and Efficiency

- Ollama runs as a background service on port 11434.
- Python client: `ollama.chat(model=..., messages=...)`.
- On this machine (no GPU), Llama 3.2 1B generates ~10–20 tokens/second.
- For production: use quantized models, GPU acceleration, or a larger model.