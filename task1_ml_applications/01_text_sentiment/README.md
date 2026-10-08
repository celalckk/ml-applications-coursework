# Text Sentiment Analysis

## 1. Library / Solution Advantages

**Library:** Hugging Face Transformers
**Model:** `distilbert-base-uncased-finetuned-sst-2-english`

Hugging Face Transformers was chosen for the following reasons:
- **Ready-made pipelines:** A single `pipeline()` call loads a model, tokenizer, and pre/post-processing logic.
- **Model hub:** Thousands of pre-trained models are available without any training effort.
- **Consistency:** Same API works for text, audio, image, and multimodal tasks.
- **Community support:** Extremely well-documented and widely used in both industry and academia.

## 2. Model Principle and Selection Reasoning

**Model:** DistilBERT fine-tuned on SST-2 (Stanford Sentiment Treebank).

**How it works:**
- DistilBERT is a smaller, faster version of BERT obtained via knowledge distillation (the student model learns to mimic the teacher's outputs).
- The input text is tokenized into subword tokens and passed through transformer layers.
- A classification head produces a probability distribution over two labels: POSITIVE and NEGATIVE.
- The label with the highest probability is returned together with a confidence score.

**Why this model:**
- Very small (~250 MB) and fast on CPU.
- High accuracy (>90%) on SST-2 benchmark.
- Perfect for demonstration and local deployment without a GPU.

## 3. Dataset Structure

**For inference (what we use):**
- Input: raw text string.
- Preprocessing: automatic tokenization (WordPiece), padding, truncation to 512 tokens.
- Output: `{ "label": "POSITIVE" | "NEGATIVE", "score": float }`.

**For training (background info):**
- SST-2 consists of ~67,000 movie review sentences labeled with binary sentiment.
- Used only by the model authors, not by us (we use the model as-is).

## 4. Quality Metrics

- **Confidence score** (0–1): returned by the model for each prediction.
- **Accuracy** on SST-2 benchmark: ~91%.
- In production, one would additionally monitor:
  - **Precision / Recall / F1** per class.
  - **Latency** per request (ms).
  - **CPU / memory usage**.

## 5. Implementation Features and Efficiency

- Loaded through `transformers.pipeline("sentiment-analysis")`.
- Runs entirely on CPU; no GPU required.
- Inference time: ~20–50 ms per short sentence on Intel Arc iGPU / CPU.
- Model is cached locally after the first download (`~/.cache/huggingface/`).
- The same pipeline can be wrapped in a REST API (e.g., FastAPI) for production use.