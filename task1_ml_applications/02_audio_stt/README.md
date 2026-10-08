# Speech-to-Text (Audio Transcription)

# 1. Library / Solution Advantages

**Library:** Hugging Face Transformers
**Model:** `openai/whisper-small`

Advantages:
- Whisper is one of the most accurate open-source ASR models.
- Multi-language support (99 languages).
- Robust to background noise, accents, and varying audio quality.
- Same `pipeline()` API as our text task — consistency.

# 2. Model Principle and Selection Reasoning

**How Whisper works:**
- Input audio is converted to a log-Mel spectrogram.
- An encoder-decoder Transformer processes the spectrogram.
- The decoder generates tokens autoregressively (text output).
- Supports transcription and translation tasks in one model.

**Why Whisper-small:**
- Balance between accuracy and size (~1 GB).
- Runs on CPU at reasonable speed.
- Whisper-tiny is faster but less accurate; Whisper-large is too heavy for CPU-only machines.

# 3. Dataset Structure

**For inference:**
- Input: audio file (WAV / FLAC / MP3, 16 kHz recommended).
- Preprocessing: resampling, log-Mel spectrogram extraction.
- Output: `{ "text": "transcribed string" }`.

**For training (background):**
- Whisper was trained on 680,000 hours of multilingual audio from the web.
- We do not retrain it; we use the released weights.

# 4. Quality Metrics

- **Word Error Rate (WER):** standard ASR metric.
- **Character Error Rate (CER):** useful for short clips.
- **Real-Time Factor (RTF):** how long inference takes vs. audio length.
- Whisper-small achieves ~7–10% WER on LibriSpeech (clean English).

# 5. Implementation Features and Efficiency

- Requires **FFmpeg** installed on the system for audio decoding.
- CPU inference: ~1–2× real-time for small model.
- Batch processing of long audio is possible using chunking.
- For production: pre-load model once, then serve via API.