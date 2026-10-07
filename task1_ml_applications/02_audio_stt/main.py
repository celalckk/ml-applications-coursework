"""
Task 1 - Assignment 2: Audio Speech-to-Text
Library: Hugging Face Transformers
Model: openai/whisper-small
"""

from transformers import pipeline

print("Loading Whisper model... (first run will download ~1GB)")
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small"
)
print("Model ready!\n")

# Path to the audio file
audio_file = "task1_ml_applications/02_audio_stt/sample.flac"

print("=" * 60)
print("SPEECH-TO-TEXT TRANSCRIPTION")
print("=" * 60)

result = asr_pipeline(audio_file)

print(f"\nAudio file   : {audio_file}")
print(f"Transcription: {result['text']}")

print("\n" + "=" * 60)
print("Transcription completed!")
print("=" * 60)