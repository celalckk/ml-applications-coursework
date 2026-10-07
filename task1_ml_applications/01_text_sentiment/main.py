"""
Task 1 - Assignment 1: Text Sentiment Analysis
Library: Hugging Face Transformers
Model: distilbert-base-uncased-finetuned-sst-2-english
"""

from transformers import pipeline

print("Loading model... (will be downloaded from internet on first run)")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("Model ready!\n")

# Test texts
test_texts = [
    "I absolutely love this product! It works perfectly.",
    "This is the worst experience I have ever had.",
    "The movie was okay, nothing special.",
    "The food was amazing and the service was excellent!",
    "I am really disappointed with the quality.",
]

print("=" * 60)
print("SENTIMENT ANALYSIS RESULTS")
print("=" * 60)

for text in test_texts:
    result = sentiment_pipeline(text)[0]
    label = result["label"]
    score = result["score"]
    emoji = "[+]" if label == "POSITIVE" else "[-]"
    print(f"\n{emoji} Text  : {text}")
    print(f"    Result : {label} (Confidence: {score:.2%})")

print("\n" + "=" * 60)
print("All texts have been analyzed!")
print("=" * 60)