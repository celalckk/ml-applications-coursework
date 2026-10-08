import ollama

MODEL_NAME = "llama3.2:1b"

print("=" * 60)
print(f"LOCAL LLM INTERACTION (Model: {MODEL_NAME})")
print("=" * 60)


prompts = [
    "Explain what machine learning is in one sentence.",
    "What is the capital of France?",
    "Write a short haiku about programming.",
]

for i, prompt in enumerate(prompts, 1):
    print(f"\n[Prompt {i}] {prompt}")
    print("-" * 60)
    
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    
    print(f"[Response] {response['message']['content']}")

print("\n" + "=" * 60)
print("All prompts completed successfully!")
print("=" * 60)