from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch, time

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"Loading {MODEL_NAME} ...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "Explain why strong passwords matter, in one simple sentence."
print(f"\nPrompt: {prompt}\n")

out = generator(prompt, max_new_tokens=40, temperature=0.7)[0]["generated_text"]
print("Response:\n", out)
print(f"\nElapsed: {time.time() - t0:.1f}s")
print("GPU memory allocated:", torch.cuda.memory_allocated() / 1e6, "MB")
