import os, json, torch, random, time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

CONFIG_DIR = "configs"
USER_QUESTION = "Explain why strong passwords matter in one simple sentence."

def format_prompt(question: str):
    return f"### Human: {question}\n### Assistant:"

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)

def run_canary(cfg):
    set_seed(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["model_repo"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_repo"],
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    gen = pipeline("text-generation", model=model, tokenizer=tok)
    prompt = format_prompt(USER_QUESTION)
    out = gen(
        prompt,
        max_new_tokens=cfg["max_new_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        do_sample=True
    )[0]["generated_text"]
    # clean echo
    response = out.split("### Assistant:")[-1].strip()
    return response

def main():
    configs = [json.load(open(os.path.join(CONFIG_DIR, f)))
               for f in sorted(os.listdir(CONFIG_DIR)) if f.endswith(".json")]
    results = []
    print(f"Running {len(configs)} canaries...\n")
    for cfg in configs:
        t0 = time.time()
        resp = run_canary(cfg)
        dt = time.time() - t0
        print(f"[{cfg['id']}] temp={cfg['temperature']}, top_p={cfg['top_p']}, time={dt:.1f}s\n→ {resp}\n")
        results.append({"id": cfg["id"], "response": resp})
    json.dump(results, open("canary_results.json","w"), indent=2)
    print("\n✅ Responses saved to canary_results.json")

if __name__ == "__main__":
    main()
