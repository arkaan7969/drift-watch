import json, os, time, statistics, itertools
from pathlib import Path
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from rich.table import Table
from rich import print

# ---------- CONFIG ----------
CONFIG_DIR = Path("configs")
PROMPTS_FILE = Path("data/prompts.json")
LOG_FILE = Path("logs/live_history.jsonl")
MODEL_CACHE = {}  # keep loaded model/pipeline objects per repo to avoid re-download
EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# stopping criteria
z_threshold = -1.0   # require mean z >= this for all canaries to stop
max_rounds = 12      # safety cap to avoid infinite runs
# per-inference generation params fallback (can be overridden by config json)
default_max_new_tokens = 80
# -----------------------------

def load_configs():
    cfgs = []
    for p in sorted(CONFIG_DIR.glob("canary_*.json")):
        cfg = json.load(open(p))
        cfg["__path"] = str(p)
        cfgs.append(cfg)
    return cfgs

def ensure_prompts():
    if not PROMPTS_FILE.exists():
        PROMPTS = [
            "Explain why strong passwords matter in one sentence.",
            "What is phishing and how can users avoid it?",
            "Define multi-factor authentication briefly.",
            "Why are software updates important for security?",
            "Describe one way to secure cloud data.",
            "What is the purpose of encryption?",
            "Explain the role of firewalls in network security.",
            "How can employees recognize social engineering?",
            "Why is data backup critical for organizations?",
            "What is least privilege access?"
        ]
        PROMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        PROMPTS_FILE.write_text(json.dumps(PROMPTS, indent=2))
    return json.load(open(PROMPTS_FILE))

def load_pipeline(repo, quant="4bit"):
    key = f"{repo}|{quant}"
    if key in MODEL_CACHE: return MODEL_CACHE[key]
    # load tokenizer + model into pipeline (keeps device_map="auto")
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    # Allow 4/8-bit flags (warning from HF is OK)
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    if quant == "4bit":
        kwargs["load_in_4bit"] = True
    elif quant == "8bit":
        kwargs["load_in_8bit"] = True
    model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, **kwargs)
    gen = pipeline("text-generation", model=model, tokenizer=tok, device_map="auto")
    MODEL_CACHE[key] = gen
    return gen

def format_prompt(template, question):
    if template:
        return template.replace("{prompt}", question)
    return f"### Human: {question}\n### Assistant:"

def run_round(cfgs, prompts):
    # returns list of dicts: {"id": id, "responses": [r1, r2, ...]}
    all_results = []
    for cfg in cfgs:
        model_repo = cfg.get("model_repo")
        quant = cfg.get("quant", "4bit")
        gen = load_pipeline(model_repo, quant=quant)
        responses = []
        for p in prompts:
            prompt_text = format_prompt(cfg.get("template",""), p)
            out = gen(prompt_text,
                      max_new_tokens=cfg.get("max_new_tokens", default_max_new_tokens),
                      temperature=cfg.get("temperature", 0.7),
                      top_p=cfg.get("top_p", 0.9),
                      do_sample=True)[0]["generated_text"]
            # remove prompt echo heuristically
            if "### Assistant:" in out:
                resp = out.split("### Assistant:")[-1].strip()
            else:
                resp = out[len(prompt_text):].strip() if out.startswith(prompt_text) else out.strip()
            responses.append(resp if resp else "<empty>")
        all_results.append({"id": cfg["id"], "responses": responses})
    return all_results

def compute_consensus_per_prompt(all_results):
    # all_results: list per canary with responses list (len = #prompts)
    n_canaries = len(all_results)
    n_prompts = len(all_results[0]["responses"])
    # embed each canary response per prompt
    mean_z_per_canary = {c["id"]: [] for c in all_results}
    for pi in range(n_prompts):
        texts = [c["responses"][pi] for c in all_results]
        embs = EMB_MODEL.encode(texts, normalize_embeddings=True)
        # pairwise sim
        sim = [[0.0]*n_canaries for _ in range(n_canaries)]
        for i,j in itertools.combinations(range(n_canaries),2):
            s = 1 - cosine(embs[i], embs[j])
            sim[i][j] = sim[j][i] = s
        consensus = []
        for i in range(n_canaries):
            others = [sim[i][k] for k in range(n_canaries) if k!=i]
            consensus.append(sum(others)/len(others))
        mu = statistics.mean(consensus)
        sd = statistics.pstdev(consensus) or 1e-6
        zs = [(v-mu)/sd for v in consensus]
        # append each canary's z for this prompt
        for i, c in enumerate(all_results):
            mean_z_per_canary[c["id"]].append(zs[i])
    # aggregate mean z per canary across prompts
    mean_z = {cid: statistics.mean(zs) for cid,zs in mean_z_per_canary.items()}
    std_z = {cid: statistics.pstdev(zs) for cid,zs in mean_z_per_canary.items()}
    return mean_z, std_z, mean_z_per_canary

def pretty_print_round(rnum, mean_z, std_z):
    tab = Table(title=f"Round {rnum} — Mean z per Canary")
    tab.add_column("Canary"); tab.add_column("Mean z"); tab.add_column("Std z"); tab.add_column("Status")
    for cid in sorted(mean_z.keys()):
        m = mean_z[cid]; s = std_z[cid]
        status = "[green]OK[/green]" if m >= z_threshold else "[red]ANOM[/red]"
        tab.add_row(cid, f"{m:+.3f}", f"{s:.3f}", status)
    print(tab)

def main():
    cfgs = load_configs()
    prompts = ensure_prompts()
    os.makedirs("logs", exist_ok=True)
    round_idx = 0
    while round_idx < max_rounds:
        round_idx += 1
        print(f"=== Running round {round_idx} ===")
        results = run_round(cfgs, prompts)
        mean_z, std_z, per_prompt_z = compute_consensus_per_prompt(results)
        pretty_print_round(round_idx, mean_z, std_z)
        # log this round
        entry = {"round": round_idx, "timestamp": time.time(), "mean_z": {k:float(v) for k,v in mean_z.items()}, "std_z": {k:float(v) for k,v in std_z.items()}}
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\\n")
        # stopping condition
        if all(m >= z_threshold for m in mean_z.values()):
            print(f"[green]All canaries >= {z_threshold}. Stopping.[/green]")
            return
        else:
            print(f"[yellow]Not converged yet — continuing (max_rounds={max_rounds}).[/yellow]")
            time.sleep(1)  # small pause between rounds
    print("[red]Reached max_rounds without full convergence[/red]")

if __name__ == '__main__':
    main()
