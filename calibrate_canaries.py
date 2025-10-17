import json, os, itertools, statistics, time
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from rich.table import Table
from rich import print

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

responses = json.load(open("canary_results.json"))
ids = [r["id"] for r in responses]
print(f"Calibrating {len(ids)} canaries on {len(PROMPTS)} prompts\n")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
history = {cid: [] for cid in ids}

for idx, prompt in enumerate(PROMPTS, 1):
    print(f"[bold cyan]Prompt {idx}/{len(PROMPTS)}[/bold cyan]: {prompt}")
    # Load responses again each round (reuse your run_canaries later)
    # Here we simulate by re-embedding last responses, replace with fresh text per canary if desired
    texts = [r["response"] + " " + prompt for r in responses]
    embs = model.encode(texts, normalize_embeddings=True)

    n = len(embs)
    sim = [[0.0]*n for _ in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        s = 1 - cosine(embs[i], embs[j])
        sim[i][j] = sim[j][i] = s

    consensus = []
    for i in range(n):
        others = [sim[i][k] for k in range(n) if k != i]
        consensus.append(sum(others)/len(others))

    mu = statistics.mean(consensus)
    sd = statistics.pstdev(consensus) or 1e-6
    zs = [(v - mu)/sd for v in consensus]

    for cid, z in zip(ids, zs):
        history[cid].append(z)

# === Aggregate ===
tab = Table(title="Baseline Calibration (mean z-scores over prompts)")
tab.add_column("Canary")
tab.add_column("Mean z")
tab.add_column("Std dev z")
for cid, zs in history.items():
    m = statistics.mean(zs)
    s = statistics.pstdev(zs)
    tab.add_row(cid, f"{m:+.2f}", f"{s:.2f}")
print(tab)

with open("baseline_calibration.json","w") as f:
    json.dump({k:{"mean":float(statistics.mean(v)),
                  "std":float(statistics.pstdev(v))} for k,v in history.items()}, f, indent=2)
print("[dim]Saved baseline_calibration.json[/dim]")
