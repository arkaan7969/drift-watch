import json, itertools, statistics
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from rich.table import Table
from rich import print

# === Load responses ===
responses = json.load(open("canary_results.json"))
ids = [r["id"] for r in responses]
texts = [r["response"] for r in responses]

print(f"Loaded {len(texts)} canary responses")

# === Embed with sentence-transformers ===
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embs = model.encode(texts, normalize_embeddings=True)

# === Compute pairwise cosine similarities ===
n = len(embs)
sim = [[0.0]*n for _ in range(n)]
for i, j in itertools.combinations(range(n), 2):
    s = 1 - cosine(embs[i], embs[j])
    sim[i][j] = sim[j][i] = s

# === Consensus score per canary ===
consensus = []
for i in range(n):
    others = [sim[i][k] for k in range(n) if k != i]
    consensus.append(sum(others)/len(others))

# === z-scores ===
mu = statistics.mean(consensus)
sd = statistics.pstdev(consensus) or 1e-6
zs = [(v - mu)/sd for v in consensus]

# === Pretty table ===
tab = Table(title="Canary Consensus Similarity")
tab.add_column("Canary")
tab.add_column("Consensus", justify="right")
tab.add_column("z-score", justify="right")
tab.add_column("Status", justify="center")

for i, cid in enumerate(ids):
    status = "[green]✓ OK[/green]" if zs[i] >= -1 else "[red]⚠ Anomaly[/red]"
    tab.add_row(cid, f"{consensus[i]:.3f}", f"{zs[i]:+.2f}", status)
print(tab)

# === Summary ===
flagged = [ids[i] for i, z in enumerate(zs) if z < -1]
if flagged:
    print(f"[red]Flagged anomalies:[/red] {flagged}")
else:
    print("[green]No anomalies detected[/green]")

# === Save results (cast to Python floats) ===
out = {
    "ids": ids,
    "consensus": [float(v) for v in consensus],
    "zscores": [float(z) for z in zs]
}
with open("consensus_scores.json", "w") as f:
    json.dump(out, f, indent=2)
print("[dim]Saved to consensus_scores.json[/dim]")
