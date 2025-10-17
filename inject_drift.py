import json, sys
from pathlib import Path

CONFIG_DIR = Path("configs")
TARGET = "canary_2"   # which canary to drift
BACKUP_DIR = CONFIG_DIR / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

cfg_path = CONFIG_DIR / f"{TARGET}.json"
if not cfg_path.exists():
    print("Target config not found:", cfg_path)
    sys.exit(1)

cfg = json.load(open(cfg_path))
# backup current version
open(BACKUP_DIR / f"{TARGET}.json.bak", "w").write(json.dumps(cfg, indent=2))

# inject benign drift by modifying the prompt template
drift_phrase = " Also include the sentence: 'blue banana'."
base_template = cfg.get("template", "### Human: {prompt}\\n### Assistant:")
cfg["template"] = base_template + drift_phrase

json.dump(cfg, open(cfg_path, "w"), indent=2)
print(f"Injected drift into {cfg_path}. Backup saved.")
