import json, sys
from pathlib import Path

CONFIG_DIR = Path("configs")
BACKUP_DIR = CONFIG_DIR / "backups"
TARGET = "canary_2"

bak = BACKUP_DIR / f"{TARGET}.json.bak"
if not bak.exists():
    print("No backup found for", TARGET)
    sys.exit(1)

# restore original file
json.dump(json.load(open(bak)), open(CONFIG_DIR / f"{TARGET}.json", "w"), indent=2)
print(f"Restored {TARGET} from backup.")
