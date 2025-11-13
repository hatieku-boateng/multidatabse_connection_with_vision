import os
import pandas as pd
from pathlib import Path

# Paths (make paths relative to the repository root where this script lives)
REPO_ROOT = Path(__file__).resolve().parent.parent
CURATED = REPO_ROOT / "id_cards" / "curated"
CSV_PATH = REPO_ROOT / "id_cards" / "manifests" / "labels.csv"
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

# Step 1: Scan folders (make sure curated exists)
if not CURATED.exists():
    print(f"Error: curated folder not found at {CURATED}. Create it and add images before running this script.")
    raise SystemExit(2)

new_records = []
for cls_dir in sorted(CURATED.iterdir()):
    if cls_dir.is_dir():
        label = cls_dir.name
        for fp in cls_dir.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                # make path relative to the manifest folder (id_cards/manifests)
                rel = os.path.relpath(fp, start=CSV_PATH.parent).replace('\\', '/')
                new_records.append((rel, label))

new_df = pd.DataFrame(new_records, columns=["filepath", "label"])

# Step 2: If CSV exists, load and merge
if CSV_PATH.exists():
    old_df = pd.read_csv(CSV_PATH)
    combined = pd.concat([old_df, new_df]).drop_duplicates(subset=["filepath"], keep="last")
    df = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    try:
        added = len(df) - len(old_df)
    except Exception:
        added = len(df)
    print(f"✓ Added {added} new rows")
else:
    df = new_df
    print("✓ CSV did not exist before, created new file")

# Step 3: Save back
df.to_csv(CSV_PATH, index=False)
print(f"✓ Updated {CSV_PATH} with {len(df)} total rows")
