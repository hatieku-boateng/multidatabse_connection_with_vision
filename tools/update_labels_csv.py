import pandas as pd
from pathlib import Path

CURATED = Path("id_cards/curated")
CSV_PATH = Path("id_cards/manifests/labels.csv")
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

# Step 1: Scan folders
new_records = []
for cls_dir in CURATED.iterdir():
    if cls_dir.is_dir():
        label = cls_dir.name
        for fp in cls_dir.rglob("*"):
            if fp.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                new_records.append((fp.as_posix(), label))

new_df = pd.DataFrame(new_records, columns=["filepath", "label"])

# Step 2: If CSV exists, load and merge
if CSV_PATH.exists():
    old_df = pd.read_csv(CSV_PATH)
    combined = pd.concat([old_df, new_df]).drop_duplicates(subset=["filepath"], keep="last")
    df = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"✓ Added {len(df) - len(old_df)} new rows")
else:
    df = new_df
    print("✓ CSV did not exist before, created new file")

# Step 3: Save back
df.to_csv(CSV_PATH, index=False)
print(f"✓ Updated {CSV_PATH} with {len(df)} total rows")
