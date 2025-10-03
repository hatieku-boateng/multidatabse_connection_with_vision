import pandas as pd
from pathlib import Path

# Paths
CURATED = Path("id_cards/curated")
XLSX_PATH = Path("id_cards/manifests/labels.xlsx")
XLSX_PATH.parent.mkdir(parents=True, exist_ok=True)

# Step 1: Build new records from folder
new_records = []
for cls_dir in CURATED.iterdir():
    if cls_dir.is_dir():
        label = cls_dir.name
        for fp in cls_dir.rglob("*"):
            if fp.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                new_records.append((fp.as_posix(), label))

new_df = pd.DataFrame(new_records, columns=["filepath", "label"])

# Step 2: If old Excel exists, load it
if XLSX_PATH.exists():
    old_df = pd.read_excel(XLSX_PATH)
    # Find new rows not already in Excel
    combined = pd.concat([old_df, new_df]).drop_duplicates(subset=["filepath"], keep="last")
    df = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"✓ Added {len(df) - len(old_df)} new rows")
else:
    df = new_df
    print("✓ Excel did not exist before, created new file")

# Step 3: Save updated Excel
df.to_excel(XLSX_PATH, index=False)
print(f"✓ Updated {XLSX_PATH} with {len(df)} total rows")
