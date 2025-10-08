import pandas as pd
from pathlib import Path

CURATED = Path("id_cards/curated")
CSV_PATH = Path("id_cards/manifests/labels.csv")
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

records = []
for cls_dir in CURATED.iterdir():
    if cls_dir.is_dir():
        label = cls_dir.name
        for fp in cls_dir.rglob("*"):
            if fp.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                records.append((fp.as_posix(), label))

df = pd.DataFrame(records, columns=["filepath", "label"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv(CSV_PATH, index=False)
print(f"✓ Overwritten {CSV_PATH} with {len(df)} rows")
print(df.head())
