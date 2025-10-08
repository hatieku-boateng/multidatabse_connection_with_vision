import pandas as pd
from pathlib import Path

# Paths
CURATED = Path("id_cards/curated")
XLSX_PATH = Path("id_cards/manifests/labels.xlsx")  # file we want to create
XLSX_PATH.parent.mkdir(parents=True, exist_ok=True)

records = []

# Walk through curated subfolders
for cls_dir in CURATED.iterdir():
    if cls_dir.is_dir():
        label = cls_dir.name
        for fp in cls_dir.rglob("*"):
            if fp.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                records.append((fp.as_posix(), label))

# Make dataframe
df = pd.DataFrame(records, columns=["filepath", "label"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Create Excel file
df.to_excel(XLSX_PATH, index=False)

print(f"✓ Created Excel manifest at {XLSX_PATH} with {len(df)} rows")
print(df.head())
