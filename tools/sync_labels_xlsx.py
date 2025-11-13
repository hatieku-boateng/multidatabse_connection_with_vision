from pathlib import Path
import csv
import sys

def write_with_pandas(csv_path, xlsx_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df.to_excel(xlsx_path, index=False)

def write_with_openpyxl(csv_path, xlsx_path):
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            ws.append(row)
    wb.save(xlsx_path)

def main():
    # csv is located at ../id_cards/manifests/labels.csv relative to this script's parent
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / 'id_cards' / 'manifests' / 'labels.csv'
    xlsx_path = repo_root / 'id_cards' / 'manifests' / 'labels.xlsx'

    if not csv_path.exists():
        print('labels.csv not found at', csv_path)
        sys.exit(1)

    # Try pandas first, then fallback to openpyxl
    try:
        write_with_pandas(csv_path, xlsx_path)
        print('Wrote', xlsx_path, 'using pandas')
        return
    except Exception:
        pass

    try:
        write_with_openpyxl(csv_path, xlsx_path)
        print('Wrote', xlsx_path, 'using openpyxl')
        return
    except Exception as e:
        print('Failed to write Excel file. Install pandas or openpyxl. Error:', e)
        sys.exit(1)

if __name__ == '__main__':
    main()
