from pathlib import Path
import csv
import sys

def main():
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / 'id_cards' / 'manifests' / 'labels.csv'

    if not csv_path.exists():
        print('labels.csv not found at', csv_path)
        sys.exit(2)

    missing = []
    counts = {}
    total = 0

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            rel = row.get('filepath')
            label = row.get('label', '')
            # Resolve path relative to the manifest directory
            candidate = (csv_path.parent / Path(rel)).resolve()
            if not candidate.exists():
                missing.append(rel)
            counts[label] = counts.get(label, 0) + 1

    print(f'Total entries in manifest: {total}')
    print('Class counts:')
    for k, v in sorted(counts.items()):
        print(f'  {k}: {v}')

    if missing:
        print('\nMissing files (listed as in manifest):')
        for p in missing:
            print(' -', p)
        print(f'\nMissing count: {len(missing)}')
        sys.exit(1)
    else:
        print('\nAll files referenced in the manifest exist.')
        sys.exit(0)

if __name__ == '__main__':
    main()
