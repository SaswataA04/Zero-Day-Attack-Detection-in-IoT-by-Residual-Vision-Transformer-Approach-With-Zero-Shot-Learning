"""Add a 10,000-row development block and label only unambiguous paper groups.
Run: python label_pilot.py "D:\\iot_23_training_extracted"
Requires prepared/pilot_raw_sorted.csv beside this script's folder.
No packages required. Writes new outputs; original pilot is preserved.
"""
import argparse
from collections import Counter
import csv
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import os

# Exact names from Table 2 of the supplied CZ-ResViT paper.
# These are the PAPER'S categories, not a claim about universal malware taxonomy.
MAPPING = {
    'DDoS': 'DDoS',
    'Okiru': 'Mirai',
    'Kenjiro': 'Mirai',
    'Hakai': 'Mirai',
    'Hajime': 'Mirai',
    'Muhstik': 'Mirai',
    'Attack': 'Web-based',
    'PartOfAHorizontalPortScan': 'Recon',
    'C&C': 'Recon',
}


def paper_class(row):
    binary = row['label'].strip().lower()
    if binary == 'benign':
        return 'Benign'
    if binary != 'malicious':
        raise ValueError(f'Unexpected binary label: {binary!r}')
    # Never guess how to assign compound labels such as Okiru-Attack.
    return MAPPING.get(row['detailed-label'].strip(), '')


def extra_records(path, maximum=10000):
    records = []
    with path.open(encoding='utf-8-sig', newline='') as source:
        reader = csv.reader(source, strict=True)
        raw = next(reader)
        groups = []
        for value in raw:
            names = value.strip().split()
            groups.append(names if names == ['tunnel_parents', 'label', 'detailed-label'] else [value.strip()])
        for values in reader:
            if not values:
                continue
            if len(records) == maximum:
                break
            if len(values) != len(groups):
                raise ValueError('Unexpected CSV row length in dataset6.csv.')
            row = {}
            for names, value in zip(groups, values):
                parts = value.split() if len(names) > 1 else [value]
                if len(parts) != len(names):
                    raise ValueError('Cannot parse combined labels.')
                row.update(zip(names, parts))
            row['label_original'] = row['label']
            row['label'] = row['label'].strip().lower()
            row['source_file'] = path.name
            row['source_row'] = str(len(records) + 1)
            timestamp = Decimal(row['ts'])
            if not timestamp.is_finite():
                raise ValueError('Non-finite timestamp.')
            records.append((timestamp, row))
    records.sort(key=lambda pair: pair[0])
    return [row for _, row in records]


def build(folder, project):
    prepared = project / 'prepared'
    pilot = prepared / 'pilot_raw_sorted.csv'
    target = prepared / 'pilot_labeled.csv'
    summary_path = prepared / 'pilot_label_summary.json'
    if target.exists() or summary_path.exists():
        raise ValueError('Outputs already exist. Move the previous labelled outputs before rerunning.')
    block = extra_records(folder / 'dataset6.csv')
    if not any(paper_class(row) == 'Mirai' for row in block):
        raise ValueError('No Table-2 Mirai-family records in this block. Original pilot is unchanged.')
    counts = Counter()
    unresolved = Counter()
    by_file = Counter()
    temporary = None
    try:
        with pilot.open(encoding='utf-8-sig', newline='') as source:
            reader = csv.DictReader(source)
            fields = reader.fieldnames
            required = {'ts', 'label', 'detailed-label', 'source_file', 'source_row', 'label_original'}
            if not fields or not required.issubset(fields) or len(set(fields)) != len(fields):
                raise ValueError('Unexpected pilot column layout.')
            with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='', dir=prepared,
                                             delete=False, suffix='.tmp') as out:
                temporary = Path(out.name)
                writer = csv.DictWriter(out, fieldnames=fields + ['paper_class'])
                writer.writeheader()

                def write(row):
                    if set(row) != set(fields) or any(value is None for value in row.values()):
                        raise ValueError('Malformed row or incompatible CSV schema.')
                    assigned = paper_class(row)
                    counts[assigned or 'UNRESOLVED'] += 1
                    if not assigned:
                        unresolved[row['detailed-label']] += 1
                    by_file[row['source_file']] += 1
                    writer.writerow(dict(row, paper_class=assigned))

                for row in reader:
                    if row.get('source_file') == 'dataset6.csv':
                        raise ValueError('dataset6.csv is already present; refusing to duplicate records.')
                    write(row)
                for row in block:
                    write(row)
        report = {
            'purpose': 'Five-class pipeline-development pilot; not a final research benchmark',
            'rows': sum(counts.values()),
            'paper_class_counts': dict(counts),
            'unresolved_detailed_labels': dict(unresolved),
            'rows_by_source_file': dict(by_file),
            'added_block': {'file': 'dataset6.csv', 'prefix_rows': len(block)},
            'notes': [
                'Mapping uses exact entries from Table 2 of the supplied paper.',
                'Unresolved rows are retained with blank paper_class; no sixth training class is implied.',
                'The added block is a file prefix sorted by timestamp, not guaranteed contiguous in real time.',
                'This selection deliberately adds class coverage and is not statistically representative.',
                'Do not create final temporal windows or benchmark splits from this pilot.',
                'Missing markers remain untouched. Labels and provenance are not model inputs.',
                'Class mapping precedence for compound labels and source provenance remain unresolved.',
            ],
        }
        os.replace(temporary, target)
        temporary = None
        summary_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        return report, summary_path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    try:
        report, path = build(args.folder, Path(__file__).resolve().parent)
        print(json.dumps(report, indent=2))
        print(f'Upload: {path}')
        return 0
    except (OSError, ValueError, ArithmeticError, KeyError, csv.Error, StopIteration) as error:
        print(f'Stopped: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
