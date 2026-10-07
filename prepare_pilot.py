"""Stage smaller CSVs for pipeline development; does not train a model.
Run: python prepare_pilot.py "D:\\iot_23_training_extracted"
Requires the completed csv_audit.json beside this script (or --audit PATH).
No extra packages. Original inputs, missing markers and detailed labels stay intact.
"""
import argparse
from collections import Counter
import csv
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import tempfile


def read_records(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle, strict=True)
        raw = next(reader)
        groups = []
        for name in raw:
            parts = name.strip().split()
            groups.append(parts if parts in [
                ['tunnel_parents', 'label', 'detailed-label'],
                ['tunnel_parents', 'label', 'det_label'],
            ] else [name.strip()])
        fields = [name for group in groups for name in group]
        if len(set(fields)) != len(fields) or not {'ts', 'label'} <= set(fields):
            raise ValueError(f'{path.name}: duplicate fields or missing ts/label.')
        records = []
        for values in reader:
            if not values:
                continue
            if len(values) != len(raw):
                raise ValueError(f'{path.name}: malformed CSV record at line {reader.line_num}.')
            record = {}
            for names, value in zip(groups, values):
                pieces = value.split() if len(names) > 1 else [value]
                if len(pieces) != len(names):
                    raise ValueError(f'{path.name}: malformed combined label field.')
                record.update(zip(names, pieces))
            original_label = record['label']
            label = original_label.strip().lower()
            if label not in {'benign', 'malicious'}:
                raise ValueError(f'{path.name}: unexpected binary label {original_label!r}.')
            record['label_original'] = original_label
            record['label'] = label
            record['source_file'] = path.name
            record['source_row'] = len(records) + 1
            timestamp = Decimal(record['ts'])
            if not timestamp.is_finite():
                raise ValueError(f'{path.name}: non-finite timestamp.')
            records.append((timestamp, record))
        # Stable sorting preserves original row order when timestamps tie.
        records.sort(key=lambda item: item[0])
        return fields, records


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--audit', type=Path, default=project / 'csv_audit.json')
    args = parser.parse_args()
    temporary = None
    try:
        audit = json.loads(args.audit.read_text(encoding='utf-8-sig'))
        files = audit.get('files', [])
        if audit.get('mode') != 'full' or not files or any(f['status'] != 'complete_file' for f in files):
            raise ValueError('Use the successful FULL audit report for this folder.')
        chosen = [f for f in files if 0 < f['rows_inspected'] <= 200000]
        if not chosen:
            raise ValueError('No files have at most 200,000 rows in this audit.')
        destination = project / 'prepared'
        destination.mkdir(exist_ok=True)
        target = destination / 'pilot_raw_sorted.csv'
        report_target = destination / 'pilot_summary.json'
        if target.exists() or report_target.exists():
            raise ValueError('Pilot output already exists. Move previous pilot outputs before rerunning.')
        counts = Counter()
        details = Counter()
        summaries = []
        expected_fields = None
        # Stage one file at a time. Do not load the 75-million-row collection.
        with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8',
                                         dir=destination, suffix='.tmp', delete=False) as output:
            temporary = Path(output.name)
            writer = None
            for entry in chosen:
                name = entry['file']
                if Path(name).name != name:
                    raise ValueError('Audit filenames must be plain filenames.')
                path = args.folder / name
                print(f'Reading and sorting {name} ...', flush=True)
                fields, records = read_records(path)
                if len(records) != entry['rows_inspected'] or fields != entry['columns']:
                    raise ValueError(f'{name}: contents do not match the full audit shape.')
                if expected_fields is None:
                    expected_fields = fields
                    writer = csv.DictWriter(output, fieldnames=fields + ['label_original', 'source_file', 'source_row'])
                    writer.writeheader()
                elif fields != expected_fields:
                    raise ValueError(f'{name}: column layout differs from other pilot files.')
                per_file = Counter(record['label'] for _, record in records)
                expected_labels = Counter()
                for label, value in entry['label_counts_in_inspected_rows']['label'].items():
                    expected_labels[label.strip().lower()] += value
                if per_file != expected_labels:
                    raise ValueError(f'{name}: labels do not match the full audit.')
                for _, record in records:
                    writer.writerow(record)
                    detail = record.get('detailed-label', record.get('det_label', ''))
                    details[detail] += 1
                counts.update(per_file)
                summaries.append({'file': name, 'rows': len(records), 'labels': dict(per_file)})
                print(f'  Staged {len(records):,} records.', flush=True)
        report = {
            'purpose': 'Pipeline-development pilot, not a representative benchmark or paper reproduction',
            'selection': 'All audited CSVs with 1 to 200,000 rows; entire selected files retained',
            'rows': sum(counts.values()), 'label_counts': dict(counts),
            'detailed_label_counts': dict(details), 'files': summaries,
            'notes': [
                'Sorted by ts within each source file; files are not merged into one temporal session.',
                'Source filenames are provisional grouping identifiers, not verified capture identifiers.',
                'Original labels are retained in label_original; label is lowercased.',
                'No imputation, scaling, feature selection, windowing, deduplication or training performed.',
                'All labels and provenance columns must remain outside model input features.',
                'Missing values remain raw markers. A missing detailed label on benign traffic is not a missing binary target.',
                'Split by verified capture/session or time boundaries before constructing training windows.',
                'The paper five-class mapping is still unresolved; this pilot does not redefine that task.',
            ],
        }
        os.replace(temporary, target)
        temporary = None
        report_target.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f"Saved {report['rows']:,} records to {target}")
        print(f'Upload this summary: {report_target}')
        return 0
    except (OSError, ValueError, KeyError, csv.Error, InvalidOperation, StopIteration) as error:
        print(f'Stopped: {error}')
        return 1
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
