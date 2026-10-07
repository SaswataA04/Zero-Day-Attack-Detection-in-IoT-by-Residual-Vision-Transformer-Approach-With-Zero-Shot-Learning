"""Audit IoT-23 CSV files without modifying them. No extra packages required.

Example:
    python inspect_csv_folder.py "D:\\iot_23_training_extracted"
Default: inspect up to 10,000 rows per file. Use --full for exact full-file counts.
Output: csv_audit.json beside this script. Sample counts are NOT dataset totals.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def audit_file(path, limit):
    labels = {}
    missing = Counter()
    empty = Counter()
    first = {}
    changed = set()
    count = 0
    sampled = False
    timestamp_min = timestamp_max = previous_ts = None
    timestamp_errors = 0
    out_of_order = 0
    with path.open('r', encoding='utf-8-sig', newline='') as source:
        reader = csv.reader(source, strict=True)
        raw_header = next(reader, None)
        if not raw_header:
            raise ValueError('Empty file.')
        raw_header = [name.strip() for name in raw_header]
        columns = []
        groups = []
        for name in raw_header:
            parts = name.split()
            if parts in [
                ['tunnel_parents', 'label', 'detailed-label'],
                ['tunnel_parents', 'label', 'det_label'],
            ]:
                groups.append(parts)
                columns.extend(parts)
            else:
                groups.append([name])
                columns.append(name)
        if any(not name for name in columns) or len(set(columns)) != len(columns):
            raise ValueError('Blank or duplicate column names; inspect the header.')
        label_names = [name for name in columns if 'label' in name.lower()]
        if not label_names:
            raise ValueError('No label columns found.')
        labels = {name: Counter() for name in label_names}
        for values in reader:
            if not values:
                continue
            if limit is not None and count >= limit:
                sampled = True
                break
            if len(values) != len(raw_header):
                raise ValueError(f'CSV line {reader.line_num}: expected {len(raw_header)} cells, got {len(values)}.')
            record = {}
            for names, value in zip(groups, values):
                parts = value.split() if len(names) > 1 else [value.strip()]
                if len(parts) != len(names):
                    raise ValueError(f'CSV line {reader.line_num}: cannot split combined label field {value!r}.')
                record.update(zip(names, parts))
            count += 1
            for name, value in record.items():
                if value.lower() in {'', '-', 'nan', 'null', 'none'}:
                    missing[name] += 1
                elif value == '(empty)':
                    empty[name] += 1
                if name not in first:
                    first[name] = value
                elif first[name] != value:
                    changed.add(name)
                if name in labels:
                    labels[name][value] += 1
            if 'ts' in record:
                try:
                    ts = float(record['ts'])
                    if not (-float('inf') < ts < float('inf')):
                        raise ValueError('Non-finite timestamp')
                    timestamp_min = ts if timestamp_min is None else min(timestamp_min, ts)
                    timestamp_max = ts if timestamp_max is None else max(timestamp_max, ts)
                    if previous_ts is not None and ts < previous_ts:
                        out_of_order += 1
                    previous_ts = ts
                except ValueError:
                    timestamp_errors += 1
    if count == 0:
        raise ValueError('Header present but no records found.')
    return {
        'file': path.name,
        'status': 'sampled_prefix' if sampled else 'complete_file',
        'rows_inspected': count,
        'original_column_count': len(raw_header),
        'parsed_column_count': len(columns),
        'columns': columns,
        'split_combined_fields': [group for group in groups if len(group) > 1],
        'label_counts_in_inspected_rows': {name: dict(counter) for name, counter in labels.items()},
        'missing_marker_counts': {name: missing[name] for name in columns},
        'empty_set_counts': dict(empty),
        'constant_columns_in_inspected_rows': [name for name in columns if name not in changed],
        'timestamps': {
            'present': 'ts' in columns,
            'minimum_inspected': timestamp_min,
            'maximum_inspected': timestamp_max,
            'invalid_inspected': timestamp_errors,
            'backward_steps_inspected': out_of_order,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, help='CSV file or folder containing CSV files')
    parser.add_argument('--full', action='store_true', help='Read every row; large files may take time')
    parser.add_argument('--limit', type=int, default=10000, help='Prefix rows per file (default 10000)')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    if args.path.is_dir():
        files = sorted(p for p in args.path.iterdir() if p.is_file() and p.suffix.lower() == '.csv')
    elif args.path.is_file() and args.path.suffix.lower() == '.csv':
        files = [args.path]
    else:
        parser.error('Provide an existing CSV file or directory.')
    if not files:
        parser.error('No CSV files found directly inside that folder.')
    results = []
    for path in files:
        print(f'Inspecting {path.name} ...', flush=True)
        try:
            result = audit_file(path, None if args.full else args.limit)
            print(f"  {result['rows_inspected']:,} rows; {result['status']}; "
                  f"labels: {result['label_counts_in_inspected_rows']}", flush=True)
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            result = {'file': path.name, 'status': 'error', 'error': str(error)}
            print(f'  ERROR: {error}', flush=True)
        results.append(result)
    report = {
        'mode': 'full' if args.full else 'prefix_sample',
        'maximum_rows_per_file': None if args.full else args.limit,
        'notes': [
            'Original CSV files are not modified. Label strings retain their case.',
            'Prefix samples cannot establish full class coverage or full label proportions.',
            'Merged label fields are split only in memory.',
            'Missing-marker counts describe stored tokens, not a prescribed cleaning rule.',
            'Retain source filenames for grouping; verify whether files correspond to original captures.',
            'Keep labels out of model inputs. Audit provenance and class mapping before training.',
        ],
        'files': results,
    }
    destination = Path(__file__).resolve().parent / 'csv_audit.json'
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'Report saved: {destination}')
    return int(any(result['status'] == 'error' for result in results))


if __name__ == '__main__':
    raise SystemExit(main())
