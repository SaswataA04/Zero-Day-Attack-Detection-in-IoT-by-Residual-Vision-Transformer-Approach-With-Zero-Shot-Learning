"""Step 2: put the prepared IoT-23 records into chronological order.

Run from the same project folder used for Step 1:
    python step02_sort_dataset7.py

Inputs (created by Step 1):
    prepared/dataset7_step01/flows.csv
    prepared/dataset7_step01/step01_summary.json
Outputs:
    prepared/dataset7_step02/flows_sorted.csv
    prepared/dataset7_step02/step02_summary.json

No extra packages are required. SQLite is included with Python and provides a
disk-backed sort, so millions of traffic records do not have to fit in RAM.
The temporary sorting database is removed when the run finishes.

Research protocol:
    Timestamp sorting is our explicit choice for establishing temporal order.
    The paper does not specify its exact raw-record sorting procedure.
    The paper's random 80:20 image split is a later operation. This script does
    not replace that protocol with chronological train/test partitions.

This step preserves every stored field value, including missing markers, labels,
and source-row provenance. Equal timestamps are ordered by numeric source_row.
No imputation, category encoding, feature selection, scaling, or training is
performed here. No rows are sampled, balanced, or removed as duplicates.
"""

import argparse
import csv
import json
import math
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from contextlib import closing
from itertools import islice
from pathlib import Path


REQUIRED_METADATA = [
    "label", "detailed-label", "label_original", "target", "source_file", "source_row",
]
TARGET_MAP = {"benign": "0", "malicious": "1"}


def read_step01_summary(path):
    """Read the previous step's feature list and expected totals."""
    with path.open("r", encoding="utf-8") as source:
        report = json.load(source)
    if report.get("step") != "01_prepare_single_iot23_csv":
        raise ValueError("Expected the summary produced by step01_prepare_dataset7.py.")
    features = report.get("candidate_features", [])
    metadata = report.get("metadata_columns_not_for_model_input", [])
    if len(features) != 17 or "ts" not in features:
        raise ValueError("Step 1 must provide 17 candidate features, including ts.")
    if metadata != REQUIRED_METADATA or set(features) & set(metadata):
        raise ValueError("Unexpected feature/metadata definitions in the Step 1 summary.")
    columns = features + metadata
    if len(columns) != len(set(columns)):
        raise ValueError("Repeated columns in the Step 1 summary.")
    return report, columns


def sort_dataset(input_path, output_dir, batch_size=25_000):
    """Validate Step 1, load a temporary database, and stream sorted CSV output."""
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    summary_path = input_path.parent / "step01_summary.json"
    if not input_path.is_file():
        raise FileNotFoundError(f"Prepared CSV not found: {input_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Step 1 summary not found: {summary_path}")

    previous_report, expected_columns = read_step01_summary(summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_path = output_dir / "flows_sorted.csv"
    report_path = output_dir / "step02_summary.json"
    partial_csv = output_dir / "flows_sorted.csv.partial"
    partial_report = output_dir / "step02_summary.json.partial"
    for path in (sorted_path, report_path, partial_csv, partial_report):
        if path.exists():
            raise FileExistsError(f"{path} already exists. Choose a new --output-dir to rerun.")

    started = time.perf_counter()
    original_stat = input_path.stat()
    input_labels = Counter()
    output_labels = Counter()
    rows_read = rows_written = 0
    source_name = None
    minimum_ts = maximum_ts = previous_key = None
    equal_timestamp_pairs = 0
    owned_temporary_outputs = []

    print(f"Input: {input_path}", flush=True)
    print("1/3 Loading a temporary sorting database...", flush=True)

    try:
        # closing() is required: SQLite's transaction context manager alone does
        # not close its file handle. Closing first also permits cleanup on Windows.
        with tempfile.TemporaryDirectory(prefix="sort_work_", dir=output_dir) as work:
            with closing(sqlite3.connect(str(Path(work) / "records.sqlite3"))) as db:
                db.execute("PRAGMA temp_store=FILE")
                db.execute("PRAGMA cache_size=-65536")  # Approximately 64 MiB page cache.

                # c0, c1, ... are internally generated SQL column names. Every
                # original CSV cell is stored as TEXT to preserve its exact value.
                sql_columns = [f"c{i}" for i in range(len(expected_columns))]
                definitions = ", ".join(f"{column} TEXT NOT NULL" for column in sql_columns)
                db.execute(
                    "CREATE TABLE flows (source_id INTEGER PRIMARY KEY, "
                    f"time_key REAL NOT NULL, {definitions})"
                )
                placeholders = ", ".join("?" for _ in range(len(expected_columns) + 2))
                insert_sql = f"INSERT INTO flows VALUES ({placeholders})"
                positions = {name: index for index, name in enumerate(expected_columns)}

                with input_path.open("r", encoding="utf-8-sig", newline="") as source:
                    reader = csv.reader(source, strict=True)
                    if next(reader, None) != expected_columns:
                        raise ValueError("CSV header does not match the Step 1 summary.")

                    while True:
                        batch = list(islice(reader, batch_size))
                        if not batch:
                            break
                        database_rows = []
                        for values in batch:
                            position = rows_read + 1
                            if len(values) != len(expected_columns):
                                raise ValueError(f"Prepared record {position}: incorrect number of fields.")
                            source_id = int(values[positions["source_row"]])
                            if source_id < 1:
                                raise ValueError(f"Prepared record {position}: source_row must be positive.")
                            timestamp = float(values[positions["ts"]])
                            if not math.isfinite(timestamp):
                                raise ValueError(f"Prepared record {position}: invalid timestamp.")

                            label = values[positions["label"]]
                            target = values[positions["target"]]
                            if label not in TARGET_MAP or target != TARGET_MAP[label]:
                                raise ValueError(f"Prepared record {position}: inconsistent label/target.")
                            current_source = values[positions["source_file"]]
                            if source_name is None:
                                source_name = current_source
                            if not current_source or current_source != source_name:
                                raise ValueError("Step 2 expects records from exactly one source file.")

                            # The primary key rejects repeated provenance IDs; it
                            # does not silently discard repeated-looking traffic.
                            database_rows.append((source_id, timestamp, *values))
                            input_labels[label] += 1
                            rows_read += 1

                        db.executemany(insert_sql, database_rows)
                        db.commit()
                        if rows_read % 250_000 == 0:
                            print(f"  Loaded {rows_read:,} rows...", flush=True)

                if rows_read == 0 or rows_read != previous_report["rows_written"]:
                    raise ValueError("Input row count differs from Step 1; check that the files match.")
                if dict(input_labels) != previous_report["label_counts"]:
                    raise ValueError("Input label counts differ from Step 1.")
                final_stat = input_path.stat()
                if (original_stat.st_size, original_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
                    raise ValueError("The input CSV changed while being read.")

                print(f"  Loaded all {rows_read:,} rows.", flush=True)
                print("2/3 Sorting by numeric timestamp, then numeric source row...", flush=True)
                db.execute("CREATE INDEX temporal_order ON flows(time_key, source_id)")
                db.commit()

                print("3/3 Writing and checking the sorted CSV...", flush=True)
                query = (
                    "SELECT time_key, source_id, " + ", ".join(sql_columns)
                    + " FROM flows ORDER BY time_key, source_id"
                )
                with partial_csv.open("x", encoding="utf-8", newline="") as destination:
                    owned_temporary_outputs.append(partial_csv)
                    writer = csv.writer(destination)
                    writer.writerow(expected_columns)
                    for stored in db.execute(query):
                        timestamp, source_id = stored[:2]
                        values = stored[2:]
                        key = (timestamp, source_id)
                        if previous_key is not None:
                            if key <= previous_key:
                                raise ValueError("Sorted record order verification failed.")
                            if timestamp == previous_key[0]:
                                equal_timestamp_pairs += 1
                        previous_key = key
                        if minimum_ts is None:
                            minimum_ts = values[positions["ts"]]
                        maximum_ts = values[positions["ts"]]
                        writer.writerow(values)
                        output_labels[values[positions["label"]]] += 1
                        rows_written += 1
                        if rows_written % 250_000 == 0:
                            print(f"  Wrote {rows_written:,} sorted rows...", flush=True)

                if rows_written != rows_read or output_labels != input_labels:
                    raise ValueError("Row-count or label-count preservation check failed.")

        report = {
            "step": "02_sort_single_iot23_csv",
            "input_file": str(input_path),
            "input_size_bytes": original_stat.st_size,
            "step01_summary": str(summary_path),
            "sorted_file": str(sorted_path),
            "source_file": source_name,
            "rows_read": rows_read,
            "rows_written": rows_written,
            "rows_dropped": 0,
            "label_counts": dict(output_labels),
            "candidate_features": previous_report["candidate_features"],
            "metadata_columns_not_for_model_input": REQUIRED_METADATA,
            "sort_keys": ["numeric ts ascending", "numeric source_row ascending"],
            "timestamps": {
                "minimum_original_text": minimum_ts,
                "maximum_original_text": maximum_ts,
                "backward_steps_after_sorting": 0,
                "adjacent_equal_timestamp_pairs": equal_timestamp_pairs,
                "order_verified_during_export": True,
            },
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "python_version": sys.version.split()[0],
            "notes": [
                "Sorting establishes temporal order; it is an explicit implementation choice.",
                "Every original source-row ID and traffic value is retained.",
                "Equal timestamps are preserved, with numeric source_row as the tie breaker.",
                "No temporal train/test split was introduced; the paper uses random image-level splitting.",
                "No windows, numeric preprocessing, RF-SHAP selection or model training yet.",
                "Chronological order does not establish verified capture/session boundaries.",
            ],
        }
        with partial_report.open("x", encoding="utf-8") as destination:
            owned_temporary_outputs.append(partial_report)
            json.dump(report, destination, indent=2, ensure_ascii=False, allow_nan=False)
            destination.write("\n")
        partial_csv.rename(sorted_path)
        partial_report.rename(report_path)
    finally:
        for path in owned_temporary_outputs:
            if path.exists():
                path.unlink()

    print("\nSTEP 2 COMPLETE", flush=True)
    print(f"Rows read:    {rows_read:,}")
    print(f"Rows written: {rows_written:,}")
    print(f"Benign:    {output_labels['benign']:,}")
    print(f"Malicious: {output_labels['malicious']:,}")
    print("Backward timestamp steps after sorting: 0")
    print(f"Sorted CSV: {sorted_path}")
    print(f"Summary:    {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Step 2: sort Step 1 records chronologically.")
    parser.add_argument("--input", type=Path, default=Path("prepared/dataset7_step01/flows.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("prepared/dataset7_step02"))
    parser.add_argument("--batch-size", type=int, default=25_000)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    try:
        sort_dataset(args.input, args.output_dir, args.batch_size)
    except (OSError, ValueError, KeyError, OverflowError, csv.Error, sqlite3.Error) as error:
        print(f"\nSTEP 2 FAILED: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped before completion. No success summary was created.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
