#!/usr/bin/env python
"""
STEP 8A — AUDIT AN EXTERNAL ZERO-DAY DATASET (CICIoT2023)

Purpose
-------
Before creating contour images or running the frozen IoT-23 model on CICIoT2023,
inspect the actual extracted CSV files and record:

* files discovered
* schemas / column consistency
* total rows
* likely label column
* full label distribution
* numeric/non-numeric columns
* missing/non-finite counts for numeric columns
* whether a benign class exists
* candidate binary mapping information

This is READ-ONLY:
* no model loading
* no training
* no feature selection
* no normalization
* no contour generation
* no test/zero-day predictions
* no source-file modification

Recommended command
-------------------
If CICIoT2023 has been extracted into:
    data\\CICIoT2023\\

run:

    python step08a_audit_ciciot2023.py `
      --input "data\\CICIoT2023" `
      --output-dir "diagnostics\\ciciot2023_step08a_audit"

You may also point --input to a single CSV file.

Outputs
-------
step08a_summary.json
file_inventory.csv
label_counts.csv
column_profiles.csv

Notes
-----
CICIoT2023 distributions can be very large. This script streams CSV files in
chunks rather than loading the complete dataset into memory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as error:
    raise SystemExit(
        "This audit requires numpy and pandas.\n"
        "Install with: python -m pip install numpy pandas\n"
        f"Original error: {error}"
    ) from error


VERSION = 1

# Common spellings used by CIC-style CSV exports.
LABEL_CANDIDATES = (
    "label",
    "Label",
    "LABEL",
    "attack",
    "Attack",
    "attack_type",
    "Attack_Type",
    "Attack Type",
    "type",
    "Type",
    "category",
    "Category",
    "traffic_type",
    "Traffic_Type",
    "Traffic Type",
)

BENIGN_TOKENS = {
    "benign",
    "normal",
    "normal traffic",
    "benigntraffic",
    "benign_traffic",
}


def atomic_json(path: Path, content):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(content, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_csvs(path: Path):
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError("--input file must be a .csv file.")
        return [path.resolve()]

    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    files = sorted(
        p.resolve()
        for p in path.rglob("*.csv")
        if p.is_file()
    )

    if not files:
        raise FileNotFoundError(
            f"No CSV files found under: {path}"
        )

    return files


def normalized_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def find_label_column(columns):
    columns = list(columns)

    for candidate in LABEL_CANDIDATES:
        if candidate in columns:
            return candidate

    lower_map = {
        str(column).strip().lower(): column
        for column in columns
    }

    for candidate in LABEL_CANDIDATES:
        key = candidate.strip().lower()
        if key in lower_map:
            return lower_map[key]

    return None


def write_csv(path: Path, fieldnames, rows):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


def audit(args):
    input_path = args.input.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()

    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(
                f"Output folder already exists and is not empty:\n{output}\n"
                "Use a new output directory or intentionally remove the old audit."
            )
    else:
        output.mkdir(parents=True)

    files = discover_csvs(input_path)

    print("STEP 8A — CICIoT2023 ZERO-DAY DATA AUDIT", flush=True)
    print(f"Input: {input_path}", flush=True)
    print(f"CSV files discovered: {len(files):,}", flush=True)
    print("READ-ONLY audit: no training, no model inference, no data modification.", flush=True)

    started = time.perf_counter()

    inventory_rows = []
    label_counter = Counter()

    schema_counter = Counter()
    schema_examples = {}

    global_columns = set()
    label_columns_seen = Counter()

    column_rows = defaultdict(int)
    column_missing = defaultdict(int)
    column_numeric_valid = defaultdict(int)
    column_numeric_invalid = defaultdict(int)
    column_nonfinite = defaultdict(int)
    column_minimum = {}
    column_maximum = {}

    total_rows = 0
    full_hashes = args.hash_files

    expected_label_column = args.label_column

    for file_number, path in enumerate(files, start=1):
        print(
            f"\n[{file_number}/{len(files)}] {path.name}",
            flush=True,
        )

        stat = path.stat()

        try:
            header = pd.read_csv(
                path,
                nrows=0,
                low_memory=False,
            )
        except Exception as exc:
            raise ValueError(
                f"Could not read header from {path}: {exc}"
            ) from exc

        columns = [str(c) for c in header.columns]
        if not columns:
            raise ValueError(f"No columns found in {path}")

        schema_key = tuple(columns)
        schema_counter[schema_key] += 1
        schema_examples.setdefault(schema_key, str(path))
        global_columns.update(columns)

        if expected_label_column is not None:
            label_column = expected_label_column
            if label_column not in columns:
                raise ValueError(
                    f"Requested label column {label_column!r} is absent in {path.name}."
                )
        else:
            label_column = find_label_column(columns)

        if label_column is not None:
            label_columns_seen[label_column] += 1

        file_rows = 0
        file_labels = Counter()

        try:
            reader = pd.read_csv(
                path,
                chunksize=args.chunk_size,
                low_memory=False,
            )

            for chunk_number, frame in enumerate(reader, start=1):
                rows = len(frame)
                file_rows += rows
                total_rows += rows

                for name in columns:
                    series = frame[name]

                    missing_mask = series.isna()
                    missing = int(missing_mask.sum())

                    column_rows[name] += rows
                    column_missing[name] += missing

                    # Audit whether values are numeric-compatible without deciding
                    # the final Step-8 feature set yet.
                    numeric = pd.to_numeric(
                        series,
                        errors="coerce",
                    )

                    nonmissing_original = ~missing_mask
                    converted_valid = numeric.notna()

                    valid_count = int(
                        (nonmissing_original & converted_valid).sum()
                    )
                    invalid_count = int(
                        (nonmissing_original & ~converted_valid).sum()
                    )

                    column_numeric_valid[name] += valid_count
                    column_numeric_invalid[name] += invalid_count

                    if valid_count:
                        values = numeric[
                            nonmissing_original & converted_valid
                        ].to_numpy(dtype=np.float64)

                        finite_mask = np.isfinite(values)
                        nonfinite = int(
                            np.count_nonzero(~finite_mask)
                        )
                        column_nonfinite[name] += nonfinite

                        finite_values = values[finite_mask]

                        if len(finite_values):
                            low = float(finite_values.min())
                            high = float(finite_values.max())

                            column_minimum[name] = (
                                low
                                if name not in column_minimum
                                else min(column_minimum[name], low)
                            )
                            column_maximum[name] = (
                                high
                                if name not in column_maximum
                                else max(column_maximum[name], high)
                            )

                if label_column is not None:
                    values = frame[label_column].map(normalized_text)
                    counts = values.value_counts(dropna=False)

                    for value, count in counts.items():
                        text = str(value).strip()
                        file_labels[text] += int(count)
                        label_counter[text] += int(count)

                if (
                    chunk_number == 1
                    or chunk_number % args.log_every_chunks == 0
                ):
                    print(
                        f"  processed {file_rows:,} rows from this file; "
                        f"{total_rows:,} total",
                        flush=True,
                    )

        except Exception as exc:
            raise ValueError(
                f"Failed while reading {path}: {exc}"
            ) from exc

        inventory_rows.append(
            {
                "file": str(path),
                "bytes": stat.st_size,
                "rows": file_rows,
                "columns": len(columns),
                "label_column": label_column or "",
                "distinct_labels_in_file": len(file_labels),
                "sha256": (
                    sha256_file(path)
                    if full_hashes
                    else ""
                ),
            }
        )

        print(
            f"  rows={file_rows:,}; columns={len(columns)}; "
            f"label_column={label_column!r}; "
            f"distinct_labels={len(file_labels)}",
            flush=True,
        )

    if not label_columns_seen:
        raise ValueError(
            "No likely label column was found. "
            "Rerun with --label-column \"EXACT_COLUMN_NAME\" after inspecting the headers."
        )

    if len(label_columns_seen) > 1:
        print(
            "\nWARNING: multiple label-column spellings were detected:",
            dict(label_columns_seen),
            flush=True,
        )

    # Determine the dominant/common label-column spelling for reporting.
    dominant_label_column = label_columns_seen.most_common(1)[0][0]

    normalized_labels = {
        label.strip().lower(): count
        for label, count in label_counter.items()
    }

    benign_count = sum(
        count
        for label, count in normalized_labels.items()
        if label in BENIGN_TOKENS
    )
    malicious_count = total_rows - benign_count

    label_rows = [
        {
            "label": label,
            "count": int(count),
            "fraction": (
                float(count / total_rows)
                if total_rows
                else None
            ),
            "normalized_label": label.strip().lower(),
            "mapped_binary_guess": (
                "benign"
                if label.strip().lower() in BENIGN_TOKENS
                else "malicious"
            ),
        }
        for label, count in label_counter.most_common()
    ]

    profile_rows = []

    for name in sorted(global_columns):
        rows = column_rows[name]
        missing = column_missing[name]
        valid = column_numeric_valid[name]
        invalid = column_numeric_invalid[name]

        nonmissing = rows - missing

        numeric_fraction = (
            valid / nonmissing
            if nonmissing
            else 0.0
        )

        profile_rows.append(
            {
                "column": name,
                "rows_seen": rows,
                "missing": missing,
                "missing_fraction": (
                    missing / rows
                    if rows
                    else None
                ),
                "numeric_valid_nonmissing": valid,
                "numeric_invalid_nonmissing": invalid,
                "numeric_compatible_fraction": numeric_fraction,
                "nonfinite_numeric_values": column_nonfinite[name],
                "minimum_numeric": column_minimum.get(name),
                "maximum_numeric": column_maximum.get(name),
                "likely_numeric": int(
                    nonmissing > 0
                    and numeric_fraction >= args.numeric_threshold
                ),
                "is_detected_label_column": int(
                    name in label_columns_seen
                ),
            }
        )

    schema_rows = []

    for index, (schema, count) in enumerate(
        schema_counter.most_common(),
        start=1,
    ):
        schema_rows.append(
            {
                "schema_id": index,
                "files": count,
                "column_count": len(schema),
                "columns": "|".join(schema),
                "example_file": schema_examples[schema],
            }
        )

    write_csv(
        output / "file_inventory.csv",
        [
            "file",
            "bytes",
            "rows",
            "columns",
            "label_column",
            "distinct_labels_in_file",
            "sha256",
        ],
        inventory_rows,
    )

    write_csv(
        output / "label_counts.csv",
        [
            "label",
            "count",
            "fraction",
            "normalized_label",
            "mapped_binary_guess",
        ],
        label_rows,
    )

    write_csv(
        output / "column_profiles.csv",
        [
            "column",
            "rows_seen",
            "missing",
            "missing_fraction",
            "numeric_valid_nonmissing",
            "numeric_invalid_nonmissing",
            "numeric_compatible_fraction",
            "nonfinite_numeric_values",
            "minimum_numeric",
            "maximum_numeric",
            "likely_numeric",
            "is_detected_label_column",
        ],
        profile_rows,
    )

    write_csv(
        output / "schema_inventory.csv",
        [
            "schema_id",
            "files",
            "column_count",
            "columns",
            "example_file",
        ],
        schema_rows,
    )

    report = {
        "step": "08a_external_zero_day_dataset_audit",
        "script_version": VERSION,
        "dataset_name": "CICIoT2023",
        "input": str(input_path),
        "csv_files": len(files),
        "total_rows": int(total_rows),
        "distinct_schemas": len(schema_counter),
        "schemas_consistent": len(schema_counter) == 1,
        "detected_label_columns": dict(label_columns_seen),
        "dominant_label_column": dominant_label_column,
        "distinct_labels": len(label_counter),
        "label_counts": {
            key: int(value)
            for key, value in label_counter.most_common()
        },
        "provisional_binary_mapping": {
            "rule": (
                "labels normalized to one of "
                f"{sorted(BENIGN_TOKENS)} -> benign; every other non-benign label -> malicious"
            ),
            "benign_rows": int(benign_count),
            "malicious_rows": int(malicious_count),
            "mapping_is_final": False,
        },
        "likely_numeric_columns": [
            row["column"]
            for row in profile_rows
            if row["likely_numeric"]
            and not row["is_detected_label_column"]
        ],
        "full_file_sha256_computed": bool(full_hashes),
        "read_only": True,
        "model_loaded": False,
        "training_performed": False,
        "feature_selection_performed": False,
        "normalization_performed": False,
        "contours_generated": False,
        "zero_day_inference_performed": False,
        "elapsed_seconds": round(
            time.perf_counter() - started,
            2,
        ),
        "notes": [
            "This step audits the external dataset only.",
            "The provisional benign/malicious mapping must be reviewed before preprocessing.",
            "No CICIoT2023 label was used to modify the already-frozen IoT-23 neural-network checkpoint.",
            "The next step should define a documented external-dataset preprocessing and contour-generation protocol before inference.",
        ],
    }

    atomic_json(
        output / "step08a_summary.json",
        report,
    )

    print("\nSTEP 8A COMPLETE", flush=True)
    print(f"CSV files: {len(files):,}", flush=True)
    print(f"Rows: {total_rows:,}", flush=True)
    print(
        f"Distinct schemas: {len(schema_counter):,}; "
        f"consistent={len(schema_counter) == 1}",
        flush=True,
    )
    print(
        f"Detected label column(s): {dict(label_columns_seen)}",
        flush=True,
    )
    print(
        f"Distinct labels: {len(label_counter):,}",
        flush=True,
    )

    print("\nTop label counts:", flush=True)
    for label, count in label_counter.most_common(20):
        print(
            f"  {label!r}: {count:,}",
            flush=True,
        )

    print(
        "\nProvisional binary totals "
        "(review before Step 8B):",
        flush=True,
    )
    print(f"  benign:    {benign_count:,}", flush=True)
    print(f"  malicious: {malicious_count:,}", flush=True)

    print(
        f"\nSummary: {output / 'step08a_summary.json'}",
        flush=True,
    )
    print(
        "NO model inference was performed. "
        "Send the complete Step-8A output before continuing.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Step 8A: read-only audit of extracted CICIoT2023 CSV files."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="A CICIoT2023 CSV file or directory containing extracted CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "diagnostics/ciciot2023_step08a_audit"
        ),
    )

    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help=(
            "Optional exact label-column name. "
            "Normally omit and allow automatic detection."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--log-every-chunks",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--numeric-threshold",
        type=float,
        default=0.999,
        help=(
            "A non-label column is called likely-numeric when this fraction "
            "of its nonmissing values convert to numeric."
        ),
    )

    parser.add_argument(
        "--hash-files",
        action="store_true",
        help=(
            "Also SHA-256 every complete source CSV. "
            "This can add substantial disk I/O for large datasets."
        ),
    )

    args = parser.parse_args()

    if (
        args.chunk_size < 1
        or args.log_every_chunks < 1
        or not math.isfinite(args.numeric_threshold)
        or not 0 < args.numeric_threshold <= 1
    ):
        parser.error(
            "Use positive chunk/log sizes and numeric-threshold in (0,1]."
        )

    audit(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nSTEP 8A INTERRUPTED. No source data was modified.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as error:
        print(
            f"\nSTEP 8A FAILED: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
