"""Step 1: prepare one IoT-23 CSV for the CZ-ResViT implementation.

Run this from your iot_zero_day_research project folder in PowerShell:
    python step01_prepare_dataset7.py --input "D:\\iot_23_training_extracted\\dataset7.csv"

No additional packages are needed: this script uses Python's standard library.
It reads one record at a time, so the entire CSV never has to fit in memory.

Paper operation implemented here:
    Remove service, local_orig, local_resp and tunnel_parents (Section IV.A.1).

File-format handling needed for your exported CSVs:
    Separate a merged tunnel_parents / label / detailed-label field when present.

Task adaptation agreed for this experiment:
    Use existing binary labels: benign = 0 and malicious = 1.
    Retain detailed labels for reference; do not guess a five-class mapping.

Later steps will split the data, fit preprocessing, select features with RF-SHAP,
construct contours, and train ResNet-50 + Transformer. Missing feature markers
remain untouched here so imputation can be fitted on training data later.
"""

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path


# The 21 traffic fields in the IoT-23 Zeek connection export.
# These are potential inputs. Labels are deliberately absent from this list.
TRAFFIC_COLUMNS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes", "conn_state",
    "local_orig", "local_resp", "missed_bytes", "history", "orig_pkts",
    "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "tunnel_parents",
]

# The paper explicitly removes these four fields for IoT-23.
DROP_COLUMNS = ["service", "local_orig", "local_resp", "tunnel_parents"]
CANDIDATE_FEATURES = [c for c in TRAFFIC_COLUMNS if c not in DROP_COLUMNS]
TARGET_MAP = {"benign": 0, "malicious": 1}

# These columns help us check results and trace predictions back to source rows.
# They must never be included in the model's input features.
METADATA_COLUMNS = [
    "label", "detailed-label", "label_original", "target", "source_file", "source_row",
]
OUTPUT_COLUMNS = CANDIDATE_FEATURES + METADATA_COLUMNS

# Count stored missing markers, but do not replace them in Step 1.
# '(empty)' is a Zeek empty collection marker and is not treated as missing here.
MISSING_MARKERS = {"", "-", "nan", "null", "none"}


def parse_header(raw_header):
    """Describe how each original CSV cell maps to one or more field names."""
    if not raw_header:
        raise ValueError("The CSV is empty or has no header.")

    groups = []
    for original_name in raw_header:
        name = original_name.strip()
        words = name.split()
        if words in (
            ["tunnel_parents", "label", "detailed-label"],
            ["tunnel_parents", "label", "det_label"],
        ):
            groups.append(["tunnel_parents", "label", "detailed-label"])
        else:
            groups.append(["detailed-label" if name == "det_label" else name])

    columns = [name for group in groups for name in group]
    if len(set(columns)) != len(columns) or "" in columns:
        raise ValueError("The header contains duplicate or blank column names.")

    # A schema mismatch stops the run instead of silently losing a feature or label.
    expected = set(TRAFFIC_COLUMNS + ["label", "detailed-label"])
    missing = sorted(expected - set(columns))
    extra = sorted(set(columns) - expected)
    if missing or extra:
        raise ValueError(f"Unexpected CSV schema. Missing columns: {missing}; extra columns: {extra}")
    return groups


def parse_record(values, groups, source_row):
    """Parse a row, including the combined final field in your original CSVs."""
    if len(values) != len(groups):
        raise ValueError(
            f"Source row {source_row}: expected {len(groups)} CSV cells, got {len(values)}."
        )

    record = {}
    for names, value in zip(groups, values):
        # split() accepts repeated spaces or tabs in the combined field.
        # Ordinary traffic values are preserved exactly as stored in the CSV.
        parts = value.split() if len(names) > 1 else [value]
        if len(parts) != len(names):
            raise ValueError(f"Source row {source_row}: cannot separate the combined label field.")
        record.update(zip(names, parts))
    return record


def prepare(input_path, output_dir, progress_every=250_000):
    """Stream all records into a prepared CSV and write a small JSON summary."""
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "flows.csv"
    output_json = output_dir / "step01_summary.json"
    partial_csv = output_dir / "flows.csv.partial"
    partial_json = output_dir / "step01_summary.json.partial"

    # Existing results are protected. A new --output-dir allows a fresh run.
    for destination in (output_csv, output_json, partial_csv, partial_json):
        if destination.exists():
            raise FileExistsError(
                f"Output already exists: {destination}. Choose a new --output-dir for this run."
            )

    labels = Counter()
    detailed_labels = Counter()
    missing_markers = Counter()
    rows = blank_records = backward_steps = 0
    previous_ts = minimum_ts = maximum_ts = None
    started = time.perf_counter()
    original_stat = input_path.stat()

    print(f"Input: {input_path}", flush=True)
    print("Preparing all rows; progress is printed every 250,000 rows by default.", flush=True)

    # Track temporary files created by this run so an error never looks like success.
    owned_temporary_files = []
    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            raw_header = next(reader, None)
            groups = parse_header(raw_header)

            with partial_csv.open("x", encoding="utf-8", newline="") as destination:
                owned_temporary_files.append(partial_csv)
                writer = csv.DictWriter(destination, fieldnames=OUTPUT_COLUMNS)
                writer.writeheader()

                # source_row is the one-based CSV record position after the header.
                # File order is preserved. Chronological ordering is handled later.
                for source_row, values in enumerate(reader, start=1):
                    if not values:
                        blank_records += 1
                        continue
                    record = parse_record(values, groups, source_row)
                    original_label = record["label"]
                    label = original_label.strip().lower()
                    if label not in TARGET_MAP:
                        raise ValueError(
                            f"Source row {source_row}: expected benign/malicious, got {original_label!r}."
                        )

                    # Validate timestamp values without changing their stored precision.
                    try:
                        timestamp = float(record["ts"])
                        if not math.isfinite(timestamp):
                            raise ValueError("Non-finite timestamp")
                    except ValueError as error:
                        raise ValueError(f"Source row {source_row}: invalid timestamp.") from error

                    if previous_ts is not None and timestamp < previous_ts:
                        backward_steps += 1
                    minimum_ts = timestamp if minimum_ts is None else min(minimum_ts, timestamp)
                    maximum_ts = timestamp if maximum_ts is None else max(maximum_ts, timestamp)
                    previous_ts = timestamp

                    # Copy only the 17 permitted traffic fields, then append metadata.
                    prepared = {name: record[name] for name in CANDIDATE_FEATURES}
                    prepared.update({
                        "label": label,
                        "detailed-label": record["detailed-label"],
                        "label_original": original_label,
                        "target": TARGET_MAP[label],
                        "source_file": input_path.name,
                        "source_row": source_row,
                    })
                    writer.writerow(prepared)

                    rows += 1
                    labels[label] += 1
                    detailed_labels[record["detailed-label"]] += 1
                    for name in CANDIDATE_FEATURES:
                        if record[name].strip().lower() in MISSING_MARKERS:
                            missing_markers[name] += 1
                    if rows % progress_every == 0:
                        print(f"  Prepared {rows:,} rows...", flush=True)

        if rows == 0:
            raise ValueError("The file has a header but no data records.")

        final_stat = input_path.stat()
        if (original_stat.st_size, original_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
            raise ValueError("The input file changed while being read. Run again on a stable copy.")

        report = {
            "step": "01_prepare_single_iot23_csv",
            "input_file": str(input_path),
            "input_size_bytes": original_stat.st_size,
            "prepared_file": str(output_csv),
            "rows_read": rows,
            "rows_written": rows,
            "rows_dropped": 0,
            "blank_csv_records_skipped": blank_records,
            "original_csv_columns": len(raw_header),
            "parsed_columns": sum(len(group) for group in groups),
            "combined_label_field_separated": any(len(group) > 1 for group in groups),
            "removed_columns": DROP_COLUMNS,
            "candidate_feature_count": len(CANDIDATE_FEATURES),
            "candidate_features": CANDIDATE_FEATURES,
            "metadata_columns_not_for_model_input": METADATA_COLUMNS,
            "target_mapping": TARGET_MAP,
            "label_counts": dict(labels),
            "detailed_label_counts": dict(detailed_labels),
            "both_binary_classes_present": set(labels) == set(TARGET_MAP),
            "missing_marker_counts_in_retained_features": {
                name: missing_markers[name] for name in CANDIDATE_FEATURES
            },
            "timestamps": {
                "minimum": minimum_ts,
                "maximum": maximum_ts,
                "backward_steps_in_original_order": backward_steps,
                "original_order_preserved": True,
            },
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "python_version": sys.version.split()[0],
            "notes": [
                "Four-column removal follows the paper's IoT-23 preparation.",
                "Existing binary labels define this subset experiment; no five-class remapping is performed.",
                "Missing feature markers are retained for training-fitted preprocessing later.",
                "A dash in detailed-label is retained; it does not invalidate a benign binary label.",
                "No sampling, balancing, encoding of traffic features, imputation, scaling or training yet.",
                "Source filenames are provisional provenance, not verified session boundaries.",
            ],
        }

        with partial_json.open("x", encoding="utf-8") as destination:
            owned_temporary_files.append(partial_json)
            json.dump(report, destination, indent=2, ensure_ascii=False, allow_nan=False)
            destination.write("\n")
        # Publish the summary last, so it marks a completed preparation run.
        partial_csv.rename(output_csv)
        partial_json.rename(output_json)
    finally:
        for path in owned_temporary_files:
            if path.exists():
                path.unlink()

    print("\nSTEP 1 COMPLETE", flush=True)
    print(f"Rows read:    {rows:,}")
    print(f"Rows written: {rows:,}")
    print(f"Candidate traffic features: {len(CANDIDATE_FEATURES)}")
    print(f"Benign:    {labels['benign']:,}")
    print(f"Malicious: {labels['malicious']:,}")
    print(f"Backward timestamp steps: {backward_steps:,}")
    print(f"Prepared CSV: {output_csv}")
    print(f"Summary:      {output_json}")
    if not report["both_binary_classes_present"]:
        print("This input contains only one binary class; both are needed for the planned classifier.")
    return report


def main():
    parser = argparse.ArgumentParser(description="Step 1: prepare an IoT-23 CSV using the paper's column removal.")
    parser.add_argument("--input", type=Path, required=True, help="Original IoT-23 CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("prepared/dataset7_step01"))
    parser.add_argument("--progress-every", type=int, default=250_000)
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be a positive integer.")

    try:
        prepare(args.input, args.output_dir, args.progress_every)
    except (OSError, ValueError, csv.Error, UnicodeError) as error:
        print(f"\nSTEP 1 FAILED: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped before completion. No success summary was created.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
