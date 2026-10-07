"""Step 3: convert the sorted CSV into numerical storage for later preprocessing.

Install dependencies and run from your iot_zero_day_research folder:
    python -m pip install numpy pandas
    python step03_numeric_dataset7.py

Reads prepared/dataset7_step02/flows_sorted.csv and its Step 2 summary.
Writes prepared/dataset7_step03/ using two chunked passes over the CSV.

What is implemented:
  * The 11 numeric traffic fields become float64 values. Float64 preserves the
    timestamp precision that float32 would lose at Unix-epoch magnitudes.
  * Each of the 6 categorical fields gets its own reversible, lexically ordered
    integer dictionary. Missing categories are stored as -1, not as an ordinary
    category. '(empty)' remains an ordinary Zeek empty-collection marker.
  * Missing numeric fields become NaN. No artificial zero or global median is
    substituted. Labels and provenance go into separate arrays, never into X.

These are STAGING ARRAYS, not final model inputs. The full-file category
dictionaries are a reversible storage format, NOT training-fitted model
encoders. Later code must remap categories using training rows only and handle
unseen categories explicitly. Missing-value replacement, RF-SHAP and scaling
also need their own training scope after candidate image membership is fixed.
The paper's random 80:20 image-level split is not replaced by a temporal split.

Column typing and missing-marker handling are documented implementation choices
where the paper does not supply executable detail. No windows or models are
created in this step. All input rows and the existing binary targets are kept.

Future scripts can read X without loading the whole matrix into RAM:
    X = numpy.load("prepared/dataset7_step03/X_storage.npy", mmap_mode="r")
    y = numpy.load("prepared/dataset7_step03/y.npy", mmap_mode="r")
"""

import argparse
import csv
import gzip
import json
import sys
import tempfile
import time
from collections import Counter
from itertools import islice
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as error:
    raise SystemExit("Install dependencies first: python -m pip install numpy pandas") from error


CATEGORICAL_COLUMNS = ["uid", "id.orig_h", "id.resp_h", "proto", "conn_state", "history"]
NUMERIC_COLUMNS = [
    "ts", "id.orig_p", "id.resp_p", "duration", "orig_bytes", "resp_bytes",
    "missed_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
]
METADATA_COLUMNS = ["label", "detailed-label", "label_original", "target", "source_file", "source_row"]
MISSING_MARKERS = {"", "-", "nan", "null", "none"}
TARGET_MAP = {"benign": "0", "malicious": "1"}


def iter_chunks(path, columns, chunk_size):
    """Read bounded batches, failing on malformed records instead of skipping."""
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, strict=True)
        if next(reader, None) != columns:
            raise ValueError("CSV header does not match the Step 2 summary.")
        position = 0
        while True:
            records = list(islice(reader, chunk_size))
            if not records:
                break
            for record in records:
                position += 1
                if len(record) != len(columns):
                    raise ValueError(f"Prepared record {position}: incorrect number of CSV fields.")
            # Every cell is still a string at this point. Column names, rather
            # than numeric positions, determine which conversion gets applied.
            yield pd.DataFrame(records, columns=columns, dtype=object)


def strings_and_missing(series):
    """Trim surrounding whitespace and identify explicit stored missing tokens."""
    values = series.str.strip()
    missing = values.str.lower().isin(MISSING_MARKERS)
    return values, missing


def to_numeric_column(series, name):
    """Convert a numeric field, preserving unknowns as NaN and rejecting junk."""
    values, missing = strings_and_missing(series)
    try:
        numbers = pd.to_numeric(values.mask(missing), errors="raise").to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Field {name!r} contains unexpected nonnumeric text: {error}") from error
    if np.isinf(numbers).any():
        raise ValueError(f"Field {name!r} contains an infinite value.")
    if name == "ts" and not np.isfinite(numbers).all():
        raise ValueError("Timestamps must be finite; Step 2 should already have checked this.")
    return numbers


def file_signature(path):
    """Detect ordinary edits to the source between the two passes."""
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def close_arrays(arrays):
    """Flush and release memory-mapped files before renaming them on Windows."""
    for array in arrays:
        array.flush()
        array._mmap.close()


def prepare_numeric(input_path, output_dir, chunk_size=50_000):
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    summary_path = input_path.parent / "step02_summary.json"
    with summary_path.open("r", encoding="utf-8") as source:
        previous = json.load(source)
    if previous.get("step") != "02_sort_single_iot23_csv":
        raise ValueError("Use the sorted CSV and summary produced by Step 2.")
    features = previous["candidate_features"]
    if (len(features) != 17 or len(set(features)) != 17
            or set(features) != set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)):
        raise ValueError("Expected the 17 traffic features retained in Step 1.")
    if previous["metadata_columns_not_for_model_input"] != METADATA_COLUMNS:
        raise ValueError("Unexpected metadata schema in the Step 2 summary.")
    columns = features + METADATA_COLUMNS
    total = int(previous["rows_written"])
    if total < 1:
        raise ValueError("No records to process.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty. Choose a new --output-dir to rerun.")
    signature = file_signature(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"Input: {input_path}", flush=True)
    print("1/3 Collecting categorical values and verifying targets...", flush=True)

    # Only categorical vocabularies are accumulated. Numeric traffic records
    # remain in chunks. Even the high-cardinality UID field is handled separately
    # from the large numerical matrix, which is backed by files on disk.
    unique_values = {name: set() for name in CATEGORICAL_COLUMNS}
    label_counts = Counter()
    missing_counts = Counter()
    count = 0
    for frame in iter_chunks(input_path, columns, chunk_size):
        labels = frame["label"]
        if not labels.isin(TARGET_MAP).all() or not frame["target"].eq(labels.map(TARGET_MAP)).all():
            raise ValueError("A binary label or its target code disagrees with Step 1.")
        if not frame["source_file"].eq(previous["source_file"]).all():
            raise ValueError("Source-file provenance differs from Step 2.")
        label_counts.update(labels.value_counts().to_dict())
        for name in CATEGORICAL_COLUMNS:
            values, missing = strings_and_missing(frame[name])
            unique_values[name].update(values[~missing].unique())
            missing_counts[name] += int(missing.sum())
        count += len(frame)
        if count % 250_000 == 0:
            print(f"  Scanned {count:,} rows...", flush=True)
    if count != total or dict(label_counts) != previous["label_counts"]:
        raise ValueError("Row or label counts differ from Step 2.")
    if file_signature(input_path) != signature:
        raise ValueError("The input CSV changed during the first pass.")

    # Build dictionaries once. Codes are column-specific and deterministic.
    # A code is only an identifier for a text value; its magnitude is not a
    # measurement or a learned relationship between categories.
    vocabularies = {}
    for name in CATEGORICAL_COLUMNS:
        vocabularies[name] = np.asarray(sorted(unique_values.pop(name)), dtype=str)
        print(f"  {name}: {len(vocabularies[name]):,} nonmissing categories", flush=True)

    print("2/3 Writing the numerical arrays...", flush=True)
    numeric_stats = {name: {"valid": 0, "minimum": None, "maximum": None} for name in NUMERIC_COLUMNS}
    with tempfile.TemporaryDirectory(prefix="numeric_work_", dir=output_dir) as temporary:
        stage = Path(temporary)
        category_files = {}
        for name, vocabulary in vocabularies.items():
            print(f"  Saving dictionary for {name}...", flush=True)
            filename = "categories_" + name.replace(".", "_") + ".json.gz"
            category_files[name] = filename
            # categories[code] recovers the original nonmissing text value.
            with gzip.open(stage / filename, "wt", encoding="utf-8") as destination:
                json.dump({"feature": name, "missing_code": -1, "scope": "full-file storage dictionary",
                           "categories": vocabulary.tolist()}, destination, ensure_ascii=False)

        arrays = []
        try:
            X = np.lib.format.open_memmap(stage / "X_storage.npy", mode="w+", dtype=np.float64, shape=(total, 17))
            arrays.append(X)
            y = np.lib.format.open_memmap(stage / "y.npy", mode="w+", dtype=np.uint8, shape=(total,))
            arrays.append(y)
            source_rows = np.lib.format.open_memmap(stage / "source_rows.npy", mode="w+", dtype=np.int64, shape=(total,))
            arrays.append(source_rows)

            offset = 0
            previous_key = None
            export_labels = Counter()
            for frame in iter_chunks(input_path, columns, chunk_size):
                end = offset + len(frame)
                if end > total:
                    raise ValueError("Input has more rows than Step 2 reported.")
                for index, name in enumerate(features):
                    if name in CATEGORICAL_COLUMNS:
                        values, missing = strings_and_missing(frame[name])
                        valid = ~missing.to_numpy()
                        codes = np.full(len(frame), -1, dtype=np.int64)
                        nonmissing = values[~missing].to_numpy(dtype=str)
                        vocabulary = vocabularies[name]
                        positions = np.searchsorted(vocabulary, nonmissing)
                        if len(positions):
                            if np.any(positions >= len(vocabulary)) or not np.array_equal(vocabulary[positions], nonmissing):
                                raise ValueError(f"Field {name}: new values appeared between the two passes.")
                            codes[valid] = positions
                        X[offset:end, index] = codes
                    else:
                        numbers = to_numeric_column(frame[name], name)
                        X[offset:end, index] = numbers
                        finite_values = numbers[np.isfinite(numbers)]
                        missing_counts[name] += int(np.isnan(numbers).sum())
                        stats = numeric_stats[name]
                        stats["valid"] += len(finite_values)
                        if len(finite_values):
                            low, high = float(finite_values.min()), float(finite_values.max())
                            stats["minimum"] = low if stats["minimum"] is None else min(stats["minimum"], low)
                            stats["maximum"] = high if stats["maximum"] is None else max(stats["maximum"], high)

                # Never mix labels or provenance into the 17-column X matrix.
                labels = frame["label"]
                if not labels.isin(TARGET_MAP).all() or not frame["target"].eq(labels.map(TARGET_MAP)).all():
                    raise ValueError("Inconsistent labels or targets in the second pass.")
                if not frame["source_file"].eq(previous["source_file"]).all():
                    raise ValueError("Source-file provenance changed in the second pass.")
                ids = np.asarray([int(value) for value in frame["source_row"]], dtype=np.int64)
                if (ids < 1).any():
                    raise ValueError("Invalid source-row identifier.")
                times = np.asarray(X[offset:end, features.index("ts")])
                for timestamp, source_id in zip(times, ids):
                    key = (float(timestamp), int(source_id))
                    if previous_key is not None and key <= previous_key:
                        raise ValueError("Input does not preserve Step 2's chronological order.")
                    previous_key = key
                y[offset:end] = frame["target"].to_numpy(dtype=np.uint8)
                source_rows[offset:end] = ids
                export_labels.update(labels.value_counts().to_dict())
                offset = end
                if offset % 250_000 == 0:
                    print(f"  Converted {offset:,} rows...", flush=True)
            if offset != total or export_labels != label_counts:
                raise ValueError("Row or label counts changed during conversion.")
            if file_signature(input_path) != signature:
                raise ValueError("The input CSV changed during conversion.")
        finally:
            close_arrays(arrays)

        print("3/3 Saving the feature definitions and summary...", flush=True)
        report = {
            "step": "03_numeric_storage_single_iot23_csv",
            "status": "numeric_storage_complete_not_model_ready",
            "input_file": str(input_path),
            "source_file": previous["source_file"],
            "rows_read": total, "rows_written": total, "rows_dropped": 0,
            "X_shape": [total, 17], "X_dtype": "float64", "y_dtype": "uint8",
            "label_counts": dict(label_counts), "target_mapping": {"benign": 0, "malicious": 1},
            "feature_names": features,
            "categorical_features": CATEGORICAL_COLUMNS, "numeric_features": NUMERIC_COLUMNS,
            "category_counts": {name: len(values) for name, values in vocabularies.items()},
            "category_dictionary_files": category_files,
            "categorical_missing_code": -1,
            "numeric_missing_representation": "NaN",
            "missing_counts": {name: missing_counts[name] for name in features},
            "numeric_statistics": numeric_stats,
            "storage_files": {"X": "X_storage.npy", "y": "y.npy", "source_rows": "source_rows.npy"},
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
            "notes": [
                "Row position corresponds to the same row in the sorted CSV; source_rows preserves original provenance.",
                "Category dictionaries cover the full file solely for reversible storage, not for fitted model preprocessing.",
                "Fit final category remapping on training rows, with an explicit code for unseen categories.",
                "Numeric missing values remain NaN; fit imputation, RF-SHAP and scaling on the appropriate training rows later.",
                "Labels, detailed labels and provenance are excluded from X. The paper's traffic identifiers remain candidates.",
                "No min-max normalization, feature selection, window construction, balancing or model training yet.",
            ],
        }
        (stage / "step03_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # The summary is published last and marks successful completion.
        names = list(category_files.values()) + ["X_storage.npy", "y.npy", "source_rows.npy", "step03_summary.json"]
        for name in names:
            (stage / name).rename(output_dir / name)

    print("\nSTEP 3 COMPLETE", flush=True)
    print(f"Rows read:    {total:,}")
    print(f"Rows written: {total:,}")
    print(f"X shape: ({total:,}, 17)")
    print("Numeric columns: 11; categorical columns: 6")
    print(f"Benign:    {label_counts['benign']:,}")
    print(f"Malicious: {label_counts['malicious']:,}")
    print("Missing numeric values remain NaN for later training-fitted imputation.")
    print(f"Summary: {output_dir / 'step03_summary.json'}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Step 3: create numerical storage from the sorted IoT-23 CSV.")
    parser.add_argument("--input", type=Path, default=Path("prepared/dataset7_step02/flows_sorted.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("prepared/dataset7_step03"))
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive.")
    try:
        prepare_numeric(args.input, args.output_dir, args.chunk_size)
    except (OSError, ValueError, KeyError, TypeError, OverflowError, csv.Error) as error:
        print(f"\nSTEP 3 FAILED: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped before completion. No success summary was created.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
