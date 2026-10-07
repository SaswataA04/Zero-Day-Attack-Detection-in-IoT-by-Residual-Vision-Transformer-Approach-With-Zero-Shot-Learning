#!/usr/bin/env python3
"""Step 6c: audit the saved train/validation representations on the CPU.

Run from the project folder:
    python step06c_audit_representation_dataset7.py

Dependency: numpy (already installed for Steps 3-6).

Purpose
-------
Step 3 found a different UID for every connection. Step 4 fitted categorical
vocabularies on training rows and retained UID among the 15 selected features.
Consequently, validation UIDs can all become the same unknown-category code.
This script measures that difference in the existing outputs. It also checks
the other selected features, rather than assuming UID explains every error.

The script reads training/validation numerical rows and correlation matrices
in small batches. Test membership metadata is used only to exclude test rows;
test feature values, matrices, targets, images and predictions are not used.
No ZIP is opened, no model is loaded, and no training is performed.

For a diagnostic comparison ONLY, an in-memory COPY of each filtered matrix
has its UID row/column set to zero. Matrix hashes before/after that operation
measure exact numerical repetition. They do NOT measure rendered PNG counts,
model accuracy, or the result of refitting feature selection without UID.
No replacement matrices, images, datasets or checkpoints are saved.

Outputs, in a new diagnostics/dataset7_step06c folder:
    representation_summary.json
    feature_audit.csv
An existing output folder is preserved; a numbered sibling is used instead.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("Missing numpy. Run: python -m pip install numpy") from exc


PARTITIONS = ((0, "train"), (1, "validation"))
WINDOW_ROWS = 15


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as source:
        return json.load(source)


def load_array(path):
    """Read-only memory maps avoid loading the full dataset into RAM."""
    return np.load(path, mmap_mode="r", allow_pickle=False)


def matrix_counts(matrices, counts):
    """Hash float32 values exactly, treating positive/negative zero equally."""
    values = np.array(matrices, dtype="<f4", order="C", copy=True)
    values[values == 0] = 0.0
    counts.update(hashlib.sha256(item.tobytes()).hexdigest() for item in values)


def describe_counts(counts):
    total = sum(counts.values())
    return {
        "windows": total,
        "distinct_hashes": len(counts),
        "distinct_hashes_divided_by_windows": len(counts) / total,
        "largest_group_windows": max(counts.values()),
    }


def compare_partitions(counts):
    training, validation = counts["train"], counts["validation"]
    return {
        name: describe_counts(counts[name]) for _, name in PARTITIONS
    } | {
        "validation_windows_matching_a_training_hash": sum(
            n for digest, n in validation.items() if digest in training
        )
    }


def audit(step04, step05, batch_windows=2048):
    """Inspect existing representations; return reports without modifying inputs."""
    if batch_windows < 1:
        raise ValueError("--batch-windows must be positive.")
    selected = read_json(step04 / "selected_features.json")
    preprocessing = read_json(step04 / "preprocessing.json")
    normalization = read_json(step05 / "normalization.json")
    old_summary = read_json(step04 / "step04_summary.json")
    features = selected["feature_names"]
    if len(features) != 15 or len(set(features)) != 15:
        raise ValueError("Expected the original 15 distinct selected features.")
    if features != normalization["features"]:
        raise ValueError("Step 4 and Step 5 feature orders differ.")
    if old_summary["window_rows"] != WINDOW_ROWS:
        raise ValueError("This audit expects the saved 15-row windows.")
    if old_summary["split_codes"]["train"] != 0 or old_summary["split_codes"]["validation"] != 1:
        raise ValueError("Unexpected split codes.")

    X = load_array(step04 / "X_selected.npy")
    row_split = load_array(step04 / "row_split.npy")
    window_split = load_array(step04 / "window_split.npy")
    splits = load_array(step05 / "splits.npy")
    activity = load_array(step05 / "feature_activity.npy")
    filtered = load_array(step05 / "correlations_filtered.npy")
    mask = load_array(step05 / "correlation_mask.npy")
    n_windows, n_features = len(splits), len(features)
    if X.ndim != 2 or X.shape[1] != n_features or len(X) // WINDOW_ROWS != n_windows:
        raise ValueError("Selected feature shape does not match Step 5 windows.")
    if row_split.shape != (len(X),) or window_split.shape != splits.shape:
        raise ValueError("Split array shapes do not match the saved data.")
    if activity.shape != (n_windows, n_features) or activity.dtype != np.dtype(bool):
        raise ValueError("Invalid saved feature-activity array.")
    if filtered.shape != (n_windows, n_features, n_features):
        raise ValueError("Invalid saved filtered-correlation shape.")
    if mask.shape != (n_features, n_features) or mask.dtype != np.dtype(bool):
        raise ValueError("Invalid saved correlation mask.")
    if not np.all(np.diag(mask)):
        raise ValueError("Expected Step 5 to retain all correlation diagonals.")
    if not np.array_equal(splits, window_split):
        raise ValueError("Step 4 and Step 5 split membership differs.")

    minima = np.asarray(normalization["training_min"], dtype=np.float64)
    maxima = np.asarray(normalization["training_max"], dtype=np.float64)
    if minima.shape != (n_features,) or maxima.shape != (n_features,):
        raise ValueError("Invalid saved normalization parameters.")
    if not np.isfinite(minima).all() or not np.isfinite(maxima).all() or np.any(maxima < minima):
        raise ValueError("Nonfinite or reversed normalization extrema.")
    spans = maxima - minima
    scalable = spans > 0
    tolerance = float(normalization["within_window_std_tolerance"])
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid within-window tolerance.")
    uid_index = features.index("uid") if "uid" in features else None
    full_counts = {name: Counter() for _, name in PARTITIONS}
    neutral_counts = {name: Counter() for _, name in PARTITIONS}
    png_counts = {name: Counter() for _, name in PARTITIONS}
    feature_rows = []
    uid_details = {}

    print("1/2 Auditing saved train/validation values and correlation matrices...", flush=True)
    for code, name in PARTITIONS:
        ids = np.flatnonzero(splits == code)
        if not len(ids):
            raise ValueError(f"No {name} windows found.")
        varying = np.zeros(n_features, dtype=np.int64)
        raw_varying = np.zeros(n_features, dtype=np.int64)
        unknown = np.zeros(n_features, dtype=np.int64)
        missing = np.zeros(n_features, dtype=np.int64)
        value_min = np.full(n_features, np.inf)
        value_max = np.full(n_features, -np.inf)
        uid_diagonal_one = uid_diagonal_zero = 0
        for start in range(0, len(ids), batch_windows):
            batch_ids = ids[start:start + batch_windows]
            rows = batch_ids[:, None] * WINDOW_ROWS + np.arange(WINDOW_ROWS)
            if not np.all(row_split[rows] == code):
                raise ValueError("A window contains rows with a different partition.")
            values = np.asarray(X[rows], dtype=np.float64)
            matrices = np.array(filtered[batch_ids], dtype=np.float32, copy=True)
            if not np.isfinite(values).all() or not np.isfinite(matrices).all():
                raise ValueError(f"Nonfinite values in the {name} representation.")
            saved_activity = np.asarray(activity[batch_ids])
            # Reproduce Step 5's activity test using the saved training extrema.
            normalized = np.zeros_like(values)
            normalized[:, :, scalable] = (
                2 * (values[:, :, scalable] - minima[scalable]) / spans[scalable] - 1
            )
            centered = normalized - normalized.mean(axis=1, keepdims=True)
            expected_activity = np.linalg.norm(centered, axis=1) / np.sqrt(WINDOW_ROWS) > tolerance
            if not np.array_equal(expected_activity, saved_activity):
                raise ValueError("Step 5 activity does not match the supplied Step 4 values.")
            if not np.array_equal(np.diagonal(matrices, axis1=1, axis2=2), saved_activity):
                raise ValueError("Filtered diagonals disagree with feature activity.")
            varying += saved_activity.sum(axis=0)
            raw_varying += np.any(values != values[:, :1, :], axis=1).sum(axis=0)
            value_min = np.minimum(value_min, values.min(axis=(0, 1)))
            value_max = np.maximum(value_max, values.max(axis=(0, 1)))
            for j, feature in enumerate(features):
                settings = preprocessing["parameters"][feature]
                if settings["kind"] == "categorical":
                    unknown[j] += np.count_nonzero(values[:, :, j] == settings["unknown_code"])
                    missing[j] += np.count_nonzero(values[:, :, j] == settings["missing_code"])
            matrix_counts(matrices, full_counts[name])
            if uid_index is not None:
                diagonal = matrices[:, uid_index, uid_index]
                uid_diagonal_one += int(np.count_nonzero(diagonal == 1))
                uid_diagonal_zero += int(np.count_nonzero(diagonal == 0))
                # This is an in-memory diagnostic, not an exported correction.
                matrices[:, uid_index, :] = 0
                matrices[:, :, uid_index] = 0
                matrix_counts(matrices, neutral_counts[name])
            processed = min(start + batch_windows, len(ids))
            if processed == len(ids) or processed // 20000 != start // 20000:
                print(f"  {name}: {processed:,}/{len(ids):,} windows", flush=True)

        for j, feature in enumerate(features):
            categorical = preprocessing["parameters"][feature]["kind"] == "categorical"
            if categorical and int(unknown[j]) != old_summary["unknown_category_counts"][name][feature]:
                raise ValueError(f"Unknown-category counts differ from Step 4: {name}/{feature}.")
            feature_rows.append({
                "split": name, "feature": feature,
                "kind": preprocessing["parameters"][feature]["kind"],
                "windows": len(ids), "rows": len(ids) * WINDOW_ROWS,
                "windows_with_different_stored_values": int(raw_varying[j]),
                "varying_windows_in_correlations": int(varying[j]),
                "varying_window_fraction": float(varying[j] / len(ids)),
                "minimum_stored_value": float(value_min[j]),
                "maximum_stored_value": float(value_max[j]),
                "unknown_rows": int(unknown[j]) if categorical else None,
                "missing_category_rows": int(missing[j]) if categorical else None,
            })
        if uid_index is not None:
            uid_details[name] = next(row.copy() for row in feature_rows if row["split"] == name and row["feature"] == "uid")
            uid_details[name].update({"uid_diagonal_one_windows": uid_diagonal_one,
                                      "uid_diagonal_zero_windows": uid_diagonal_zero})

    print("2/2 Checking existing train/validation PNG hash metadata...", flush=True)
    seen = set()
    with (step05 / "image_index.csv").open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"window_id", "split", "png_sha256"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Missing required image-index columns.")
        for row in reader:
            # Skip test metadata before accessing its ID, target or PNG hash.
            if row["split"] == "test":
                continue
            name = row["split"]
            if name not in png_counts:
                raise ValueError("Unexpected partition in the image index.")
            window_id = int(row["window_id"])
            if not 0 <= window_id < n_windows or int(splits[window_id]) != dict((n, c) for c, n in PARTITIONS)[name]:
                raise ValueError("Image index and split array disagree.")
            if window_id in seen:
                raise ValueError("Repeated window ID in the image index.")
            digest = row["png_sha256"].strip().lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Invalid saved PNG SHA-256.")
            seen.add(window_id)
            png_counts[name][digest] += 1
    expected_windows = sum(sum(counts.values()) for counts in full_counts.values())
    if len(seen) != expected_windows:
        raise ValueError("The image index does not contain every train/validation window.")

    report = {
        "step": "06c_train_validation_representation_audit",
        "step04_input": str(step04), "step05_input": str(step05),
        "feature_names": features, "uid_selected": uid_index is not None,
        "uid_by_partition": uid_details,
        "uid_retained_off_diagonal_correlations": (
            [feature for j, feature in enumerate(features) if j != uid_index and mask[uid_index, j]]
            if uid_index is not None else []
        ),
        "png_hash_metadata": compare_partitions(png_counts),
        "saved_filtered_matrix_hashes": compare_partitions(full_counts),
        "diagnostic_matrix_hashes_with_uid_row_and_column_zeroed": (
            compare_partitions(neutral_counts) if uid_index is not None else None
        ),
        "feature_audit": feature_rows,
        "test_feature_rows_read": 0, "test_correlation_matrices_read": 0,
        "test_targets_used": 0, "test_images_read": 0, "test_predictions_used": 0,
        "datasets_or_checkpoints_changed": False,
        "notes": [
            "PNG hashes are saved metadata; encoded image bytes were not re-read.",
            "Distinct encoded hashes need not imply different decoded pixels.",
            "Matrix hashes compare exact float32 entries after canonicalizing signed zero.",
            "Small numerical matrix differences can render to the same PNG.",
            "Zeroing UID here keeps the previous training-fitted mask and all other entries fixed.",
            "This comparison does not estimate accuracy after refitting without UID.",
            "Feature counts are window-based; categorical unknown counts are connection-row-based.",
            "A representation mismatch does not by itself identify the cause of all prediction errors.",
        ],
    }
    return report, feature_rows


def save_reports(output, report, feature_rows):
    """Save reports to a new folder, preserving previous reports and inputs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    requested = output
    counter = 1
    while output.exists():
        output = requested.with_name(f"{requested.name}_{counter:02d}")
        counter += 1
    stage = Path(tempfile.mkdtemp(prefix="representation_audit_", dir=output.parent))
    try:
        with (stage / "representation_summary.json").open("w", encoding="utf-8") as destination:
            json.dump(report, destination, indent=2, allow_nan=False)
            destination.write("\n")
        with (stage / "feature_audit.csv").open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(feature_rows[0]))
            writer.writeheader()
            writer.writerows(feature_rows)
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step04-dir", type=Path, default=Path("prepared/dataset7_step04"))
    parser.add_argument("--step05-dir", type=Path, default=Path("prepared/dataset7_step05"))
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/dataset7_step06c"))
    parser.add_argument("--batch-windows", type=int, default=2048)
    args = parser.parse_args()
    try:
        report, feature_rows = audit(args.step04_dir.resolve(), args.step05_dir.resolve(), args.batch_windows)
        output = save_reports(args.output_dir.resolve(), report, feature_rows)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"Audit stopped: {exc}") from exc
    print("\nSTEP 6C AUDIT COMPLETE")
    if report["uid_selected"]:
        for _, name in PARTITIONS:
            info = report["uid_by_partition"][name]
            print(f"{name} UID: unknown rows {info['unknown_rows']:,}/{info['rows']:,}; "
                  f"varying in {info['varying_windows_in_correlations']:,}/{info['windows']:,} windows")
    for _, name in PARTITIONS:
        png = report["png_hash_metadata"][name]
        before = report["saved_filtered_matrix_hashes"][name]["distinct_hashes"]
        print(f"{name}: {png['distinct_hashes']:,} distinct saved PNG hashes; {before:,} distinct filtered matrices")
        if report["uid_selected"]:
            after = report["diagnostic_matrix_hashes_with_uid_row_and_column_zeroed"][name]["distinct_hashes"]
            print(f"  Diagnostic matrix count with UID row/column zeroed: {after:,} (not a PNG count)")
    print(f"Summary: {output / 'representation_summary.json'}")
    print("No input files changed; no GPU, model execution or test-data evaluation used.")


if __name__ == "__main__":
    main()
