#!/usr/bin/env python
"""
step05c_hash_group_split.py

Create a leakage-clean TRAIN/VALIDATION view of an existing Step-5 Dataset 7 run.

IMPORTANT METHODOLOGY
---------------------
Step 5 fitted normalization/correlation filtering using the ORIGINAL training
partition. Therefore this script does NOT create a fresh 80/20 resplit and does
NOT move any original-training window into validation.

Instead:
  * every original TRAIN window stays TRAIN;
  * an original VALIDATION window stays VALIDATION only if its saved PNG SHA-256
    was never present in original TRAIN;
  * original VALIDATION windows whose PNG hash was already in original TRAIN are
    reassigned to TRAIN;
  * TEST assignments are left unchanged;
  * TEST image bytes are never opened, hashed, or decoded.

This gives an exact-PNG-novel validation subset while avoiding the more serious
mistake of validating on windows that participated in Step-5's training-fitted
preprocessing.

The script uses the saved png_sha256 metadata in image_index.csv. It does not
rehash PNG bytes.

Output is a lightweight Step-5-compatible folder:
  step05_summary.json       updated split counts + audit metadata
  labels.npy                hardlink/copy of original labels
  splits.npy                updated train/validation assignments
  image_index.csv           updated train/validation split column
  images/                   junction/symlink to original immutable ZIP shards

Compatible with step06_train_dataset7.py.

The --validation-fraction option is accepted only for backward compatibility
with the earlier command. It is intentionally NOT used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("NumPy is required: python -m pip install numpy") from exc


CLASS_NAMES = ("benign", "malicious")
SPLIT_NAMES = ("train", "validation", "test")
MODIFIED_FILES = {"step05_summary.json", "splits.npy", "image_index.csv"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, value):
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_npy(path: Path, value):
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as f:
        np.save(f, value, allow_pickle=False)
    tmp.replace(path)


def file_sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def class_counts(labels, splits):
    return {
        name: {
            CLASS_NAMES[c]: int(np.count_nonzero((splits == s) & (labels == c)))
            for c in (0, 1)
        }
        for s, name in enumerate(SPLIT_NAMES)
    }


def validate_input(root: Path):
    required = [
        root / "step05_summary.json",
        root / "labels.npy",
        root / "splits.npy",
        root / "image_index.csv",
        root / "images",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Step-5 input(s):\n  " + "\n  ".join(missing))

    summary = read_json(root / "step05_summary.json")
    if summary.get("step") != "05_contours_single_iot23_csv":
        raise ValueError("Expected a completed Step-5 contour dataset.")

    labels = np.load(root / "labels.npy", allow_pickle=False)
    splits = np.load(root / "splits.npy", allow_pickle=False)

    n = int(summary["images"])
    if labels.shape != (n,) or splits.shape != (n,):
        raise ValueError("labels.npy/splits.npy shape disagrees with Step-5 summary.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels.npy must contain only 0/1.")
    if not np.isin(splits, [0, 1, 2]).all():
        raise ValueError("splits.npy must contain only train/validation/test codes 0/1/2.")

    observed = class_counts(labels, splits)
    if observed != summary["window_class_counts"]:
        raise ValueError(
            "Saved labels/splits disagree with step05_summary.json.\n"
            f"Observed: {observed}\n"
            f"Summary : {summary['window_class_counts']}"
        )
    return summary, labels, splits


def read_index_and_plan(root: Path, labels, splits):
    """
    Read TRAIN/VALIDATION hash metadata.
    TEST rows are retained for later rewriting but target/hash/id are not
    interpreted when planning the clean validation subset.
    """
    index_path = root / "image_index.csv"
    rows = []

    train_hashes = set()
    train_hash_labels = defaultdict(set)
    validation_entries = []

    with index_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        required = {"window_id", "split", "target", "png_sha256"}
        if not required <= set(fieldnames):
            raise ValueError(
                f"image_index.csv lacks required columns: {sorted(required - set(fieldnames))}"
            )

        for row in reader:
            split_name = row["split"]
            rows.append(row)

            if split_name == "test":
                # Do not interpret held-out window id, target or PNG hash here.
                continue

            if split_name not in ("train", "validation"):
                raise ValueError(f"Unexpected split in image_index.csv: {split_name!r}")

            window_id = int(row["window_id"])
            target = int(row["target"])
            digest = row["png_sha256"].strip().lower()

            if not (0 <= window_id < len(labels)):
                raise ValueError(f"Invalid window_id {window_id}")
            if target not in (0, 1):
                raise ValueError(f"Invalid target for window {window_id}")
            expected_split = 0 if split_name == "train" else 1
            if int(splits[window_id]) != expected_split:
                raise ValueError(
                    f"image_index.csv and splits.npy disagree for window {window_id}"
                )
            if int(labels[window_id]) != target:
                raise ValueError(
                    f"image_index.csv and labels.npy disagree for window {window_id}"
                )
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Invalid saved PNG SHA-256 for window {window_id}")

            if split_name == "train":
                train_hashes.add(digest)
                train_hash_labels[digest].add(target)
            else:
                validation_entries.append((window_id, target, digest))

    conflicting_train = {
        digest: sorted(values)
        for digest, values in train_hash_labels.items()
        if len(values) > 1
    }
    if conflicting_train:
        sample = next(iter(conflicting_train.items()))
        raise ValueError(
            "Identical training PNG hashes carry conflicting labels. "
            "This is a label-integrity problem that should be fixed before training. "
            f"Example: {sample[0]} -> labels {sample[1]}"
        )

    keep_validation = set()
    move_to_train = set()

    seen_counts = [0, 0]
    unseen_counts = [0, 0]

    for window_id, target, digest in validation_entries:
        if digest in train_hashes:
            move_to_train.add(window_id)
            seen_counts[target] += 1
        else:
            keep_validation.add(window_id)
            unseen_counts[target] += 1

    return {
        "rows": rows,
        "fieldnames": fieldnames,
        "train_hashes": train_hashes,
        "keep_validation": keep_validation,
        "move_to_train": move_to_train,
        "seen_counts": seen_counts,
        "unseen_counts": unseen_counts,
        "original_validation_count": len(validation_entries),
    }


def create_images_reference(source: Path, destination: Path):
    """
    Make output/images reference the immutable original Step-5 ZIP directory.
    Prefer a Windows directory junction. On non-Windows use a symlink.
    """
    source = source.resolve()

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Destination already exists: {destination}")

    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(destination), str(source)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return {"type": "windows_junction", "source": str(source)}

        # Fallback: create real folder and hardlink/copy each immutable shard.
        destination.mkdir(parents=True)
        linked = copied = 0
        for src in source.iterdir():
            if not src.is_file():
                continue
            dst = destination / src.name
            try:
                os.link(src, dst)
                linked += 1
            except OSError:
                shutil.copy2(src, dst)
                copied += 1
        return {
            "type": "hardlink_copy_fallback",
            "source": str(source),
            "hardlinked_files": linked,
            "copied_files": copied,
            "junction_error": (result.stderr or result.stdout).strip(),
        }

    os.symlink(source, destination, target_is_directory=True)
    return {"type": "directory_symlink", "source": str(source)}


def link_or_copy_file(source: Path, destination: Path):
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def write_index(output_path: Path, fieldnames, rows, new_splits):
    tmp = output_path.with_name(output_path.name + ".partial")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            out = dict(row)

            if row["split"] == "test":
                # Preserve held-out row metadata unchanged.
                writer.writerow(out)
                continue

            window_id = int(row["window_id"])
            new_code = int(new_splits[window_id])
            if new_code not in (0, 1):
                raise ValueError(
                    f"Non-test row {window_id} unexpectedly has split code {new_code}"
                )
            out["split"] = SPLIT_NAMES[new_code]
            writer.writerow(out)

    tmp.replace(output_path)


def verify_hash_separation(output_index: Path):
    train_hashes = set()
    validation_hashes = set()
    counts = {
        "train": {"benign": 0, "malicious": 0},
        "validation": {"benign": 0, "malicious": 0},
    }

    with output_index.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_name = row["split"]
            if split_name == "test":
                continue

            digest = row["png_sha256"].strip().lower()
            target = int(row["target"])
            class_name = CLASS_NAMES[target]
            counts[split_name][class_name] += 1

            if split_name == "train":
                train_hashes.add(digest)
            elif split_name == "validation":
                validation_hashes.add(digest)
            else:
                raise ValueError(f"Unexpected non-test split: {split_name}")

    overlap = train_hashes & validation_hashes
    return {
        "train_unique_hashes": len(train_hashes),
        "validation_unique_hashes": len(validation_hashes),
        "train_validation_hash_overlap": len(overlap),
        "counts_from_index": counts,
        "sample_overlapping_hashes": sorted(overlap)[:10],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create an exact-PNG-novel validation subset for Dataset 7."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=None,
        help=(
            "Accepted only for compatibility with the earlier command. "
            "It is intentionally ignored; original training never moves to validation."
        ),
    )
    args = parser.parse_args()

    root = args.input_dir.resolve()
    out = args.output_dir.resolve()

    if out == root or root in out.parents or out in root.parents:
        raise ValueError("Choose an output directory separate from the input directory.")
    if out.exists():
        raise FileExistsError(
            f"Output already exists: {out}\n"
            "Delete that failed/partial output intentionally or choose a new output folder."
        )

    print("STEP 5C — SAFE HASH-CLEAN VALIDATION VIEW")
    print(f"Input : {root}")
    print(f"Output: {out}")
    print("Policy: original TRAIN stays TRAIN; only original VALIDATION can remain VALIDATION.")
    print("TEST image bytes will NOT be opened, hashed, or decoded.")

    if args.validation_fraction is not None:
        print(
            f"Note: --validation-fraction {args.validation_fraction} is ignored intentionally; "
            "a fresh post-hoc 80/20 resplit would contaminate validation with "
            "Step-5 training-fitted preprocessing."
        )

    summary, labels, original_splits = validate_input(root)

    print("\n1/4 Reading saved train/validation PNG hashes from image_index.csv...")
    plan = read_index_and_plan(root, labels, original_splits)

    print(
        "Original validation images seen in original training: "
        f"{sum(plan['seen_counts']):,}"
    )
    print(
        f"  benign={plan['seen_counts'][0]:,}, malicious={plan['seen_counts'][1]:,}"
    )
    print(
        "Original validation images unseen in original training: "
        f"{sum(plan['unseen_counts']):,}"
    )
    print(
        f"  benign={plan['unseen_counts'][0]:,}, malicious={plan['unseen_counts'][1]:,}"
    )

    if min(plan["unseen_counts"]) < 1:
        raise ValueError(
            "The exact-PNG-novel validation subset does not contain both classes. "
            "Do not train/evaluate with it as a binary validation set."
        )

    print("\n2/4 Building updated split assignments...")
    new_splits = np.array(original_splits, copy=True)

    # Only original validation windows with hashes already represented in train
    # are moved into train. No original train window ever enters validation.
    for window_id in plan["move_to_train"]:
        if int(original_splits[window_id]) != 1:
            raise ValueError(f"Window {window_id} was not originally validation.")
        new_splits[window_id] = 0

    # Prove the safety invariant explicitly.
    original_train_ids = set(np.flatnonzero(original_splits == 0).tolist())
    new_validation_ids = set(np.flatnonzero(new_splits == 1).tolist())
    if original_train_ids & new_validation_ids:
        raise RuntimeError(
            "Safety invariant failed: an original training window entered validation."
        )

    new_counts = class_counts(labels, new_splits)

    print(f"New class counts: {new_counts}")

    print("\n3/4 Creating lightweight Step-5-compatible output...")
    out.mkdir(parents=True)

    # labels.npy is immutable and needed by Step 6.
    labels_mode = link_or_copy_file(root / "labels.npy", out / "labels.npy")
    atomic_npy(out / "splits.npy", new_splits)

    # Keep a few small provenance/config files when present.
    copied_meta = {}
    for name in ("generation_config.json", "feature_activity.npy"):
        src = root / name
        if src.is_file():
            copied_meta[name] = link_or_copy_file(src, out / name)

    images_reference = create_images_reference(root / "images", out / "images")

    write_index(
        out / "image_index.csv",
        plan["fieldnames"],
        plan["rows"],
        new_splits,
    )

    # Update only the split-dependent Step-5 summary fields; image-generation
    # configuration and archive metadata remain unchanged.
    new_summary = dict(summary)
    new_summary["window_class_counts"] = new_counts
    notes = list(new_summary.get("notes", []))
    notes.extend(
        [
            "STEP 5C VIEW: original training windows remain training; no original training window was moved to validation.",
            "STEP 5C VIEW: validation contains only original-validation windows whose saved PNG SHA-256 was unseen in original training.",
            "STEP 5C VIEW: original-validation windows with PNG hashes seen in training were reassigned to training.",
            "STEP 5C VIEW: Step-5 image bytes, normalization, correlation filtering and rendering were not regenerated.",
            "STEP 5C VIEW: test split assignments and test image bytes were left untouched.",
            "STEP 5C VIEW: this is an exact encoded-PNG novelty filter; distinct hashes can still represent visually identical or near-identical decoded inputs.",
        ]
    )
    new_summary["notes"] = notes
    new_summary["step05c_hashclean_validation"] = {
        "method": "keep_original_train_filter_original_validation_by_training_png_sha256",
        "source_step05_dir": str(root),
        "original_splits_sha256": file_sha256(root / "splits.npy"),
        "original_image_index_sha256": file_sha256(root / "image_index.csv"),
        "original_validation_images": int(plan["original_validation_count"]),
        "validation_images_seen_in_original_training": {
            "total": int(sum(plan["seen_counts"])),
            "benign": int(plan["seen_counts"][0]),
            "malicious": int(plan["seen_counts"][1]),
        },
        "validation_images_unseen_in_original_training": {
            "total": int(sum(plan["unseen_counts"])),
            "benign": int(plan["unseen_counts"][0]),
            "malicious": int(plan["unseen_counts"][1]),
        },
        "original_train_windows_moved_to_validation": 0,
        "test_assignments_changed": 0,
        "test_images_opened_or_decoded": 0,
        "saved_png_hash_metadata_used": True,
    }
    atomic_json(out / "step05_summary.json", new_summary)

    print("\n4/4 Verifying exact train/validation hash separation...")
    verification = verify_hash_separation(out / "image_index.csv")

    expected_non_test = {
        "train": new_counts["train"],
        "validation": new_counts["validation"],
    }
    if verification["counts_from_index"] != expected_non_test:
        raise RuntimeError(
            "Updated image_index.csv counts do not match updated splits.npy.\n"
            f"Index : {verification['counts_from_index']}\n"
            f"Splits: {expected_non_test}"
        )

    if verification["train_validation_hash_overlap"] != 0:
        raise RuntimeError(
            "Hash leakage remains after filtering. "
            f"Examples: {verification['sample_overlapping_hashes']}"
        )

    audit = {
        "step": "05c_exact_png_novel_validation_view",
        "input_dir": str(root),
        "output_dir": str(out),
        "validation_fraction_argument": args.validation_fraction,
        "validation_fraction_used": False,
        "reason_no_posthoc_resplit": (
            "Step-5 preprocessing was fitted on original training; moving original "
            "training windows into validation would cause preprocessing leakage."
        ),
        "original_counts": summary["window_class_counts"],
        "new_counts": new_counts,
        "seen_validation_by_class": {
            "benign": int(plan["seen_counts"][0]),
            "malicious": int(plan["seen_counts"][1]),
        },
        "unseen_validation_by_class": {
            "benign": int(plan["unseen_counts"][0]),
            "malicious": int(plan["unseen_counts"][1]),
        },
        "verification": verification,
        "labels_materialization": labels_mode,
        "extra_metadata_materialization": copied_meta,
        "images_reference": images_reference,
        "test_pixels_read": 0,
        "test_predictions_performed": 0,
    }
    atomic_json(out / "step05c_hash_group_summary.json", audit)

    print("\nVERIFY")
    print(
        "Train/validation PNG hash overlap:",
        verification["train_validation_hash_overlap"],
    )
    print(
        "Original-train windows moved to validation:",
        len(original_train_ids & new_validation_ids),
    )
    print(
        "Clean validation:",
        f"{new_counts['validation']['benign']:,} benign + "
        f"{new_counts['validation']['malicious']:,} malicious = "
        f"{sum(new_counts['validation'].values()):,}",
    )
    print("Test pixels read: 0")
    print(f"Images reference: {images_reference}")
    print(f"Audit: {out / 'step05c_hash_group_summary.json'}")

    print("\nSTEP 5C COMPLETE")
    print("This output can now be passed to step06_train_dataset7.py.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nSTEP 5C FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
