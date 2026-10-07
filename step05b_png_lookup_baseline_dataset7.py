"""Step 5b: measure PNG repetition and an exact-match validation baseline.

Run from iot_zero_day_research:
    python step05b_png_lookup_baseline_dataset7.py

No extra packages are required; this script uses Python's standard library.
Default input: prepared/dataset7_step05_no_uid
Default output: diagnostics/dataset7_step05b_no_uid

Why this baseline matters
-------------------------
Different source windows can become identical contour PNGs. A model can then
recognize a repeated image even though no connection row crosses partitions.
This diagnostic measures that simpler explanation for validation performance.

Prediction rules, fixed before using any validation targets:
1. For a PNG hash seen in training, predict its majority TRAINING label.
2. For an unseen hash, predict the overall majority TRAINING label.
3. Resolve a within-hash training tie using the overall training majority.
   If the overall training counts tie, choose benign (label 0).

Validation labels are used only for counts, overlap composition and scoring.
Test rows are skipped before their IDs, targets or hashes are interpreted.
No test prediction is computed. No PNG or ZIP is opened: the saved SHA-256
metadata is used, so its integrity is assumed rather than reverified here.
No image, assignment, feature, checkpoint or model setting is changed.

Outputs:
    png_lookup_summary.json
    validation_lookup_predictions.csv  (also records seen/unseen membership)
    train_validation_png_groups.csv

An existing report folder is preserved; a numbered sibling is used instead.
Distinct encoded hashes can still have identical/similar decoded pixels. Exact
hash novelty is not independence, a new attack family, or zero-day novelty.
Subset macro F1 and balanced accuracy are reported as null when either actual
class is absent; class counts and the confusion matrix are always provided.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile


CLASS_NAMES = ("benign", "malicious")
PARTITIONS = ("train", "validation")
PREDICTION_FIELDS = (
    "window_id", "target", "png_sha256", "training_hash_found",
    "lookup_prediction", "train_benign_same_png", "train_malicious_same_png",
    "validation_images_same_png",
)
GROUP_FIELDS = (
    "png_sha256", "train_benign", "train_malicious", "validation_benign",
    "validation_malicious", "both_labels_in_train_validation",
)


def read_index(root, summary):
    """Collect train/validation metadata, with four counts per PNG hash."""
    groups, validation, seen_ids = {}, [], set()
    counts = {name: [0, 0] for name in PARTITIONS}
    with (root / "image_index.csv").open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"window_id", "split", "target", "png_sha256"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Image index lacks required columns.")
        for row in reader:
            split = row["split"]
            if split == "test":
                continue
            if split not in PARTITIONS:
                raise ValueError(f"Unexpected partition: {split!r}")
            window_id, target = int(row["window_id"]), int(row["target"])
            if not 0 <= window_id < summary["images"] or window_id in seen_ids:
                raise ValueError(f"Invalid or repeated train/validation window ID: {window_id}")
            if target not in (0, 1):
                raise ValueError("Targets must be 0=benign or 1=malicious.")
            digest = row["png_sha256"].strip().lower()
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Invalid saved PNG SHA-256 for window {window_id}")
            seen_ids.add(window_id)
            counts[split][target] += 1
            group = groups.setdefault(digest, [0, 0, 0, 0])
            group[(0 if split == "train" else 2) + target] += 1
            if split == "validation":
                validation.append((window_id, target, digest))
    for split in PARTITIONS:
        expected = summary["window_class_counts"][split]
        if counts[split] != [expected[name] for name in CLASS_NAMES]:
            raise ValueError(f"{split} counts disagree with the Step 5 summary.")
        if min(counts[split]) < 1:
            raise ValueError(f"This baseline expects both classes in {split}.")
    return groups, sorted(validation), counts


def choose_label(training_counts, training_majority):
    """This function never receives a validation target."""
    benign, malicious = training_counts
    if benign == malicious:
        return training_majority
    return int(malicious > benign)


def metrics(matrix):
    """Actual classes are rows; predicted classes are columns, ordered [0, 1]."""
    support = [sum(row) for row in matrix]
    total = sum(support)
    by_class = {}
    for label, name in enumerate(CLASS_NAMES):
        true_positive = matrix[label][label]
        false_negative = matrix[label][1 - label]
        false_positive = matrix[1 - label][label]
        predicted_count = true_positive + false_positive
        denominator = 2 * true_positive + false_negative + false_positive
        by_class[name] = {
            "support": support[label],
            "precision": true_positive / predicted_count if predicted_count else 0.0,
            "recall": true_positive / support[label] if support[label] else None,
            "f1": (2 * true_positive / denominator if denominator else 0.0)
                  if support[label] else None,
        }
    both_present = min(support) > 0
    return {
        "images": total,
        "confusion_matrix_actual_rows_predicted_columns": matrix,
        "confusion_order": list(CLASS_NAMES),
        "accuracy": (matrix[0][0] + matrix[1][1]) / total if total else None,
        "macro_f1": sum(by_class[name]["f1"] for name in CLASS_NAMES) / 2 if both_present else None,
        "balanced_accuracy": sum(by_class[name]["recall"] for name in CLASS_NAMES) / 2 if both_present else None,
        "both_actual_classes_present": both_present,
        "per_class": by_class,
    }


def analyze(root):
    summary = json.loads((root / "step05_summary.json").read_text(encoding="utf-8-sig"))
    if summary.get("step") != "05_contours_single_iot23_csv":
        raise ValueError("Use a completed Step 5 output folder.")
    print("1/2 Reading train/validation PNG hash metadata; test rows skipped...", flush=True)
    groups, validation, counts = read_index(root, summary)
    # Derive the complete prediction dictionary exclusively from training labels.
    training_majority = int(counts["train"][1] > counts["train"][0])
    lookup = {digest: choose_label(group[:2], training_majority)
              for digest, group in groups.items() if group[0] + group[1]}
    print("2/2 Scoring the training-derived lookup on validation...", flush=True)
    matrices = {name: [[0, 0], [0, 0]] for name in ("all", "seen_png", "unseen_png", "always_training_majority")}
    predictions = []
    seen_by_class = [0, 0]
    for window_id, target, digest in validation:
        found = digest in lookup
        prediction = lookup.get(digest, training_majority)
        subset = "seen_png" if found else "unseen_png"
        matrices["all"][target][prediction] += 1
        matrices[subset][target][prediction] += 1
        matrices["always_training_majority"][target][training_majority] += 1
        seen_by_class[target] += int(found)
        group = groups[digest]
        predictions.append(dict(zip(PREDICTION_FIELDS, (
            window_id, target, digest, int(found), prediction, group[0], group[1], group[2] + group[3],
        ))))

    partitions = {}
    for split, offset in (("train", 0), ("validation", 2)):
        relevant = [group for group in groups.values() if group[offset] + group[offset + 1]]
        partitions[split] = {
            "images": sum(counts[split]),
            "class_counts": dict(zip(CLASS_NAMES, counts[split])),
            "distinct_png_hashes": len(relevant),
            "distinct_png_hashes_by_class": {name: sum(group[offset + label] > 0 for group in relevant)
                                             for label, name in enumerate(CLASS_NAMES)},
            "hash_groups_with_both_labels": sum(group[offset] > 0 and group[offset + 1] > 0 for group in relevant),
            "largest_hash_group_images": max(group[offset] + group[offset + 1] for group in relevant),
        }
    novel_by_class = [counts["validation"][label] - seen_by_class[label] for label in (0, 1)]
    report = {
        "step": "05b_training_png_lookup_validation_baseline",
        "input_dir": str(root),
        "step05_configuration_hash": summary["configuration_hash"],
        "validation_fold": summary["validation_fold"],
        "feature_order": summary["feature_order"],
        "partitions": partitions,
        "validation_overlap": {
            "images_seen_in_training": sum(seen_by_class),
            "fraction_seen_in_training": sum(seen_by_class) / len(validation),
            "seen_by_actual_class": dict(zip(CLASS_NAMES, seen_by_class)),
            "unseen_by_actual_class": dict(zip(CLASS_NAMES, novel_by_class)),
            "hash_groups_shared_by_train_and_validation": sum(
                bool(group[0] + group[1]) and bool(group[2] + group[3]) for group in groups.values()),
            "train_validation_hash_groups_with_both_labels": sum(
                bool(group[0] + group[2]) and bool(group[1] + group[3]) for group in groups.values()),
        },
        "prediction_rule": {
            "seen_hash": "Majority training label for that hash; a tie uses the overall training majority.",
            "unseen_hash": "Overall training majority label; a global tie chooses benign.",
            "training_majority_label": training_majority,
            "training_majority_class": CLASS_NAMES[training_majority],
            "validation_targets_used_to_fit_predictions": 0,
        },
        "lookup_validation_metrics": {name: metrics(matrices[name]) for name in ("all", "seen_png", "unseen_png")},
        "always_training_majority_validation_metrics": metrics(matrices["always_training_majority"]),
        "test_predictions_computed": 0,
        "test_targets_or_hashes_used": 0,
        "png_bytes_read": 0,
        "neural_network_trained": False,
        "datasets_or_checkpoints_changed": False,
        "notes": [
            "This is an exact-PNG lookup baseline, not CZ-ResViT performance.",
            "Saved hashes are trusted metadata; PNG bytes were not rehashed.",
            "Identical image hashes do not imply that the same source rows were reused.",
            "Distinct hashes can still correspond to identical or similar decoded pixels.",
            "Unseen PNG hashes are not necessarily independent captures or unseen attack families.",
            "No new split, duplicate removal, rendering change, or threshold tuning is performed.",
            "Macro F1 and balanced accuracy are null for subsets missing an actual class.",
        ],
    }
    group_rows = [dict(zip(GROUP_FIELDS, (digest, *group, int(bool(group[0] + group[2]) and bool(group[1] + group[3])))))
                  for digest, group in sorted(groups.items())]
    return report, predictions, group_rows


def save_reports(output, report, predictions, group_rows):
    output.parent.mkdir(parents=True, exist_ok=True)
    requested, suffix = output, 1
    while output.exists():
        output = requested.with_name(f"{requested.name}_{suffix:02d}")
        suffix += 1
    stage = Path(tempfile.mkdtemp(prefix="png_lookup_", dir=output.parent))
    try:
        with (stage / "png_lookup_summary.json").open("w", encoding="utf-8") as destination:
            json.dump(report, destination, indent=2, allow_nan=False)
            destination.write("\n")
        for name, fields, rows in (
            ("validation_lookup_predictions.csv", PREDICTION_FIELDS, predictions),
            ("train_validation_png_groups.csv", GROUP_FIELDS, group_rows),
        ):
            with (stage / name).open("w", encoding="utf-8", newline="") as destination:
                writer = csv.DictWriter(destination, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return output


def main():
    parser = argparse.ArgumentParser(description="Step 5b: training-only PNG lookup, scored on validation.")
    parser.add_argument("--input-dir", type=Path, default=Path("prepared/dataset7_step05_no_uid"))
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/dataset7_step05b_no_uid"))
    args = parser.parse_args()
    root, output = args.input_dir.resolve(), args.output_dir.resolve()
    try:
        if output == root or root in output.parents or output in root.parents:
            raise ValueError("Choose an output folder separate from the Step 5 input folder.")
        report, predictions, group_rows = analyze(root)
        output = save_reports(output, report, predictions, group_rows)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"PNG lookup audit stopped: {error}", file=sys.stderr)
        return 1
    print("\nSTEP 5B LOOKUP BASELINE COMPLETE")
    for split in PARTITIONS:
        info = report["partitions"][split]
        print(f"{split}: {info['images']:,} images; {info['distinct_png_hashes']:,} distinct PNG hashes")
        print(f"  Distinct hashes by class: {info['distinct_png_hashes_by_class']}")
    overlap = report["validation_overlap"]
    print(f"Validation PNGs seen in training: {overlap['images_seen_in_training']:,} "
          f"({overlap['fraction_seen_in_training']:.2%})")
    print(f"  Seen by class: {overlap['seen_by_actual_class']}")
    print(f"  Unseen by class: {overlap['unseen_by_actual_class']}")
    score = report["lookup_validation_metrics"]["all"]
    print(f"Lookup VALIDATION baseline: accuracy={score['accuracy']:.6f}; macro F1={score['macro_f1']:.6f}; "
          f"benign recall={score['per_class']['benign']['recall']:.6f}")
    print("This is a dictionary baseline, not neural-network performance.")
    print(f"Summary: {output / 'png_lookup_summary.json'}")
    print("No test evaluation, image changes or checkpoint changes performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
