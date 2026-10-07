"""Step 6b: inspect saved validation scores and contour-image label conflicts.

Save this file beside step06_train_dataset7.py, then run:
    python step06b_diagnose_dataset7.py

Requires numpy and scikit-learn, already used by Step 6. Runs on the CPU.
Reads the Step 5 image index and the BALANCED run's existing JSON/CSV exports.
It does not import torch, load model checkpoints, train, or decode any images.
Test rows in the image index are skipped before their labels/hashes are used.

The selected best epoch's predictions are audited at the original rule:
    predict malicious if probability_malicious > 0.5; otherwise benign.
Other thresholds are EXPLORATORY validation analyses, not new test results.
No threshold, checkpoint, image, label, or training setting is changed.

Identical PNG hashes are taken from Step 5's saved index; image bytes are not
re-read. Byte-identical PNGs with conflicting labels can expose information
lost during window/correlation/rendering construction. A lack of exact hash
conflicts does not establish that images are distinct or preprocessing is sound.

Outputs (a new folder, never overwrite an existing report folder):
    diagnostics/dataset7_step06b/diagnostic_summary.json
    diagnostics/dataset7_step06b/epoch_metrics.csv
    diagnostics/dataset7_step06b/validation_thresholds.csv
    diagnostics/dataset7_step06b/validation_errors.csv
    diagnostics/dataset7_step06b/conflicting_image_groups.csv

To inspect another run, use --run-dir and a NEW --output-dir. Examples:
    python step06b_diagnose_dataset7.py --run-dir trained/dataset7_step06 --output-dir diagnostics/unweighted_step06b

This diagnoses one validation fold. It performs no test or zero-day evaluation.
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError as error:
    raise SystemExit("Install dependencies in your existing environment: "
                     "python -m pip install numpy scikit-learn\n" + str(error)) from error


CLASS_NAMES = ("benign", "malicious")
THRESHOLD_FIELDS = ["threshold", "true_benign", "benign_flagged_malicious",
                    "malicious_missed", "true_malicious", "accuracy", "macro_f1",
                    "balanced_accuracy", "benign_precision", "benign_recall",
                    "malicious_precision", "malicious_recall"]
ERROR_FIELDS = ["window_id", "target", "predicted_target", "probability_malicious",
                "varying_features", "png_sha256", "train_benign_same_png",
                "train_malicious_same_png", "validation_benign_same_png",
                "validation_malicious_same_png"]
GROUP_FIELDS = ["png_sha256", "train_benign", "train_malicious", "validation_benign",
                "validation_malicious", "minimum_validation_errors_if_same_image_same_label"]
HISTORY_FIELDS = ["epoch", "train_weighted_ce", "validation_ce", "validation_macro_f1",
                  "validation_balanced_accuracy", "validation_benign_precision",
                  "validation_benign_recall", "validation_malicious_recall",
                  "validation_roc_auc_malicious", "validation_average_precision_benign",
                  "amp_skipped_updates", "selected_as_best"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def csv_rows(path, required):
    """Stream CSV records so the full 238,735-row index need not be a dataframe."""
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        yield from reader


def index_train_validation(path, expected_counts, feature_count):
    """Group only train/validation images by their stored exact PNG digest."""
    groups, validation, seen = {}, {}, set()
    counts, low_variation = Counter(), Counter()
    fields = ["window_id", "split", "target", "png_sha256", "varying_features"]
    for row in csv_rows(path, fields):
        split = row["split"]
        if split == "test":
            continue  # Never use test labels/hashes to choose an experiment.
        if split not in ("train", "validation"):
            raise ValueError(f"Unexpected split in image index: {split!r}")
        window, target = int(row["window_id"]), int(row["target"])
        varying, digest = int(row["varying_features"]), row["png_sha256"].lower()
        if window < 0 or window in seen or target not in (0, 1):
            raise ValueError(f"Invalid or repeated train/validation window: {window}")
        if not 0 <= varying <= feature_count or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid variation count/hash for window {window}")
        seen.add(window)
        counts[(split, target)] += 1
        low_variation[(split, target)] += int(varying < 2)
        group = groups.setdefault(digest, [0, 0, 0, 0])
        group[(0 if split == "train" else 2) + target] += 1
        if split == "validation":
            validation[window] = {"target": target, "png_sha256": digest,
                                  "varying_features": varying}
    for split in ("train", "validation"):
        for target, name in enumerate(CLASS_NAMES):
            if counts[(split, target)] != expected_counts[split][name]:
                raise ValueError(f"Image index count disagrees with Step 5: {split}/{name}")
            if counts[(split, target)] == 0:
                raise ValueError(f"Both classes are required in {split}.")
    low = {split: {name: low_variation[(split, c)] for c, name in enumerate(CLASS_NAMES)}
           for split in ("train", "validation")}
    return groups, validation, low


def load_predictions(path, validation):
    """Require exactly one saved score per validation window with matching labels."""
    ids, targets, scores, seen = [], [], [], set()
    for row in csv_rows(path, ["window_id", "target", "probability_malicious", "predicted_target"]):
        window, target = int(row["window_id"]), int(row["target"])
        score, prediction = float(row["probability_malicious"]), int(row["predicted_target"])
        if window not in validation or window in seen:
            raise ValueError(f"Saved predictions contain a non-validation/repeated window: {window}")
        if target != validation[window]["target"]:
            raise ValueError(f"Prediction/index target mismatch for window {window}")
        if not math.isfinite(score) or not 0 <= score <= 1 or prediction != int(score > 0.5):
            raise ValueError(f"Invalid saved probability/decision for window {window}")
        seen.add(window)
        ids.append(window)
        targets.append(target)
        scores.append(score)
    if seen != set(validation):
        raise ValueError("Saved predictions do not cover the complete validation partition.")
    return np.asarray(ids), np.asarray(targets, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def score_counts(tn, fp, fn, tp, threshold):
    """Malicious is the positive class; FN means malicious traffic called benign."""
    divide = lambda a, b: float(a / b) if b else 0.0
    benign_recall, malicious_recall = divide(tn, tn + fp), divide(tp, tp + fn)
    benign_f1, malicious_f1 = divide(2 * tn, 2 * tn + fp + fn), divide(2 * tp, 2 * tp + fp + fn)
    return {"threshold": float(threshold), "true_benign": int(tn),
            "benign_flagged_malicious": int(fp), "malicious_missed": int(fn),
            "true_malicious": int(tp), "accuracy": divide(tn + tp, tn + fp + fn + tp),
            "macro_f1": (benign_f1 + malicious_f1) / 2,
            "balanced_accuracy": (benign_recall + malicious_recall) / 2,
            "benign_precision": divide(tn, tn + fn), "benign_recall": benign_recall,
            "malicious_precision": divide(tp, tp + fp), "malicious_recall": malicious_recall}


def threshold_analysis(targets, scores):
    """Scan attainable score boundaries in O(n log n), keeping tied scores together.

    Decisions always follow score > threshold. These validation-fitted candidates
    are not independent performance estimates and are never applied to a model.
    """
    order = np.argsort(scores, kind="stable")
    sorted_scores, sorted_targets = scores[order], targets[order]
    benign_prefix = np.r_[0, np.cumsum(sorted_targets == 0)]
    n_benign, n_malicious = int(benign_prefix[-1]), int(np.count_nonzero(targets == 1))
    thresholds = np.unique(np.r_[0.0, scores, 0.5, 1.0])
    boundaries = np.searchsorted(sorted_scores, thresholds, side="right")
    rows = []
    for threshold, k in zip(thresholds, boundaries):
        tn = int(benign_prefix[k])
        fn = int(k) - tn
        rows.append(score_counts(tn, n_benign - tn, fn, n_malicious - fn, threshold))
    base = next(row for row in rows if row["threshold"] == 0.5)
    # Choose the candidate closest to 0.5 if the diagnostic macro F1 ties.
    exploratory = max(rows, key=lambda row: (row["macro_f1"], -abs(row["threshold"] - 0.5)))
    return rows, base, exploratory


def hash_diagnostics(groups, validation, ids, targets, scores):
    """Summarize exact duplicates and label conflicts, without accessing PNG bytes."""
    conflicts = []
    for digest, (tb, tm, vb, vm) in groups.items():
        if (tb + vb) and (tm + vm):
            conflicts.append(dict(zip(GROUP_FIELDS, [digest, tb, tm, vb, vm, min(vb, vm)])))
    conflicts.sort(key=lambda row: (-(row["validation_benign"] + row["validation_malicious"]), row["png_sha256"]))
    errors = []
    for window, target, score in zip(ids.tolist(), targets.tolist(), scores.tolist()):
        prediction = int(score > 0.5)
        if prediction != target:
            item = validation[window]
            values = [window, target, prediction, score, item["varying_features"], item["png_sha256"],
                      *groups[item["png_sha256"]]]
            errors.append(dict(zip(ERROR_FIELDS, values)))
    report = {
        "hash_basis": "Stored SHA-256 of encoded PNG bytes from Step 5; bytes not re-read.",
        "unique_train_png_hashes": sum(bool(g[0] + g[1]) for g in groups.values()),
        "unique_validation_png_hashes": sum(bool(g[2] + g[3]) for g in groups.values()),
        "validation_images_matching_any_training_png": sum(g[2] + g[3] for g in groups.values() if g[0] + g[1]),
        "validation_benign_matching_training_malicious_png": sum(g[2] for g in groups.values() if g[1]),
        "validation_malicious_matching_training_benign_png": sum(g[3] for g in groups.values() if g[0]),
        "training_hash_groups_with_both_labels": sum(bool(g[0] and g[1]) for g in groups.values()),
        "validation_hash_groups_with_both_labels": sum(bool(g[2] and g[3]) for g in groups.values()),
        "train_validation_hash_groups_with_both_labels": len(conflicts),
        "minimum_validation_errors_if_same_image_same_label": sum(min(g[2], g[3]) for g in groups.values()),
        "malicious_misses_with_png_also_used_by_benign_training": sum(
            row["target"] == 1 and row["train_benign_same_png"] > 0 for row in errors),
        "interpretation": "The minimum-error count is an empirical constraint if identical images receive identical labels; it is not a generalization bound. Different encoded hashes can still have identical or similar pixels.",
    }
    return report, conflicts, errors


def epoch_metrics(history):
    rows = []
    for item in history:
        train, val = item["training"], item["validation"]
        values = [item["epoch"], train["weighted_cross_entropy"], val["cross_entropy"],
                  val["macro_f1"], val["balanced_accuracy"], val["per_class"]["benign"]["precision"],
                  val["per_class"]["benign"]["recall"], val["per_class"]["malicious"]["recall"],
                  val["roc_auc_malicious"], val["average_precision_benign"],
                  train["amp_skipped_updates"], item["selected_as_best"]]
        rows.append(dict(zip(HISTORY_FIELDS, values)))
    return rows


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    root, run_dir, out = (p.expanduser().resolve() for p in (args.input_dir, args.run_dir, args.output_dir))
    if out.exists():
        raise ValueError("Diagnostic output folder already exists. Use a NEW --output-dir.")
    if out.is_relative_to(root) or out.is_relative_to(run_dir):
        raise ValueError("Keep diagnostic outputs outside the input and training folders.")
    step5 = read_json(root / "step05_summary.json")
    step6 = read_json(run_dir / "step06_summary.json")
    config = read_json(run_dir / "training_config.json")
    marker = read_json(run_dir / "best_export.json")
    history = read_json(run_dir / "history.json")
    if step5.get("step") != "05_contours_single_iot23_csv" or step6.get("step") != "06_binary_resnet50_transformer_training":
        raise ValueError("Expected completed Step 5 and Step 6 outputs.")
    if config["step05_configuration_hash"] != step5["configuration_hash"]:
        raise ValueError("This training run was created from a different Step 5 configuration.")
    if marker != {"epoch": step6["best_epoch"], "config_hash": step6["configuration_hash"]}:
        raise ValueError("Best-epoch exports and summary disagree. Finish the training run first.")
    if not history or history[-1]["epoch"] != step6["epochs_completed"]:
        raise ValueError("History and summary disagree. Finish the training run first.")
    selected = [item for item in history if item["epoch"] == step6["best_epoch"]]
    if len(selected) != 1 or selected[0]["validation"] != step6["best_validation_metrics"]:
        raise ValueError("Selected epoch history and summary disagree.")

    print("1/3 Checking train/validation image metadata (test rows ignored)...", flush=True)
    groups, validation, low_variation = index_train_validation(
        root / "image_index.csv", step5["window_class_counts"], len(step5["feature_order"]))
    for split in ("train", "validation"):
        if config["window_class_counts"][split] != step5["window_class_counts"][split]:
            raise ValueError("Training and image metadata have different partition counts.")
    print("2/3 Reading saved best-epoch validation scores and exploring thresholds...", flush=True)
    ids, targets, scores = load_predictions(run_dir / "best_validation_predictions.csv", validation)
    curves, base, exploratory = threshold_analysis(targets, scores)
    matrix = [[base["true_benign"], base["benign_flagged_malicious"]],
              [base["malicious_missed"], base["true_malicious"]]]
    if matrix != step6["best_validation_metrics"]["confusion_matrix"]:
        raise ValueError("CSV scores do not reproduce the saved best-epoch confusion matrix.")
    for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
        if not math.isclose(base[metric], step6["best_validation_metrics"][metric], abs_tol=1e-10):
            raise ValueError(f"CSV scores do not reproduce saved {metric}.")
    hash_report, conflict_rows, errors = hash_diagnostics(groups, validation, ids, targets, scores)
    history_rows = epoch_metrics(history)
    score_summaries = {}
    for target, name in enumerate(CLASS_NAMES):
        values = scores[targets == target]
        score_summaries[name] = {
            "n": len(values), "unique_scores": len(np.unique(values)),
            "probability_malicious_quantiles": dict(zip(
                ["min", "p01", "p25", "median", "p75", "p99", "max"],
                np.quantile(values, [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]).tolist()))}
    report = {
        "step": "06b_validation_score_and_image_hash_diagnostic", "training_run": str(run_dir),
        "selected_best_epoch": step6["best_epoch"], "epochs_completed": step6["epochs_completed"],
        "class_weighting": config["class_weighting"], "validation_n": len(ids),
        "default_decision_rule": "probability_malicious > 0.5; equality predicts benign",
        "default_threshold_metrics": base, "confusion_matrix_actual_rows_predicted_columns": matrix,
        "confusion_order": list(CLASS_NAMES), "probability_summaries": score_summaries,
        "ranking_metrics": {
            "roc_auc_malicious": float(roc_auc_score(targets, scores)),
            "average_precision_malicious": float(average_precision_score(targets, scores)),
            "average_precision_benign": float(average_precision_score(1 - targets, 1 - scores))},
        "exploratory_validation_best_macro_f1_threshold": exploratory,
        "threshold_applied_to_model": False, "epoch_metrics": history_rows,
        "image_hash_diagnostics": hash_report, "fewer_than_two_varying_features": low_variation,
        "test_images_read": 0, "test_predictions_used": 0, "zero_day_evaluated": False,
        "notes": [
            "Per-image probabilities are from the selected best epoch only, not necessarily the latest epoch.",
            "Equal confusion counts across epochs do not establish identical error IDs or unchanged probabilities.",
            "The threshold search uses validation labels; its best score is exploratory and optimistically selected, not an independent evaluation.",
            "Neither exact duplicates nor class imbalance alone prove the cause of the observed errors.",
            "The original model, threshold, datasets and checkpoints have not been changed.",
        ]}
    print("3/3 Writing diagnostic reports to a new folder...", flush=True)
    # Create outputs only after the consistency checks and all analysis succeed.
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "epoch_metrics.csv", HISTORY_FIELDS, history_rows)
    write_csv(out / "validation_thresholds.csv", THRESHOLD_FIELDS, curves)
    write_csv(out / "validation_errors.csv", ERROR_FIELDS, errors)
    write_csv(out / "conflicting_image_groups.csv", GROUP_FIELDS, conflict_rows)
    (out / "diagnostic_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print("\nSTEP 6B DIAGNOSTIC COMPLETE")
    print(f"Scores checked: selected epoch {step6['best_epoch']}; {len(ids):,} validation images")
    print(f"Default 0.5: macro F1={base['macro_f1']:.6f}; benign recall={base['benign_recall']:.6f}; "
          f"benign precision={base['benign_precision']:.6f}; malicious missed={base['malicious_missed']:,}")
    print(f"Exploratory validation-only threshold={exploratory['threshold']:.9g}; "
          f"macro F1={exploratory['macro_f1']:.6f}; benign recall={exploratory['benign_recall']:.6f}; "
          f"malicious missed={exploratory['malicious_missed']:,} (NOT applied)")
    print(f"Validation PNG hash groups with both labels: {hash_report['validation_hash_groups_with_both_labels']:,}")
    print(f"Validation images matching a training PNG: {hash_report['validation_images_matching_any_training_png']:,}")
    print(f"Malicious misses matching a benign training PNG: {hash_report['malicious_misses_with_png_also_used_by_benign_training']:,}")
    print(f"Summary: {out / 'diagnostic_summary.json'}")
    print("No training, checkpoint changes, image decoding, or test evaluation performed.")


def main():
    parser = argparse.ArgumentParser(description="CPU-only Step 6b diagnostic of saved validation scores and image hashes.")
    parser.add_argument("--input-dir", type=Path, default=Path("prepared/dataset7_step05"))
    parser.add_argument("--run-dir", type=Path, default=Path("trained/dataset7_step06_balanced_trial01"))
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/dataset7_step06b"))
    try:
        run(parser.parse_args())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"\nSTEP 6B FAILED: {error}", file=sys.stderr)
        print("Use the completed Step 5 and Step 6 folders; preserve the original files.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
