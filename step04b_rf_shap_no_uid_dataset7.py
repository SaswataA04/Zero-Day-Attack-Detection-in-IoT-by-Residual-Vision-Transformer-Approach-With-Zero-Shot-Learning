"""Step 4b: select 15 features with UID excluded, reusing the saved split.

Run from iot_zero_day_research:
    python step04b_rf_shap_no_uid_dataset7.py

If dependencies are missing:
    python -m pip install numpy scikit-learn shap joblib

Inputs:
    prepared/dataset7_step03       Original 17-column numerical storage.
    prepared/dataset7_step04       Original memberships, RF settings and SHAP rows.
Output:
    prepared/dataset7_step04_no_uid (must be new or empty).

Why this correction is needed
-----------------------------
Every connection in this dataset has a distinct UID. A training-fitted category
vocabulary gives training UIDs distinct codes but maps validation UIDs to one
unknown value. UID therefore varies in training windows and is constant in
validation windows. This difference propagates into the correlation images.

UID is retained in the original storage/provenance, but excluded BEFORE fitting
the new RF selector. The remaining 16 candidates compete for 15 selected slots.
This is a documented pipeline correction, not an unchanged reproduction of the
earlier candidate-feature set. It does not guarantee better detector metrics.

What is preserved
-----------------
* Exact 15-row window memberships, targets, train/validation/test assignments,
  folds and window manifest are copied from the original Step 4.
* The same training rows fit category vocabularies and numeric medians.
* RF hyperparameters and the exact balanced SHAP explanation rows are reused.
* Unknown=-1, missing=-2, training-known category codes=0..K-1.
* No rows are removed. The original incomplete tail stays unused for images.
* Test/validation rows never enter RF fitting, preprocessing fitting, or SHAP.

The script transforms all rows for later image generation using training-only
parameters. It does not read existing images, run the neural network, change
checkpoints, or evaluate test/zero-day performance. All original files remain
in place. The output schema is compatible with the existing Step 5 script;
image regeneration is a separate next step.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

try:
    import joblib
    import numpy as np
    import shap
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
except ImportError as error:
    raise SystemExit(
        "Missing dependency. Run: python -m pip install numpy scikit-learn shap joblib"
    ) from error

WINDOW_ROWS = 15
N_FOLDS = 5
TRAIN, VALIDATION, TEST, UNUSED = 0, 1, 2, -1
SPLIT_NAMES = {TRAIN: "train", VALIDATION: "validation", TEST: "test", UNUSED: "unused_tail"}
CLASS_NAMES = {0: "benign", 1: "malicious"}
CAT_MISSING, CAT_UNKNOWN = -2, -1
PRESERVED_FILES = (
    "row_split.npy", "window_split.npy", "window_targets.npy", "window_folds.npy",
    "window_manifest.csv", "shap_row_indices.npy",
)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_array(path, values):
    # An explicit file handle avoids filename-extension surprises.
    with path.open("wb") as destination:
        np.save(destination, values, allow_pickle=False)


def close_map(array, writable=False):
    if array is not None and isinstance(array, np.memmap):
        if writable:
            array.flush()
        array._mmap.close()


def binary_counts(values):
    counts = np.bincount(np.asarray(values, dtype=np.int64), minlength=2)
    return {CLASS_NAMES[i]: int(counts[i]) for i in range(2)}


def fit_preprocessors(X, source_columns, features, categorical, train_indices):
    """Learn only from fitting rows; storage IDs are NOT used directly by RF."""
    state = {}
    known_codes = {}
    for index, name in zip(source_columns, features):
        values = np.asarray(X[train_indices, index], dtype=np.float64)
        if name in categorical:
            if not np.isfinite(values).all() or np.any(values < -1) or np.any(values != np.floor(values)):
                raise ValueError(f"Invalid categorical storage codes in {name}.")
            known = np.unique(values[values >= 0].astype(np.int64))
            known_codes[name] = known
            state[name] = {
                "kind": "categorical", "known_count": len(known),
                "known_storage_codes_file": "known_" + name.replace(".", "_") + ".npy",
                "missing_code": CAT_MISSING, "unknown_code": CAT_UNKNOWN,
            }
        else:
            if np.isinf(values).any():
                raise ValueError(f"Infinite training values in {name}.")
            observed = values[np.isfinite(values)]
            median = float(np.median(observed)) if len(observed) else 0.0
            state[name] = {
                "kind": "numeric", "median": median,
                "all_missing_training_fallback": len(observed) == 0,
                "training_missing_count": int(np.isnan(values).sum()),
            }
        print(f"  Fitted preprocessing for {name}", flush=True)
    return state, known_codes


def transform_block(block, features, state, known_codes):
    """Use the saved training parameters for fitting, validation and test rows."""
    result = np.array(block, dtype=np.float64, copy=True)
    for index, name in enumerate(features):
        values = result[:, index]
        settings = state[name]
        if settings["kind"] == "numeric":
            if np.isinf(values).any():
                raise ValueError(f"Infinite numeric values in {name}.")
            values[np.isnan(values)] = settings["median"]
        else:
            if not np.isfinite(values).all() or np.any(values < -1) or np.any(values != np.floor(values)):
                raise ValueError(f"Invalid categorical storage values in {name}.")
            missing = values == -1
            known = known_codes[name]
            positions = np.searchsorted(known, values)
            matches = np.zeros(len(values), dtype=bool)
            in_range = positions < len(known)
            matches[in_range] = known[positions[in_range]] == values[in_range]
            encoded = np.full(len(values), CAT_UNKNOWN, dtype=np.float64)
            encoded[matches] = positions[matches]
            encoded[missing] = CAT_MISSING
            result[:, index] = encoded
    if not np.isfinite(result).all():
        raise ValueError("Preprocessing produced nonfinite values.")
    return result


def positive_class_shap(values, n_rows, n_features, positive_index):
    """Support both legacy lists and current [rows, features, outputs] arrays."""
    if isinstance(values, list):
        result = np.asarray(values[positive_index])
    else:
        array = np.asarray(values)
        if array.ndim != 3 or array.shape[:2] != (n_rows, n_features):
            raise ValueError(f"Unexpected Random Forest SHAP shape: {array.shape}")
        result = array[:, :, positive_index]
    if result.shape != (n_rows, n_features) or not np.isfinite(result).all():
        raise ValueError("SHAP values have an invalid shape or contain nonfinite values.")
    return result

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_membership(y, row_split, splits, targets, folds, baseline, shap_indices):
    """Check saved memberships against Step 3; never generate a replacement split."""
    total = len(y)
    n_windows = total // WINDOW_ROWS
    used = n_windows * WINDOW_ROWS
    if n_windows < 1:
        raise ValueError("No complete windows.")
    if row_split.shape != (total,) or any(a.shape != (n_windows,) for a in (splits, targets, folds)):
        raise ValueError("Saved membership shapes disagree with Step 3.")
    if any(a.dtype.kind not in "iu" for a in (row_split, splits, targets, folds, shap_indices)):
        raise ValueError("Membership and explanation indices must be integer arrays.")
    if not np.isin(splits, [TRAIN, VALIDATION, TEST]).all():
        raise ValueError("Unexpected saved window split codes.")
    if not np.array_equal(row_split[:used].reshape(-1, WINDOW_ROWS),
                          np.broadcast_to(splits[:, None], (n_windows, WINDOW_ROWS))):
        raise ValueError("A saved window contains rows from different partitions.")
    if not np.all(row_split[used:] == UNUSED):
        raise ValueError("Incomplete tail rows must remain unused.")
    if not np.array_equal(targets, y[WINDOW_ROWS - 1:used:WINDOW_ROWS]):
        raise ValueError("Saved final-row targets no longer match Step 3.")
    fold = baseline["validation_fold"]
    if fold not in range(N_FOLDS):
        raise ValueError("Unexpected validation fold.")
    if (not np.all(folds[splits == TEST] == -1)
            or not np.all(folds[splits == VALIDATION] == fold)
            or not np.isin(folds[splits == TRAIN], [f for f in range(N_FOLDS) if f != fold]).all()):
        raise ValueError("Saved folds and partition assignments disagree.")
    for code in (TRAIN, VALIDATION, TEST):
        if binary_counts(targets[splits == code]) != baseline["window_class_counts"][SPLIT_NAMES[code]]:
            raise ValueError("Window class counts no longer match the original Step 4.")
    digest = hashlib.sha256(row_split.tobytes()).hexdigest()
    if digest != baseline["row_split_sha256"]:
        raise ValueError("Saved row assignments do not match their original fingerprint.")
    if (shap_indices.ndim != 1 or len(shap_indices) < 2
            or np.any(shap_indices < 0) or np.any(shap_indices >= total)
            or len(np.unique(shap_indices)) != len(shap_indices)):
        raise ValueError("Invalid saved SHAP explanation indices.")
    if not np.all(row_split[shap_indices] == TRAIN):
        raise ValueError("Saved SHAP explanation rows include a held-out row.")
    expected = int(baseline["shap_rows_per_class"])
    if binary_counts(y[shap_indices]) != {"benign": expected, "malicious": expected}:
        raise ValueError("Saved SHAP rows no longer match the original balanced sample.")
    return n_windows, used


def run(input_dir, baseline_dir, output_dir, jobs=None):
    input_dir = Path(input_dir).expanduser().resolve()
    baseline_dir = Path(baseline_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    for source in (input_dir, baseline_dir):
        if output_dir == source or source in output_dir.parents or output_dir in source.parents:
            raise ValueError("Choose an output folder separate from both input folders.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty. Choose a new --output-dir.")
    if jobs is not None and jobs < 1:
        raise ValueError("--jobs must be positive.")

    source_summary = read_json(input_dir / "step03_summary.json")
    baseline = read_json(baseline_dir / "step04_summary.json")
    if source_summary.get("step") != "03_numeric_storage_single_iot23_csv":
        raise ValueError("Use the original Step 3 numerical storage.")
    if (baseline.get("step") != "04_rf_shap_single_iot23_csv"
            or baseline["window_rows"] != WINDOW_ROWS or baseline["stride"] != WINDOW_ROWS):
        raise ValueError("Use the original Step 4 with 15-row, nonoverlapping windows.")
    if baseline["source_file"] != source_summary["source_file"]:
        raise ValueError("Step 3 and the baseline summary name different source files.")
    all_features = source_summary["feature_names"]
    if (len(all_features) != 17 or len(set(all_features)) != 17
            or "uid" not in all_features or "ts" not in all_features
            or set(all_features) & {"label", "target", "source_row", "source_file",
                                   "detailed-label", "label_original"}):
        raise ValueError("Unexpected original 17-feature schema.")
    if not set(source_summary["categorical_features"]).issubset(all_features):
        raise ValueError("Invalid categorical feature definitions.")
    features = [name for name in all_features if name != "uid"]
    source_columns = [all_features.index(name) for name in features]
    categorical = [name for name in source_summary["categorical_features"] if name != "uid"]
    if len(features) != 16:
        raise ValueError("Expected exactly 16 candidates after excluding UID.")

    started = time.perf_counter()
    input_maps = []
    def mapped(folder, name):
        array = np.load(folder / name, mmap_mode="r", allow_pickle=False)
        input_maps.append(array)
        return array

    try:
        X = mapped(input_dir, "X_storage.npy")
        y = mapped(input_dir, "y.npy")
        total = int(source_summary["rows_written"])
        if X.shape != (total, 17) or X.dtype != np.float64 or y.shape != (total,):
            raise ValueError("Step 3 array shapes/dtypes disagree with its summary.")
        if not np.isin(y, [0, 1]).all() or binary_counts(y) != {
                name: source_summary["label_counts"].get(name, 0) for name in CLASS_NAMES.values()}:
            raise ValueError("Step 3 labels disagree with its summary.")
        row_split = mapped(baseline_dir, "row_split.npy")
        splits = mapped(baseline_dir, "window_split.npy")
        targets = mapped(baseline_dir, "window_targets.npy")
        folds = mapped(baseline_dir, "window_folds.npy")
        shap_indices = mapped(baseline_dir, "shap_row_indices.npy")
        print("1/6 Verifying the existing window assignments and SHAP rows...", flush=True)
        n_windows, used = validate_membership(
            y, row_split, splits, targets, folds, baseline, shap_indices)
        train_indices = np.flatnonzero(row_split == TRAIN)
        train_labels = np.asarray(y[train_indices])
        if len(np.unique(train_labels)) != 2:
            raise ValueError("Both classes are required in the fitting rows.")
        preserved_hashes = {name: file_sha256(baseline_dir / name) for name in PRESERVED_FILES}
        window_counts = {SPLIT_NAMES[s]: binary_counts(targets[splits == s]) for s in (TRAIN, VALIDATION, TEST)}
        print(json.dumps(window_counts, indent=2), flush=True)
        print(f"Candidates: 16 (UID excluded); fitting rows: {len(train_indices):,}", flush=True)

        params = dict(baseline["rf_parameters"])
        if jobs is not None:
            params["n_jobs"] = jobs
        seed = int(baseline["seed"])
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="rf_shap_no_uid_", dir=output_dir) as temporary:
            stage = Path(temporary)
            for name in PRESERVED_FILES:
                shutil.copyfile(baseline_dir / name, stage / name)
                if file_sha256(stage / name) != preserved_hashes[name]:
                    raise ValueError(f"Saved membership copy failed verification: {name}")

            print("2/6 Fitting preprocessing on training rows, with UID excluded...", flush=True)
            state, known_codes = fit_preprocessors(X, source_columns, features, categorical, train_indices)
            for name, codes in known_codes.items():
                save_array(stage / state[name]["known_storage_codes_file"], codes)
            unknown_counts = {SPLIT_NAMES[s]: {name: 0 for name in categorical}
                              for s in (TRAIN, VALIDATION, TEST, UNUSED)}
            prepared = selected_map = None
            try:
                prepared = np.lib.format.open_memmap(
                    stage / "working_16_features.npy", mode="w+", dtype=np.float64, shape=(total, 16))
                for start in range(0, total, 100_000):
                    stop = min(start + 100_000, total)
                    block = transform_block(X[start:stop, source_columns], features, state, known_codes)
                    prepared[start:stop] = block
                    for split in (TRAIN, VALIDATION, TEST, UNUSED):
                        membership = row_split[start:stop] == split
                        for name in categorical:
                            unknown_counts[SPLIT_NAMES[split]][name] += int(
                                np.count_nonzero(block[membership, features.index(name)] == CAT_UNKNOWN))
                    if stop % 500_000 == 0 or stop == total:
                        print(f"  Prepared {stop:,}/{total:,} rows", flush=True)

                print("3/6 Refitting the Random Forest with the saved hyperparameters (CPU)...", flush=True)
                rf_X = np.array(prepared[train_indices], dtype=np.float64, copy=True)
                time_index = features.index("ts")
                time_origin = float(rf_X[:, time_index].min())
                # Translation avoids losing timestamp detail in RF's float32 input.
                # It changes neither the saved float64 matrix nor Pearson correlation.
                rf_X[:, time_index] -= time_origin
                rf_X = rf_X.astype(np.float32)
                if not np.isfinite(rf_X).all():
                    raise ValueError("RF inputs exceed the finite float32 range.")
                rf = RandomForestClassifier(**params)
                rf.fit(rf_X, train_labels)
                joblib.dump(rf, stage / "random_forest.joblib", compress=3)

                print("4/6 Computing Tree SHAP on the exact original training examples...", flush=True)
                positions = np.searchsorted(train_indices, shap_indices)
                if not np.array_equal(train_indices[positions], shap_indices):
                    raise ValueError("SHAP indices are not a subset of the fitting rows.")
                explanation_X = rf_X[positions]
                explainer = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent", model_output="raw")
                values = explainer.shap_values(explanation_X, check_additivity=True)
                positive_index = int(np.flatnonzero(rf.classes_ == 1)[0])
                phi = positive_class_shap(values, len(explanation_X), len(features), positive_index)
                importance = np.abs(phi).mean(axis=0)
                if not np.isfinite(importance).all() or not np.any(importance > 0):
                    raise ValueError("SHAP returned no finite, nonzero feature contributions.")
                ranking = np.lexsort((np.arange(len(features)), -importance))
                selected = ranking[:15]
                selected_names = [features[int(i)] for i in selected]
                selected_source_columns = [source_columns[int(i)] for i in selected]
                if "uid" in selected_names:
                    raise RuntimeError("UID entered the selected model features.")
                save_array(stage / "shap_values_malicious.npy", phi)
                with (stage / "feature_ranking.csv").open("w", encoding="utf-8", newline="") as destination:
                    writer = csv.writer(destination)
                    writer.writerow(["rank", "feature", "step03_column_index", "mean_absolute_shap", "selected"])
                    for rank, index in enumerate(ranking, start=1):
                        writer.writerow([rank, features[int(index)], source_columns[int(index)],
                                         float(importance[index]), rank <= 15])
                del rf_X, explanation_X, values

                print("5/6 Writing the 15 selected features and training extrema...", flush=True)
                selected_map = np.lib.format.open_memmap(
                    stage / "X_selected.npy", mode="w+", dtype=np.float64, shape=(total, 15))
                for start in range(0, total, 100_000):
                    stop = min(start + 100_000, total)
                    selected_map[start:stop] = prepared[start:stop, selected]
                extrema = {}
                for name, index in zip(selected_names, selected):
                    column = np.asarray(prepared[train_indices, int(index)])
                    extrema[name] = {"min": float(column.min()), "max": float(column.max())}
            finally:
                close_map(selected_map, writable=True)
                close_map(prepared, writable=True)
            (stage / "working_16_features.npy").unlink()

            print("6/6 Saving the correction details and compatible Step 4 outputs...", flush=True)
            save_json(stage / "preprocessing.json", {
                "fit_partition": "train", "validation_fold": baseline["validation_fold"],
                "feature_names": features, "parameters": state,
                "excluded_features": ["uid"], "rf_timestamp_origin": time_origin,
                "rf_timestamp_origin_applies_to": "RF and SHAP inputs only",
                "selected_training_extrema": extrema,
            })
            save_json(stage / "selected_features.json", {
                "selection_method": "mean absolute Tree SHAP for malicious output on original balanced fitting-row sample",
                "feature_names": selected_names, "step03_column_indices": selected_source_columns,
                "candidate_feature_names": features, "excluded_features": ["uid"],
                "mean_absolute_shap": [float(importance[i]) for i in selected],
                "X_selected_shape": [total, 15], "normalized": False,
            })
            # Retain the Step 4 schema tag because Step 5 validates this interface.
            # The explicit revision fields identify the changed experiment.
            report = {
                "step": "04_rf_shap_single_iot23_csv",
                "revision": "04b_uid_excluded_with_original_membership",
                "input_dir": str(input_dir), "baseline_step04_dir": str(baseline_dir),
                "source_file": source_summary["source_file"],
                "excluded_features": ["uid"], "candidate_features": features,
                "candidate_feature_count": len(features),
                "seed": seed, "validation_fold": baseline["validation_fold"],
                "rows_in_X_selected": total, "candidate_windows": n_windows,
                "window_rows": WINDOW_ROWS, "stride": WINDOW_ROWS,
                "window_target_rule": baseline["window_target_rule"],
                "mixed_windows": baseline["mixed_windows"],
                "tail_rows_excluded_from_images_and_fitting": total - used,
                "window_class_counts": window_counts,
                "row_class_counts": {SPLIT_NAMES[s]: binary_counts(y[row_split == s])
                                     for s in (TRAIN, VALIDATION, TEST, UNUSED)},
                "split_codes": {"train": TRAIN, "validation": VALIDATION, "test": TEST, "unused_tail": UNUSED},
                "row_split_sha256": baseline["row_split_sha256"],
                "preserved_baseline_files_sha256": preserved_hashes,
                "window_assignments_preserved": True, "shap_explanation_rows_preserved": True,
                "rf_available_fitting_rows": len(train_indices),
                "rf_bootstrap_draws_per_tree": baseline["rf_bootstrap_draws_per_tree"],
                "rf_parameters": rf.get_params(),
                "shap_rows_per_class": baseline["shap_rows_per_class"],
                "selected_features": selected_names,
                "zero_shap_selected_features": [features[int(i)] for i in selected if importance[i] == 0],
                "unknown_category_counts": unknown_counts,
                "all_missing_training_numeric_columns": [name for name in features
                                                           if state[name].get("all_missing_training_fallback")],
                "output_matrix": "X_selected.npy", "output_shape": [total, 15],
                "normalized": False, "detector_trained": False,
                "validation_or_test_rows_used_for_fitting": 0,
                "test_evaluation_performed": False, "zero_day_evaluated": False,
                "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                             "sklearn": sklearn.__version__, "shap": shap.__version__},
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "notes": [
                    "UID was excluded before fitting the new RF selector and before SHAP ranking.",
                    "This is a documented correction of the earlier candidate-feature set.",
                    "The other feature definitions and categorical-code rules are unchanged.",
                    "Original window assignments, manifest, targets and SHAP rows were copied byte-for-byte.",
                    "All 16 candidates use training-fitted preprocessing; the highest-ranked 15 are exported.",
                    "Validation/test features are transformed using training parameters; they do not fit the selector.",
                    "The complete image dataset and labels are not resampled or rebalanced.",
                    "Original prepared data, images and neural-network checkpoints remain unchanged.",
                    "New contour images must be generated from this output before any new detector training.",
                    "This correction alone does not establish improved accuracy or zero-day generalization.",
                ],
            }
            save_json(stage / "step04_summary.json", report)
            # Publish the completion marker last. An incomplete run has no success marker.
            names = sorted(p.name for p in stage.iterdir() if p.name != "step04_summary.json")
            for name in names + ["step04_summary.json"]:
                (stage / name).rename(output_dir / name)

        print("\nSTEP 4B COMPLETE", flush=True)
        print("UID excluded. Selected 15 features from 16 candidates:")
        for rank, name in enumerate(selected_names, start=1):
            print(f"  {rank:2}. {name}")
        print(f"X_selected shape: ({total}, 15)")
        print("Original window assignments and SHAP explanation rows preserved: YES")
        print(f"Candidate image counts: {window_counts}")
        print(f"Summary: {output_dir / 'step04_summary.json'}")
        print("Feature selection complete. New contour images and detector training come next.")
        return report
    finally:
        for array in input_maps:
            close_map(array)


def main():
    parser = argparse.ArgumentParser(description="Step 4b: training-only RF-SHAP with UID excluded and original splits.")
    parser.add_argument("--input-dir", type=Path, default=Path("prepared/dataset7_step03"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("prepared/dataset7_step04"))
    parser.add_argument("--output-dir", type=Path, default=Path("prepared/dataset7_step04_no_uid"))
    parser.add_argument("--jobs", type=int, default=None, help="Optional CPU worker override; default reuses saved RF setting.")
    args = parser.parse_args()
    try:
        run(args.input_dir, args.baseline_dir, args.output_dir, args.jobs)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, AssertionError) as error:
        print(f"\nSTEP 4B FAILED: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped before completion. No success summary was created.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

