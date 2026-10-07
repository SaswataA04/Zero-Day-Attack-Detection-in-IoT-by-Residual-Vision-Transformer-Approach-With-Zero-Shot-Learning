"""Step 4: fix candidate image membership and select 15 features with RF-SHAP.

From your iot_zero_day_research folder:
    python -m pip install scikit-learn shap joblib
    python step04_rf_shap_dataset7.py

INPUT:  prepared/dataset7_step03/ and its Step 3 summary.
OUTPUT: prepared/dataset7_step04/ (must be empty or new).

Method and explicit implementation choices
------------------------------------------
The paper describes RF-SHAP selection of 15 features and an approximately 80:20
random image train/test split. It does not fully define raw-window construction,
stride, mixed-window targets, missing-value rules, or Random Forest settings.
This script fixes those choices before looking at model performance:

1. One candidate image uses 15 consecutive sorted rows, with stride 15. Its
   binary target is the FINAL row's label. This means predicting the final
   connection from its 15-row context. Mixed windows are retained and their
   composition is reported. Labels are never used to construct feature values.
   A final incomplete block is excluded from images and from fitting; its raw
   rows remain in the earlier files and in the selected-feature output.
2. Randomly reserve approximately 20% of candidate images PER CLASS for testing.
   Split the remaining 80% into five stratified folds. Fold 0 is validation for
   this run; the other four folds fit preprocessing, RF and SHAP. Thus the usual
   overall proportions are about 64% fitting / 16% validation / 20% test.
   Fixing image membership BEFORE feature selection keeps held-out rows out of
   fitting. The actual contour pixels will be generated in the next step.
3. No source row belongs to more than one candidate image. Windows never cross
   the train/validation/test assignments. This single-file random split is NOT
   a separate-capture or zero-day evaluation. No maximum time gap is imposed.
4. Refit dense category codes on fitting rows only. Missing=-2, unseen=-1,
   known=0..K-1. Fit numeric medians on the same rows; all-missing training
   columns use a documented zero fallback. No rows are dropped for missingness.
5. Every fitting row is available to Random Forest. Each of the 100 trees uses
   at most 200,000 bootstrap draws, with max_depth=12, min_samples_leaf=2 and
   class_weight='balanced_subsample'. These are implementation settings, not
   claimed author hyperparameters. --bootstrap-rows 0 uses a full-size bootstrap.
6. Explain at most 500 FITTING rows per binary class with exact Tree SHAP.
   Rank by the mean absolute SHAP value for the malicious output. Equal per-class
   explanation counts define a balanced explanation sample, not a population
   estimate. The actual model input/window dataset is never balanced by dropping
   majority-class rows. RF predicts row labels for feature selection; the later
   image model predicts the final-row window targets defined above.

Run only this step now. This script does NOT train CZ-ResViT or report detector
accuracy/F1. It preserves the paper's identifier features as candidates.
X_selected.npy has 15 fitted/imputed features, still in float64 and NOT min-max
normalized. The next stage must use this saved split and feature order.

For subsequent cross-validation runs, preprocessing and feature selection must
be refitted, for example:
    python step04_rf_shap_dataset7.py --fold 1 --output-dir prepared/dataset7_step04_fold1
This run alone is not five-fold cross-validation. Keep the seed unchanged when
comparing folds or architectures so the outer test set stays fixed.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

try:
    import joblib
    import numpy as np
    import shap
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
except ImportError as error:
    raise SystemExit("Install dependencies: python -m pip install numpy scikit-learn shap joblib") from error


WINDOW_ROWS = 15
N_FOLDS = 5
TRAIN, VALIDATION, TEST, UNUSED = 0, 1, 2, -1
SPLIT_NAMES = {TRAIN: "train", VALIDATION: "validation", TEST: "test", UNUSED: "unused_tail"}
CLASS_NAMES = {0: "benign", 1: "malicious"}
CAT_MISSING, CAT_UNKNOWN = -2, -1


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


def define_samples(y, seed=42, validation_fold=0):
    """Fix nonoverlapping windows, the outer test set, and training folds."""
    total = len(y)
    n_windows = total // WINDOW_ROWS
    used = n_windows * WINDOW_ROWS
    if not n_windows:
        raise ValueError("Fewer than 15 rows; no complete candidate image can be formed.")
    window_rows = np.asarray(y[:used]).reshape(n_windows, WINDOW_ROWS)
    targets = window_rows[:, -1].copy()
    malicious_counts = window_rows.sum(axis=1, dtype=np.int16)
    target_counts = np.where(targets == 1, malicious_counts, WINDOW_ROWS - malicious_counts)
    purity = target_counts / float(WINDOW_ROWS)
    class_counts = binary_counts(targets)
    print(f"Candidate image targets: {class_counts}", flush=True)

    # Six images per class are the minimum for one outer-test example plus five
    # training folds. Never secretly switch to a different labeling/split rule.
    if min(class_counts.values()) < 6:
        raise ValueError(
            f"Candidate image class counts are {class_counts}. At least six per class "
            "are required for this split. Window targets need review before training."
        )
    rng = np.random.default_rng(seed)
    folds = np.full(n_windows, -1, dtype=np.int8)  # -1 denotes the outer test set.
    window_split = np.full(n_windows, TEST, dtype=np.int8)
    for target in (0, 1):
        members = rng.permutation(np.flatnonzero(targets == target))
        n_test = min(max(1, math.ceil(0.20 * len(members))), len(members) - N_FOLDS)
        training_pool = members[n_test:]
        folds[training_pool] = np.arange(len(training_pool)) % N_FOLDS
        window_split[training_pool] = TRAIN
    window_split[folds == validation_fold] = VALIDATION

    row_split = np.full(total, UNUSED, dtype=np.int8)
    row_split[:used] = np.repeat(window_split, WINDOW_ROWS)
    if not np.array_equal(row_split[:used].reshape(-1, WINDOW_ROWS),
                          np.broadcast_to(window_split[:, None], (n_windows, WINDOW_ROWS))):
        raise RuntimeError("A window spans multiple partitions.")
    return {
        "targets": targets, "malicious_counts": malicious_counts, "purity": purity,
        "folds": folds, "window_split": window_split, "row_split": row_split,
        "n_windows": n_windows, "used_rows": used, "tail_rows": total - used,
    }


def write_window_manifest(path, samples, X, features, source_rows):
    """Indices address Step 3 arrays; source_rows[start:stop] gives full provenance."""
    time_column = features.index("ts")
    columns = ["window_id", "start_index", "stop_index_exclusive", "target", "split",
               "outer_training_fold", "benign_rows", "malicious_rows", "target_purity",
               "start_ts", "end_ts", "duration_seconds", "first_source_row", "last_source_row"]
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(columns)
        for i in range(samples["n_windows"]):
            start, stop = i * WINDOW_ROWS, (i + 1) * WINDOW_ROWS
            first, last = float(X[start, time_column]), float(X[stop - 1, time_column])
            if not np.isfinite(first) or not np.isfinite(last) or last < first:
                raise ValueError("Invalid or out-of-order window timestamps.")
            malicious = int(samples["malicious_counts"][i])
            writer.writerow([i, start, stop, int(samples["targets"][i]),
                             SPLIT_NAMES[int(samples["window_split"][i])], int(samples["folds"][i]),
                             WINDOW_ROWS - malicious, malicious, float(samples["purity"][i]),
                             first, last, last - first, int(source_rows[start]), int(source_rows[stop - 1])])


def fit_preprocessors(X, features, categorical, train_indices):
    """Learn only from fitting rows; storage IDs are NOT used directly by RF."""
    state = {}
    known_codes = {}
    for index, name in enumerate(features):
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


def run(input_dir, output_dir, seed=42, fold=0, trees=100, depth=12,
        bootstrap_rows=200_000, shap_per_class=500, jobs=4):
    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    summary = json.loads((input_dir / "step03_summary.json").read_text(encoding="utf-8"))
    if summary.get("step") != "03_numeric_storage_single_iot23_csv":
        raise ValueError("Use the Step 3 output directory.")
    features = summary["feature_names"]
    categorical = summary["categorical_features"]
    if (len(features) != 17 or len(set(features)) != 17 or "ts" not in features
            or not set(categorical).issubset(features)
            or set(features) & {"label", "target", "source_row", "source_file", "detailed-label", "label_original"}):
        raise ValueError("Invalid 17-feature schema.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty. Choose a new --output-dir to rerun.")

    input_maps = []
    started = time.perf_counter()
    try:
        X = np.load(input_dir / "X_storage.npy", mmap_mode="r", allow_pickle=False)
        input_maps.append(X)
        y = np.load(input_dir / "y.npy", mmap_mode="r", allow_pickle=False)
        input_maps.append(y)
        source_rows = np.load(input_dir / "source_rows.npy", mmap_mode="r", allow_pickle=False)
        input_maps.append(source_rows)
        total = int(summary["rows_written"])
        if (X.shape != (total, 17) or X.dtype != np.float64 or y.shape != (total,)
                or source_rows.shape != (total,) or not np.isin(y, [0, 1]).all()
                or binary_counts(y) != {name: summary["label_counts"].get(name, 0) for name in CLASS_NAMES.values()}):
            raise ValueError("Step 3 array shapes, values or class counts disagree with its summary.")

        print("1/6 Defining candidate images and fixing their partitions...", flush=True)
        samples = define_samples(y, seed, fold)
        row_split = samples["row_split"]
        train_indices = np.flatnonzero(row_split == TRAIN)
        train_labels = np.asarray(y[train_indices])
        if len(np.unique(train_labels)) != 2:
            raise ValueError("Both binary classes are required in fitting rows.")
        window_counts = {
            SPLIT_NAMES[s]: binary_counts(samples["targets"][samples["window_split"] == s])
            for s in (TRAIN, VALIDATION, TEST)
        }
        print(json.dumps(window_counts, indent=2), flush=True)
        print(f"Complete candidate images: {samples['n_windows']:,}; unused tail rows: {samples['tail_rows']}", flush=True)

        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="rf_shap_work_", dir=output_dir) as temporary:
            stage = Path(temporary)
            write_window_manifest(stage / "window_manifest.csv", samples, X, features, source_rows)
            save_array(stage / "row_split.npy", row_split)
            save_array(stage / "window_targets.npy", samples["targets"])
            save_array(stage / "window_split.npy", samples["window_split"])
            save_array(stage / "window_folds.npy", samples["folds"])

            print("2/6 Fitting category codes and numeric medians on training rows...", flush=True)
            state, known_codes = fit_preprocessors(X, features, categorical, train_indices)
            for name, codes in known_codes.items():
                save_array(stage / state[name]["known_storage_codes_file"], codes)

            prepared = selected_map = None
            unknown_counts = {SPLIT_NAMES[s]: {name: 0 for name in categorical} for s in (TRAIN, VALIDATION, TEST, UNUSED)}
            try:
                prepared = np.lib.format.open_memmap(stage / "working_17_features.npy", mode="w+", dtype=np.float64, shape=X.shape)
                for start in range(0, total, 100_000):
                    stop = min(start + 100_000, total)
                    block = transform_block(X[start:stop], features, state, known_codes)
                    prepared[start:stop] = block
                    for split in (TRAIN, VALIDATION, TEST, UNUSED):
                        mask = row_split[start:stop] == split
                        for name in categorical:
                            unknown_counts[SPLIT_NAMES[split]][name] += int(np.count_nonzero(block[mask, features.index(name)] == CAT_UNKNOWN))
                    if stop % 500_000 == 0 or stop == total:
                        print(f"  Prepared {stop:,} rows with training-fitted parameters...", flush=True)

                print("3/6 Fitting the Random Forest feature selector (CPU)...", flush=True)
                # scikit-learn trees internally use float32. Subtracting a
                # training-derived time origin first preserves substantially more
                # timestamp detail than casting raw Unix timestamps to float32.
                # This translation affects only the RF/SHAP view; stored features
                # stay float64 and are not yet min-max normalized.
                rf_X = np.asarray(prepared[train_indices], dtype=np.float64)
                time_index = features.index("ts")
                time_origin = float(rf_X[:, time_index].min())
                rf_X[:, time_index] -= time_origin
                rf_X = rf_X.astype(np.float32)
                if not np.isfinite(rf_X).all():
                    raise ValueError("Features exceed the finite float32 range accepted by Random Forest.")
                max_samples = None if bootstrap_rows == 0 else min(bootstrap_rows, len(train_indices))
                rf = RandomForestClassifier(
                    n_estimators=trees, max_depth=depth, min_samples_leaf=2,
                    max_features="sqrt", bootstrap=True, max_samples=max_samples,
                    class_weight="balanced_subsample", random_state=seed, n_jobs=jobs, verbose=1,
                )
                print(f"  Available fitting rows: {len(train_indices):,}; per-tree bootstrap draws: {max_samples or len(train_indices):,}", flush=True)
                rf.fit(rf_X, train_labels)
                joblib.dump(rf, stage / "random_forest.joblib", compress=3)

                print("4/6 Computing Tree SHAP on training examples...", flush=True)
                rng = np.random.default_rng(seed + 1000)
                by_class = [np.flatnonzero(train_labels == label) for label in (0, 1)]
                n_each = min(shap_per_class, *(len(indices) for indices in by_class))
                if n_each < 1:
                    raise ValueError("Cannot select a SHAP explanation sample containing both classes.")
                explanation_positions = np.concatenate([rng.choice(indices, n_each, replace=False) for indices in by_class])
                explanation_indices = train_indices[explanation_positions]
                if not np.all(row_split[explanation_indices] == TRAIN):
                    raise RuntimeError("A held-out row entered the SHAP sample.")
                explanation_X = rf_X[explanation_positions]
                print(f"  Explaining {n_each:,} benign + {n_each:,} malicious fitting rows...", flush=True)
                explainer = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent", model_output="raw")
                values = explainer.shap_values(explanation_X, check_additivity=True)
                positive_index = int(np.flatnonzero(rf.classes_ == 1)[0])
                phi = positive_class_shap(values, len(explanation_X), len(features), positive_index)
                importance = np.abs(phi).mean(axis=0)
                if not np.isfinite(importance).all() or not np.any(importance > 0):
                    raise ValueError("SHAP returned no finite, nonzero feature contributions; inspect the fitting data.")
                # Stable feature-order tie break; never substitute Gini importance
                # for SHAP or claim that all 15 retained features must be useful.
                ranking = np.lexsort((np.arange(len(features)), -importance))
                selected = ranking[:15]
                selected_names = [features[int(i)] for i in selected]
                save_array(stage / "shap_values_malicious.npy", phi)
                save_array(stage / "shap_row_indices.npy", explanation_indices)
                with (stage / "feature_ranking.csv").open("w", encoding="utf-8", newline="") as destination:
                    writer = csv.writer(destination)
                    writer.writerow(["rank", "feature", "step03_column_index", "mean_absolute_shap", "selected"])
                    for rank, index in enumerate(ranking, start=1):
                        writer.writerow([rank, features[int(index)], int(index), float(importance[index]), rank <= 15])
                del rf_X, explanation_X, values

                print("5/6 Writing the 15 selected features...", flush=True)
                selected_map = np.lib.format.open_memmap(stage / "X_selected.npy", mode="w+", dtype=np.float64, shape=(total, 15))
                for start in range(0, total, 100_000):
                    stop = min(start + 100_000, total)
                    selected_map[start:stop] = prepared[start:stop, selected]
                # Store the exact training extrema for the next min-max stage.
                extrema = {}
                for name, index in zip(selected_names, selected):
                    values = np.asarray(prepared[train_indices, int(index)])
                    extrema[name] = {"min": float(values.min()), "max": float(values.max())}
            finally:
                close_map(selected_map, writable=True)
                close_map(prepared, writable=True)
            (stage / "working_17_features.npy").unlink()

            print("6/6 Saving the split, fitted preprocessing and feature ranking...", flush=True)
            save_json(stage / "preprocessing.json", {
                "fit_partition": "train", "validation_fold": fold, "feature_names": features,
                "parameters": state, "rf_timestamp_origin": time_origin,
                "rf_timestamp_origin_applies_to": "RF and SHAP inputs only",
                "selected_training_extrema": extrema,
            })
            save_json(stage / "selected_features.json", {
                "selection_method": "mean absolute Tree SHAP for malicious output on balanced fitting-row sample",
                "feature_names": selected_names, "step03_column_indices": selected.tolist(),
                "mean_absolute_shap": [float(importance[i]) for i in selected],
                "X_selected_shape": [total, 15], "normalized": False,
            })
            report = {
                "step": "04_rf_shap_single_iot23_csv", "input_dir": str(input_dir),
                "source_file": summary["source_file"], "seed": seed, "validation_fold": fold,
                "rows_in_X_selected": total, "candidate_windows": samples["n_windows"],
                "window_rows": WINDOW_ROWS, "stride": WINDOW_ROWS,
                "window_target_rule": "final row's existing binary target",
                "mixed_windows": int(np.count_nonzero(samples["purity"] < 1)),
                "tail_rows_excluded_from_images_and_fitting": samples["tail_rows"],
                "window_class_counts": window_counts,
                "row_class_counts": {SPLIT_NAMES[s]: binary_counts(y[row_split == s]) for s in (TRAIN, VALIDATION, TEST, UNUSED)},
                "split_codes": {"train": TRAIN, "validation": VALIDATION, "test": TEST, "unused_tail": UNUSED},
                "row_split_sha256": hashlib.sha256(row_split.tobytes()).hexdigest(),
                "rf_available_fitting_rows": len(train_indices),
                "rf_bootstrap_draws_per_tree": max_samples or len(train_indices),
                "rf_parameters": rf.get_params(),
                "shap_rows_per_class": n_each, "selected_features": selected_names,
                "zero_shap_selected_features": [features[int(i)] for i in selected if importance[i] == 0],
                "unknown_category_counts": unknown_counts,
                "all_missing_training_numeric_columns": [name for name in features if state[name].get("all_missing_training_fallback")],
                "output_matrix": "X_selected.npy", "output_shape": [total, 15],
                "normalized": False, "detector_trained": False,
                "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "sklearn": sklearn.__version__, "shap": shap.__version__},
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "notes": [
                    "Sample construction, stride, endpoint target rule, seed and RF settings are explicit reconstruction choices.",
                    "Approximately 20% of windows per class form a fixed outer test set; one of five remaining folds is validation.",
                    "This run fits preprocessing and selection on four training folds only; five-fold CV requires refitting the other folds.",
                    "Adjacent nonoverlapping windows can remain correlated; this is not independent-capture evaluation.",
                    "Raw training-row labels fit RF; final-row labels are the targets for the later image model.",
                    "All fitting rows are available to RF; per-tree bootstrap draws are capped by configuration.",
                    "Balanced SHAP explanation sampling changes the explanation population, not the retained window dataset.",
                    "No unseen-attack or zero-day performance has been measured; no detector metrics are reported here.",
                    "Reuse these window assignments when constructing contours; do not randomly split their rows again.",
                    "The original sorted CSV and Step 3 source_rows.npy resolve every window's full provenance.",
                ],
            }
            save_json(stage / "step04_summary.json", report)
            # Publish completion summary last; unfinished work is removed on error.
            names = sorted(p.name for p in stage.iterdir() if p.name != "step04_summary.json")
            for name in names + ["step04_summary.json"]:
                (stage / name).rename(output_dir / name)

        print("\nSTEP 4 COMPLETE", flush=True)
        print(f"Selected 15 features from {len(features)} candidates:")
        for rank, name in enumerate(selected_names, start=1):
            print(f"  {rank:2}. {name}")
        print(f"X_selected shape: ({total}, 15)")
        print(f"Candidate image counts: {window_counts}")
        print(f"Summary: {output_dir / 'step04_summary.json'}")
        print("Feature selection is complete. CZ-ResViT has not been trained yet.")
        return report
    finally:
        for array in input_maps:
            close_map(array)


def main():
    parser = argparse.ArgumentParser(description="Step 4: candidate image split and training-only RF-SHAP selection.")
    parser.add_argument("--input-dir", type=Path, default=Path("prepared/dataset7_step03"))
    parser.add_argument("--output-dir", type=Path, default=Path("prepared/dataset7_step04"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0, choices=range(N_FOLDS))
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--bootstrap-rows", type=int, default=200_000, help="Draws per tree; 0 means full-size bootstrap")
    parser.add_argument("--shap-per-class", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if min(args.trees, args.depth, args.shap_per_class, args.jobs) < 1 or args.bootstrap_rows < 0 or args.seed < 0:
        parser.error("Use positive trees/depth/shap-per-class/jobs and nonnegative bootstrap-rows/seed.")
    try:
        run(args.input_dir, args.output_dir, args.seed, args.fold, args.trees,
            args.depth, args.bootstrap_rows, args.shap_per_class, args.jobs)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, AssertionError) as error:
        print(f"\nSTEP 4 FAILED: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped before completion. No success summary was created.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
