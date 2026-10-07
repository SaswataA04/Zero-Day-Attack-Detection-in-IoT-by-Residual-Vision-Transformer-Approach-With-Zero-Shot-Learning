"""Step 5: training-only normalization, Pearson matrices, and contour images.

Run from iot_zero_day_research:
    python -m pip install "matplotlib>=3.8" pillow
    python step05_contours_dataset7.py

Inputs:  prepared/dataset7_step04/
Outputs: prepared/dataset7_step05/
The same command resumes completed phases and ZIP batches after interruption.
Changing input data or image/filter settings requires a new output directory.
--workers 1 runs without multiprocessing; the default is four workers.

EXPLICIT RECONSTRUCTION CHOICES (not claimed author implementation details):
* Reuse Step 4's 15-row windows, target of the final row, feature order, and
  partitions. Never resample, relabel or drop complete windows in this step.
* Fit min/max on fitting rows only and map their values to [-1,1]. Held-out
  values may exceed that range: they are NOT clipped, which avoids destroying
  local variation before Pearson correlation. Features constant throughout the
  fitting data map to zero everywhere. Numeric work stays in float64.
* Compute a 15x15 feature correlation matrix WITHIN each fixed 15-row window.
  A normalized feature with within-window standard deviation <=1e-12 is treated
  as constant. Its undefined correlations, including its diagonal, use zero.
  Save a separate activity mask; zero is therefore not always an observed PCC.
* The PDF does not specify its low-variance/redundancy filter. Here we learn a
  fixed mask using only fitting-window correlation entries. Consider the 105
  unique off-diagonal positions; mask entries whose across-window variance is
  <=1e-8. In descending variance order, also mask entries whose values correlate
  with an already retained entry at absolute Pearson >=0.999. Apply this same
  symmetric mask everywhere. Diagonals retain their activity values. This is
  correlation-entry filtering, not another selection of raw traffic features.
  Raw matrices are retained for audit; filtered matrices are masked image
  representations and need not be positive semidefinite correlation matrices.
* Render real Matplotlib filled contours: 224x224 RGB, 21 fixed filled bands,
  coolwarm palette, feature 0 at the top/left, no axes/text/class labels inside
  model images, and no per-image color autoscaling. Palette and levels are our
  choices. Store lossless uint8 PNGs; load_model_input() divides by 255 and
  returns float32 [3,224,224] values in [0,1] for the later network.
* Store 1,000 PNGs per ZIP (not a huge uncompressed image tensor or 238,735 loose
  files). Identical matrices may reuse cached PNG bytes; every window keeps its
  own indexed image. Six preview files are copied from these exact ZIP entries.

No detector is trained here. All complete-window counts from Step 4 must remain
unchanged. The three incomplete tail rows in dataset7 remain excluded as before.
The data/filter/render choices and library versions are saved with the output.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import multiprocessing
import sqlite3
import sys
import time
import zipfile
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import PIL
    from PIL import Image
except ImportError as error:
    raise SystemExit('Install dependencies: python -m pip install numpy "matplotlib>=3.8" pillow') from error


VERSION = 1
FEATURES = WINDOW_ROWS = 15
HEIGHT = WIDTH = 224
LEVELS = np.linspace(-1.0, 1.0, 22)  # 21 filled bands, with a neutral central band.
SPLIT_NAMES = {0: "train", 1: "validation", 2: "test"}
CLASS_NAMES = {0: "benign", 1: "malicious"}
BATCH_WINDOWS = 2048
_CORRELATIONS = _FIGURE = _CANVAS = _AXES = None
_PNG_CACHE = OrderedDict()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path, data):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_array(path, values):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as destination:
        np.save(destination, values, allow_pickle=False)
    temporary.replace(path)


def close_map(array, writable=False):
    if isinstance(array, np.memmap):
        if writable:
            array.flush()
        array._mmap.close()


def counts_by_split(labels, splits):
    return {name: {CLASS_NAMES[c]: int(np.count_nonzero((splits == s) & (labels == c)))
                   for c in (0, 1)} for s, name in SPLIT_NAMES.items()}


def normalize(values, minima, maxima):
    """One training-fitted affine map; leave held-out extrapolation unclipped."""
    span = maxima - minima
    variable = span > 0
    result = np.zeros_like(values, dtype=np.float64)
    result[..., variable] = 2.0 * ((values[..., variable] - minima[variable]) / span[variable]) - 1.0
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite values after normalization.")
    return result


def pearson_windows(values, tolerance=1e-12):
    """Batched Pearson coefficients; values have [windows, rows, features]."""
    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    activity = norms / np.sqrt(values.shape[1]) > tolerance
    inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=activity)
    unit = centered * inverse[:, None, :]
    correlations = np.matmul(unit.transpose(0, 2, 1), unit)
    correlations = np.clip(correlations, -1.0, 1.0)
    diagonal = np.arange(values.shape[2])
    correlations[:, diagonal, diagonal] = activity.astype(np.float64)
    if not np.isfinite(correlations).all():
        raise ValueError("Nonfinite Pearson coefficients.")
    return correlations.astype(np.float32), activity


def learn_correlation_mask(raw, train_mask, feature_names, variance_threshold, redundancy_threshold):
    """Learn low-variance and redundant ENTRY positions using fitting windows only."""
    upper = np.triu_indices(FEATURES, k=1)
    p = len(upper[0])
    n = 0
    mean = np.zeros(p, dtype=np.float64)
    m2 = np.zeros((p, p), dtype=np.float64)
    # Batch Welford covariance avoids subtracting two nearly equal large sums.
    for start in range(0, len(raw), BATCH_WINDOWS):
        stop = min(start + BATCH_WINDOWS, len(raw))
        batch = np.asarray(raw[start:stop])[train_mask[start:stop]][:, upper[0], upper[1]].astype(np.float64)
        if not len(batch):
            continue
        batch_mean = batch.mean(axis=0)
        centered = batch - batch_mean
        delta = batch_mean - mean
        new_n = n + len(batch)
        m2 += centered.T @ centered + np.outer(delta, delta) * (n * len(batch) / new_n)
        mean += delta * (len(batch) / new_n)
        n = new_n
    if n < 2:
        raise ValueError("At least two fitting windows are required to learn correlation filtering.")
    covariance = m2 / n
    variances = np.maximum(np.diag(covariance), 0.0)
    priority = np.lexsort((np.arange(p), -variances))
    kept = []
    reasons = [None] * p
    duplicate_of = [None] * p
    for index in priority:
        index = int(index)
        if variances[index] <= variance_threshold:
            reasons[index] = "low_variance"
            continue
        for earlier in kept:
            denominator = np.sqrt(variances[index] * variances[earlier])
            association = float(np.clip(covariance[index, earlier] / denominator, -1, 1))
            if abs(association) >= redundancy_threshold:
                reasons[index] = "redundant"
                duplicate_of[index] = earlier
                break
        if reasons[index] is None:
            reasons[index] = "retained"
            kept.append(index)
    mask = np.eye(FEATURES, dtype=bool)
    for index in kept:
        a, b = int(upper[0][index]), int(upper[1][index])
        mask[a, b] = mask[b, a] = True
    entries = []
    for i, (a, b) in enumerate(zip(*upper)):
        entries.append({"features": [feature_names[int(a)], feature_names[int(b)]],
                        "variance_on_training_windows": float(variances[i]), "decision": reasons[i],
                        "redundant_with_pair_index": duplicate_of[i]})
    return mask, {"fit_windows": n, "candidate_off_diagonal_pairs": p,
                  "retained_pairs": len(kept), "low_variance_pairs": reasons.count("low_variance"),
                  "redundant_pairs": reasons.count("redundant"), "entries": entries,
                  "variance_threshold": variance_threshold, "redundancy_threshold": redundancy_threshold}


def create_correlations(source_dir, out, X, row_split, splits, features, variance_threshold, redundancy_threshold):
    """Write finite raw/filtered matrices without changing the window population."""
    marker = out / "correlations_complete.json"
    if marker.exists():
        print("Reusing completed normalization and correlation matrices.", flush=True)
        return read_json(marker)
    print("1/4 Verifying training min/max and computing Pearson matrices...", flush=True)
    n_windows = len(splits)
    minima = np.full(FEATURES, np.inf)
    maxima = np.full(FEATURES, -np.inf)
    for start in range(0, len(X), 100_000):
        stop = min(start + 100_000, len(X))
        batch = np.asarray(X[start:stop])
        if not np.isfinite(batch).all():
            raise ValueError("Step 4 selected features contain nonfinite values.")
        fitting = batch[row_split[start:stop] == 0]
        if len(fitting):
            minima = np.minimum(minima, fitting.min(axis=0))
            maxima = np.maximum(maxima, fitting.max(axis=0))
    expected = read_json(source_dir / "preprocessing.json")["selected_training_extrema"]
    if not all(minima[i] == expected[name]["min"] and maxima[i] == expected[name]["max"] for i, name in enumerate(features)):
        raise ValueError("Selected feature extrema disagree with Step 4; check the input files.")
    atomic_json(out / "normalization.json", {
        "fit_partition": "train", "features": features, "training_min": minima.tolist(),
        "training_max": maxima.tolist(), "training_output_range": [-1, 1],
        "clip_held_out_values": False, "global_constant_value": 0,
        "global_constant_features": [features[i] for i in np.flatnonzero(maxima == minima)],
        "within_window_std_tolerance": 1e-12,
    })
    raw = activity = filtered = None
    raw_partial = out / "correlations_raw.npy.partial"
    activity_partial = out / "feature_activity.npy.partial"
    filtered_partial = out / "correlations_filtered.npy.partial"
    out_of_range = {name: 0 for name in SPLIT_NAMES.values()}
    try:
        raw = np.lib.format.open_memmap(raw_partial, mode="w+", dtype=np.float32, shape=(n_windows, FEATURES, FEATURES))
        activity = np.lib.format.open_memmap(activity_partial, mode="w+", dtype=np.bool_, shape=(n_windows, FEATURES))
        for start in range(0, n_windows, BATCH_WINDOWS):
            stop = min(start + BATCH_WINDOWS, n_windows)
            values = np.asarray(X[start * WINDOW_ROWS:stop * WINDOW_ROWS]).reshape(-1, WINDOW_ROWS, FEATURES)
            values = normalize(values, minima, maxima)
            outside = np.count_nonzero((values < -1 - 1e-10) | (values > 1 + 1e-10), axis=(1, 2))
            for s, name in SPLIT_NAMES.items():
                out_of_range[name] += int(outside[splits[start:stop] == s].sum())
            matrices, varying = pearson_windows(values)
            raw[start:stop], activity[start:stop] = matrices, varying
            if stop % (10 * BATCH_WINDOWS) == 0 or stop == n_windows:
                print(f"  Pearson matrices: {stop:,}/{n_windows:,}", flush=True)

        print("2/4 Learning a fixed correlation-entry filter on training windows...", flush=True)
        mask, filter_report = learn_correlation_mask(raw, splits == 0, features, variance_threshold, redundancy_threshold)
        atomic_array(out / "correlation_mask.npy", mask)
        atomic_json(out / "filtering.json", filter_report)
        filtered = np.lib.format.open_memmap(filtered_partial, mode="w+", dtype=np.float32, shape=raw.shape)
        for start in range(0, n_windows, BATCH_WINDOWS):
            stop = min(start + BATCH_WINDOWS, n_windows)
            filtered[start:stop] = np.where(mask, raw[start:stop], np.float32(0))
        low_activity = {
            name: int(np.count_nonzero(np.asarray(activity)[splits == s].sum(axis=1) < 2))
            for s, name in SPLIT_NAMES.items()
        }
    finally:
        for array in (filtered, activity, raw):
            close_map(array, writable=True)
    raw_partial.replace(out / "correlations_raw.npy")
    activity_partial.replace(out / "feature_activity.npy")
    filtered_partial.replace(out / "correlations_filtered.npy")
    report = {"windows": n_windows, "filtered_pairs_retained": filter_report["retained_pairs"],
              "windows_with_fewer_than_two_varying_features": low_activity,
              "unclipped_values_outside_training_range": out_of_range,
              "complete_windows_dropped": 0}
    atomic_json(marker, report)
    return report


def init_renderer(correlation_path):
    """Called once inside each worker. Keep figures and a small PNG cache local."""
    global _CORRELATIONS, _FIGURE, _CANVAS, _AXES, _PNG_CACHE
    _CORRELATIONS = np.load(correlation_path, mmap_mode="r", allow_pickle=False)
    _FIGURE = Figure(figsize=(2.24, 2.24), dpi=100)
    _CANVAS = FigureCanvasAgg(_FIGURE)
    _AXES = _FIGURE.add_axes([0, 0, 1, 1])
    _AXES.set_axis_off()
    _AXES.set_xlim(0, FEATURES - 1)
    _AXES.set_ylim(FEATURES - 1, 0)
    _PNG_CACHE = OrderedDict()


def png_for_matrix(matrix):
    key = matrix.tobytes()
    cached = _PNG_CACHE.get(key)
    if cached is not None:
        _PNG_CACHE.move_to_end(key)
        return cached
    contours = _AXES.contourf(matrix, levels=LEVELS, cmap="coolwarm", vmin=-1, vmax=1, antialiased=False)
    try:
        _CANVAS.draw()
        pixels = np.asarray(_CANVAS.buffer_rgba())[:, :, :3].copy()
        if pixels.shape != (HEIGHT, WIDTH, 3):
            raise ValueError(f"Unexpected rendered shape: {pixels.shape}")
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG", compress_level=3)
        encoded = buffer.getvalue()
    finally:
        contours.remove()
    _PNG_CACHE[key] = encoded
    if len(_PNG_CACHE) > 512:
        _PNG_CACHE.popitem(last=False)
    return encoded


def render_shard(out_directory, shard, start, stop, configuration_hash):
    """Publish a ZIP only after every PNG has been written and the ZIP is closed."""
    directory = Path(out_directory)
    archive = directory / f"shard_{shard:05d}.zip"
    temporary = archive.with_name(archive.name + ".partial")
    hashes = []
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as destination:
        for window in range(start, stop):
            encoded = png_for_matrix(np.asarray(_CORRELATIONS[window]))
            hashes.append(hashlib.sha256(encoded).hexdigest())
            # Fixed ZIP timestamps make archives deterministic too. Image names
            # contain window IDs only; labels never get rendered into the pixels.
            entry = zipfile.ZipInfo(f"window_{window:09d}.png", date_time=(2020, 1, 1, 0, 0, 0))
            destination.writestr(entry, encoded)
    temporary.replace(archive)
    metadata = {"configuration_hash": configuration_hash, "shard": shard, "start": start, "stop": stop,
                "archive_bytes": archive.stat().st_size, "png_sha256": hashes}
    atomic_json(directory / f"shard_{shard:05d}.json", metadata)
    return stop - start


def completed_shard(directory, shard, start, stop, configuration_hash):
    archive = directory / f"shard_{shard:05d}.zip"
    marker = directory / f"shard_{shard:05d}.json"
    if not archive.exists() or not marker.exists():
        return False
    try:
        data = read_json(marker)
        if (data["configuration_hash"] != configuration_hash or data["start"] != start
                or data["stop"] != stop or len(data["png_sha256"]) != stop - start
                or data["archive_bytes"] != archive.stat().st_size):
            return False
        with zipfile.ZipFile(archive) as source:
            names = source.namelist()
        return names == [f"window_{i:09d}.png" for i in range(start, stop)]
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False


def render_all(out, n_windows, shard_size, workers, configuration_hash):
    directory = out / "images"
    directory.mkdir(exist_ok=True)
    pending_jobs = []
    completed = 0
    for shard, start in enumerate(range(0, n_windows, shard_size)):
        stop = min(start + shard_size, n_windows)
        if completed_shard(directory, shard, start, stop, configuration_hash):
            completed += stop - start
        else:
            pending_jobs.append((str(directory), shard, start, stop, configuration_hash))
    print(f"3/4 Generating contour PNGs: {completed:,}/{n_windows:,} already complete.", flush=True)
    if not pending_jobs:
        return
    correlation_path = str(out / "correlations_filtered.npy")
    if workers == 1:
        init_renderer(correlation_path)
        try:
            for arguments in pending_jobs:
                completed += render_shard(*arguments)
                print(f"  Images: {completed:,}/{n_windows:,}", flush=True)
        finally:
            close_map(_CORRELATIONS)
    else:
        # Bound queued work. Each worker reads its own matrix slice from disk.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=init_renderer, initargs=(correlation_path,)) as pool:
            jobs = iter(pending_jobs)
            futures = set()
            for _ in range(min(2 * workers, len(pending_jobs))):
                futures.add(pool.submit(render_shard, *next(jobs)))
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    completed += future.result()
                    print(f"  Images: {completed:,}/{n_windows:,}", flush=True)
                    arguments = next(jobs, None)
                    if arguments is not None:
                        futures.add(pool.submit(render_shard, *arguments))


def load_model_input(output_dir, window_id, shard_size=1000):
    """Read exactly the stored PNG and normalize pixels for the later CNN."""
    root = Path(output_dir)
    with zipfile.ZipFile(root / "images" / f"shard_{window_id // shard_size:05d}.zip") as archive:
        png = archive.read(f"window_{window_id:09d}.png")
    with Image.open(io.BytesIO(png)) as picture:
        values = np.asarray(picture.convert("RGB"), dtype=np.float32) / np.float32(255.0)
    return np.ascontiguousarray(values.transpose(2, 0, 1))


def index_and_previews(out, labels, splits, shard_size):
    """Index every image and copy previews from the same bytes used for training."""
    print("4/4 Writing the image index and six previews...", flush=True)
    previews = out / "previews"
    previews.mkdir(exist_ok=True)
    preview_ids = {}
    for split, split_name in SPLIT_NAMES.items():
        for target, class_name in CLASS_NAMES.items():
            matches = np.flatnonzero((splits == split) & (labels == target))
            if len(matches):
                preview_ids[int(matches[0])] = f"{split_name}_{class_name}.png"
    activity = np.load(out / "feature_activity.npy", mmap_mode="r")
    hash_counts = {}  # Per hash: train, validation and test counts, plus class bits.
    temporary = out / "image_index.csv.partial"
    try:
        with temporary.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(["window_id", "split", "target", "archive", "member", "varying_features", "png_sha256"])
            for shard, start in enumerate(range(0, len(labels), shard_size)):
                stop = min(start + shard_size, len(labels))
                metadata = read_json(out / "images" / f"shard_{shard:05d}.json")
                relative = f"images/shard_{shard:05d}.zip"
                for window, digest in zip(range(start, stop), metadata["png_sha256"]):
                    split, target = int(splits[window]), int(labels[window])
                    member = f"window_{window:09d}.png"
                    writer.writerow([window, SPLIT_NAMES[split], target, relative, member, int(activity[window].sum()), digest])
                    numbers = hash_counts.setdefault(digest, [0, 0, 0, 0])
                    numbers[split] += 1
                    numbers[3] |= 1 << target
                    if window in preview_ids:
                        with zipfile.ZipFile(out / relative) as archive:
                            encoded = archive.read(member)
                        (previews / preview_ids[window]).write_bytes(encoded)
                        tensor = load_model_input(out, window, shard_size)
                        if tensor.shape != (3, HEIGHT, WIDTH) or not np.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
                            raise ValueError("Image loading/pixel normalization verification failed.")
    finally:
        close_map(activity)
    temporary.replace(out / "image_index.csv")
    return {"preview_files": list(preview_ids.values()), "unique_png_hashes": len(hash_counts),
            "duplicate_image_instances": len(labels) - len(hash_counts),
            "unique_hashes_shared_by_train_and_test": sum(1 for n in hash_counts.values() if n[0] and n[2]),
            "test_images_matching_a_training_image": sum(n[2] for n in hash_counts.values() if n[0]),
            "unique_hashes_with_both_binary_labels": sum(1 for n in hash_counts.values() if n[3] == 3)}


def run(input_dir, output_dir, workers=4, shard_size=1000, variance_threshold=1e-8, redundancy_threshold=0.999):
    source = Path(input_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    previous = read_json(source / "step04_summary.json")
    selection = read_json(source / "selected_features.json")
    features = selection["feature_names"]
    if (previous.get("step") != "04_rf_shap_single_iot23_csv" or previous["window_rows"] != 15
            or previous["stride"] != 15 or features != previous["selected_features"] or len(set(features)) != 15):
        raise ValueError("Unexpected Step 4 schema or window definition.")
    tracked = ["X_selected.npy", "row_split.npy", "window_split.npy", "window_targets.npy",
               "preprocessing.json", "selected_features.json", "step04_summary.json"]
    signatures = {name: {"bytes": (source / name).stat().st_size, "mtime_ns": (source / name).stat().st_mtime_ns}
                  for name in tracked}
    configuration = {"version": VERSION, "source_dir": str(source), "inputs": signatures,
                     "features": features, "shard_size": shard_size,
                     "variance_threshold": variance_threshold, "redundancy_threshold": redundancy_threshold,
                     "levels": LEVELS.tolist(), "palette": "coolwarm", "pixel_shape": [224, 224, 3],
                     "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__, "pillow": PIL.__version__}}
    configuration_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    # SQLite's OS-managed exclusive lock prevents simultaneous writers and is
    # released automatically if the process exits. There is no stale PID lock.
    lock = sqlite3.connect(out / "run_lock.sqlite3", timeout=0)
    maps = []
    started = time.perf_counter()
    try:
        lock.execute("CREATE TABLE IF NOT EXISTS writer_lock (id INTEGER PRIMARY KEY)")
        lock.execute("BEGIN EXCLUSIVE")
        config_path = out / "generation_config.json"
        if config_path.exists():
            if read_json(config_path) != configuration:
                raise ValueError("Input/configuration changed. Choose a new --output-dir to keep runs separate.")
        elif any(p.name != "run_lock.sqlite3" for p in out.iterdir()):
            raise ValueError("Output folder contains unrelated files. Choose a new --output-dir.")
        else:
            atomic_json(config_path, configuration)
        X = np.load(source / "X_selected.npy", mmap_mode="r", allow_pickle=False)
        maps.append(X)
        row_split = np.load(source / "row_split.npy", mmap_mode="r", allow_pickle=False)
        maps.append(row_split)
        splits = np.load(source / "window_split.npy", mmap_mode="r", allow_pickle=False)
        maps.append(splits)
        labels = np.load(source / "window_targets.npy", mmap_mode="r", allow_pickle=False)
        maps.append(labels)
        total, n_windows = int(previous["rows_in_X_selected"]), int(previous["candidate_windows"])
        used = n_windows * WINDOW_ROWS
        if (X.shape != (total, 15) or X.dtype != np.float64 or row_split.shape != (total,)
                or splits.shape != (n_windows,) or labels.shape != (n_windows,)
                or total // WINDOW_ROWS != n_windows or not np.isin(splits, [0, 1, 2]).all()
                or not np.isin(labels, [0, 1]).all()):
            raise ValueError("Step 4 array shapes or values disagree with its summary.")
        if (not np.array_equal(row_split[:used].reshape(-1, WINDOW_ROWS), np.broadcast_to(splits[:, None], (n_windows, WINDOW_ROWS)))
                or not np.all(row_split[used:] == -1)
                or hashlib.sha256(row_split.tobytes()).hexdigest() != previous["row_split_sha256"]
                or counts_by_split(labels, splits) != previous["window_class_counts"]):
            raise ValueError("Saved window membership or class counts changed after Step 4.")
        print(f"Input: {source}\nComplete windows: {n_windows:,}", flush=True)
        correlations = create_correlations(source, out, X, row_split, splits, features, variance_threshold, redundancy_threshold)
        for filename in ("correlations_raw.npy", "correlations_filtered.npy"):
            check = np.load(out / filename, mmap_mode="r", allow_pickle=False)
            try:
                if check.shape != (n_windows, 15, 15) or check.dtype != np.float32:
                    raise ValueError(f"Invalid cached matrix file: {filename}")
            finally:
                close_map(check)
        render_all(out, n_windows, shard_size, workers, configuration_hash)
        diagnostics = index_and_previews(out, labels, splits, shard_size)
        atomic_array(out / "labels.npy", np.asarray(labels))
        atomic_array(out / "splits.npy", np.asarray(splits))
        report = {"step": "05_contours_single_iot23_csv", "input_dir": str(source), "source_file": previous["source_file"],
                  "validation_fold": previous["validation_fold"], "configuration_hash": configuration_hash,
                  "images": n_windows, "complete_windows_dropped": 0,
                  "tail_rows_excluded_as_in_step04": total - used,
                  "image_shape_hwc": [224, 224, 3], "stored_dtype": "uint8", "stored_range": [0, 255],
                  "model_input_shape_chw": [3, 224, 224], "model_input_dtype": "float32",
                  "model_input_range": [0, 1], "model_input_conversion": "RGB pixels / 255, then HWC -> CHW",
                  "feature_order": features, "shard_size": shard_size, "image_archives": math.ceil(n_windows / shard_size),
                  "window_class_counts": counts_by_split(labels, splits),
                  "correlations": correlations, "image_diagnostics": diagnostics,
                  "versions": configuration["versions"], "elapsed_seconds_this_invocation": round(time.perf_counter() - started, 2),
                  "detector_trained": False,
                  "notes": [
                      "All Step 4 complete windows, endpoint labels and assignments are preserved.",
                      "Normalization extrema and correlation filtering are fitted only on the training partition.",
                      "Filtering thresholds, constant handling and rendering settings are explicit reconstruction choices.",
                      "Zero entries can represent undefined or masked correlations; consult raw matrices and activity/mask files.",
                      "Identical rendered inputs across partitions are reported, not automatically treated as shared source rows.",
                      "Only this fixed validation fold is represented; this is not a completed five-fold experiment.",
                  ]}
        atomic_json(out / "step05_summary.json", report)
        print("\nSTEP 5 COMPLETE", flush=True)
        print(f"Contour images: {n_windows:,}")
        print("Image size: 224 x 224; channels: RGB")
        print(f"ZIP batches: {report['image_archives']}")
        print(f"Complete windows dropped: 0\nClass counts: {report['window_class_counts']}")
        print(f"Preview folder: {out / 'previews'}")
        print(f"Summary: {out / 'step05_summary.json'}")
        return report
    finally:
        for array in maps:
            close_map(array)
        lock.rollback()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description="Step 5: Pearson correlations and resumable contour image generation.")
    parser.add_argument("--input-dir", type=Path, default=Path("prepared/dataset7_step04"))
    parser.add_argument("--output-dir", type=Path, default=Path("prepared/dataset7_step05"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--variance-threshold", type=float, default=1e-8)
    parser.add_argument("--redundancy-threshold", type=float, default=0.999)
    args = parser.parse_args()
    if (args.workers < 1 or args.shard_size < 1 or not np.isfinite(args.variance_threshold)
            or args.variance_threshold < 0 or not 0 < args.redundancy_threshold <= 1):
        parser.error("Use positive workers/shard-size, nonnegative finite variance, and redundancy in (0,1].")
    try:
        run(args.input_dir, args.output_dir, args.workers, args.shard_size, args.variance_threshold, args.redundancy_threshold)
    except KeyboardInterrupt:
        print("\nInterrupted. Rerun the same command to resume completed phases and ZIP batches.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"\nSTEP 5 FAILED: {error}", file=sys.stderr)
        print("Completed phases and ZIP batches are retained for a matching rerun.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
