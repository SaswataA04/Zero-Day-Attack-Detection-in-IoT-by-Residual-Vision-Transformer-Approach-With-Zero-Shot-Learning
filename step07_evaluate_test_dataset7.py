#!/usr/bin/env python
"""
STEP 7 — ONE-SHOT TEST EVALUATION FOR DATASET 7

Purpose
-------
Evaluate the already-frozen Step-6D best_model.pt exactly once on the untouched
Dataset-7 test split.

This script is evaluation-only:
* NO optimizer
* NO scheduler
* NO backward pass
* NO training
* NO checkpoint modification
* NO threshold search
* fixed decision threshold = 0.5
* loads only the frozen best_model.pt
* test images are decoded only during this evaluation step

Expected frozen model:
    trained/dataset7_step06d_hash_balanced_trial01/best_model.pt

Expected clean dataset:
    prepared/dataset7_step05c_hashclean_no_uid_v2

Recommended command:
    python step07_evaluate_test_dataset7.py ^
      --input-dir "prepared/dataset7_step05c_hashclean_no_uid_v2" ^
      --model "trained/dataset7_step06d_hash_balanced_trial01/best_model.pt" ^
      --output-dir "evaluation/dataset7_step07_test_trial01"

PowerShell multiline version uses backticks instead of ^.

Outputs
-------
step07_test_summary.json
test_predictions.csv
confusion_matrix.csv
evaluation_manifest.json

Scientific rule
---------------
Do NOT use the test results from this script to:
* choose another checkpoint,
* alter the 0.5 threshold,
* change model architecture,
* change sampling,
* change preprocessing,
* tune hyperparameters.

If methodology changes after viewing test results, that is a NEW experiment and
requires a new untouched final holdout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import multiprocessing
import os
import platform
import random
import sys
import time
import zipfile
from collections import OrderedDict
from pathlib import Path

try:
    import numpy as np
    import PIL
    from PIL import Image
    import sklearn
    from sklearn.metrics import average_precision_score, roc_auc_score
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
    import torchvision
    from torchvision.models import resnet50
except ImportError as error:
    raise SystemExit(
        "Missing dependency. Install torch/torchvision plus numpy pillow scikit-learn.\n"
        f"Original error: {error}"
    ) from error


VERSION = 1
CLASS_NAMES = ("benign", "malicious")
SPLIT_NAMES = ("train", "validation", "test")
FIXED_THRESHOLD = 0.5


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, content):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(content, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def class_counts(labels, splits):
    return {
        name: {
            CLASS_NAMES[c]: int(
                np.count_nonzero((splits == s) & (labels == c))
            )
            for c in (0, 1)
        }
        for s, name in enumerate(SPLIT_NAMES)
    }


def inspect_inputs(root):
    """
    Verify Step-5C input integrity.

    This stage reads metadata/arrays only. It does not decode image pixels.
    """
    summary = read_json(root / "step05_summary.json")

    if (
        summary.get("step") != "05_contours_single_iot23_csv"
        or summary.get("complete_windows_dropped") != 0
        or summary.get("image_shape_hwc") != [224, 224, 3]
        or summary.get("model_input_range") != [0, 1]
    ):
        raise ValueError(
            "Expected a completed Step-5/5C dataset with 224x224 RGB images."
        )

    hashclean = summary.get("step05c_hashclean_validation")
    if not isinstance(hashclean, dict):
        raise ValueError(
            "This evaluator expects the Step-5C hash-clean dataset view."
        )
    if hashclean.get("original_train_windows_moved_to_validation") != 0:
        raise ValueError(
            "Step-5C safety invariant failed: original training entered validation."
        )
    if hashclean.get("test_assignments_changed") != 0:
        raise ValueError("Step-5C reports modified test assignments.")

    labels = np.load(root / "labels.npy", allow_pickle=False)
    splits = np.load(root / "splits.npy", allow_pickle=False)

    n = int(summary["images"])
    shard_size = int(summary["shard_size"])

    if (
        shard_size <= 0
        or labels.shape != (n,)
        or splits.shape != (n,)
        or not np.isin(labels, [0, 1]).all()
        or not np.isin(splits, [0, 1, 2]).all()
    ):
        raise ValueError("Invalid labels/splits arrays.")

    observed_counts = class_counts(labels, splits)
    if observed_counts != summary["window_class_counts"]:
        raise ValueError(
            "labels.npy/splits.npy disagree with step05_summary.json."
        )

    if min(observed_counts["test"].values()) < 1:
        raise ValueError("Test split must contain both classes.")

    # Recreate the same dataset fingerprint used by Step 6D.
    archives = []
    for shard, start in enumerate(range(0, n, shard_size)):
        path = root / "images" / f"shard_{shard:05d}.zip"
        marker_path = path.with_suffix(".json")

        if not path.is_file() or not marker_path.is_file():
            raise FileNotFoundError(
                f"Missing image shard or metadata: {path.name}"
            )

        marker = read_json(marker_path)
        stat = path.stat()

        if (
            marker["configuration_hash"] != summary["configuration_hash"]
            or marker["start"] != start
            or marker["stop"] != min(start + shard_size, n)
            or marker["archive_bytes"] != stat.st_size
            or len(marker["png_sha256"]) != marker["stop"] - start
        ):
            raise ValueError(
                f"Incomplete or inconsistent image shard: {path.name}"
            )

        archives.append(
            {
                "name": path.name,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "metadata_sha256": file_digest(marker_path),
            }
        )

    if len(archives) != int(summary["image_archives"]):
        raise ValueError("Step-5 archive count disagrees with summary.")

    signature = {
        "step05_configuration_hash": summary["configuration_hash"],
        "labels_sha256": file_digest(root / "labels.npy"),
        "splits_sha256": file_digest(root / "splits.npy"),
        "archives": archives,
    }

    fingerprint = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode()
    ).hexdigest()

    return summary, labels, splits, fingerprint


class ContourZipDataset(Dataset):
    """Decode only the requested windows directly from Step-5 ZIP shards."""

    def __init__(self, root, indices, labels, shard_size):
        self.root = Path(root)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.shard_size = int(shard_size)
        self._archives = OrderedDict()
        self._pid = os.getpid()

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_archives"] = OrderedDict()
        state["_pid"] = None
        return state

    def close(self):
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def __getitem__(self, position):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()

        window = int(self.indices[position])
        shard = window // self.shard_size

        if shard not in self._archives:
            path = self.root / "images" / f"shard_{shard:05d}.zip"
            self._archives[shard] = zipfile.ZipFile(path)

            if len(self._archives) > 256:
                self._archives.popitem(last=False)[1].close()

        self._archives.move_to_end(shard)

        encoded = self._archives[shard].read(
            f"window_{window:09d}.png"
        )

        with Image.open(io.BytesIO(encoded)) as picture:
            if picture.size != (224, 224) or picture.mode != "RGB":
                raise ValueError(
                    f"Window {window}: expected 224x224 RGB PNG."
                )
            array = np.array(picture, dtype=np.uint8, copy=True)

        image = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return image, int(self.labels[window]), window


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


def make_loader(dataset, args, device):
    kwargs = {}
    if args.workers:
        kwargs.update(
            multiprocessing_context="spawn",
            prefetch_factor=2,
        )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        timeout=120 if args.workers else 0,
        generator=torch.Generator().manual_seed(12345),
        **kwargs,
    )


class CZResViTBinary(nn.Module):
    """Same architecture used by Step 6D."""

    def __init__(
        self,
        embedding_dim=256,
        depth=4,
        heads=8,
        dropout=0.1,
    ):
        super().__init__()

        backbone = resnet50(weights=None)
        self.backbone = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        self.projection = nn.Linear(2048, embedding_dim)
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embedding_dim)
        )
        self.position = nn.Parameter(
            torch.zeros(1, 50, embedding_dim)
        )
        self.embedding_dropout = nn.Dropout(dropout)

        self.encoders = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=heads,
                    dim_feedforward=4 * embedding_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )

        self.final_norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 2),
        )

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, images):
        features = self.backbone(images)

        if features.shape[1:] != (2048, 7, 7):
            raise ValueError(
                "Expected ResNet output [B,2048,7,7]; "
                f"got {tuple(features.shape)}"
            )

        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.projection(tokens)

        cls = self.cls_token.expand(
            tokens.shape[0], -1, -1
        )
        tokens = torch.cat((cls, tokens), dim=1)

        tokens = self.embedding_dropout(
            tokens + self.position
        )

        for encoder in self.encoders:
            tokens = encoder(tokens)

        return self.head(
            self.final_norm(tokens)[:, 0]
        )


def metrics_from_confusion(confusion):
    matrix = np.asarray(confusion, dtype=np.int64)

    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)

    true_positive = np.diag(matrix).astype(np.float64)

    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros(2),
        where=predicted != 0,
    )

    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(2),
        where=support != 0,
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(2),
        where=(precision + recall) != 0,
    )

    n = int(support.sum())

    return {
        "n": n,
        "accuracy": float(true_positive.sum() / n),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / n),
        "balanced_accuracy": float(recall.mean()),
        "confusion_matrix_actual_rows_predicted_columns": matrix.tolist(),
        "confusion_order": list(CLASS_NAMES),
        "per_class": {
            name: {
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
                "support": int(support[c]),
                "predicted_count": int(predicted[c]),
            }
            for c, name in enumerate(CLASS_NAMES)
        },
    }


def probability_metrics(labels, probability):
    # Fixed decision rule. Ties at exactly 0.5 become benign.
    prediction = (
        probability > FIXED_THRESHOLD
    ).astype(np.int64)

    matrix = np.bincount(
        labels * 2 + prediction,
        minlength=4,
    ).reshape(2, 2)

    result = metrics_from_confusion(matrix)

    if len(np.unique(labels)) == 2:
        result.update(
            roc_auc_malicious=float(
                roc_auc_score(labels, probability)
            ),
            average_precision_malicious=float(
                average_precision_score(labels, probability)
            ),
            average_precision_benign=float(
                average_precision_score(
                    1 - labels,
                    1 - probability,
                )
            ),
        )

    result["decision_threshold"] = FIXED_THRESHOLD
    return result


def choose_device(args):
    if args.device == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    if args.device == "auto":
        return torch.device("cpu")

    raise RuntimeError(
        "CUDA requested but unavailable in this Python environment."
    )


def runtime_info(device, amp, args):
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "scikit_learn": sklearn.__version__,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "cuda_runtime": torch.version.cuda,
        "amp": amp,
        "workers": args.workers,
        "torch_threads": torch.get_num_threads(),
    }


def load_frozen_model(model_path, summary, fingerprint, device):
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    required = {
        "model_config",
        "model_state",
        "epoch",
        "validation_metrics",
        "configuration",
        "config_hash",
        "class_names",
        "input_conversion",
    }

    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            f"Frozen checkpoint is missing fields: {sorted(missing)}"
        )

    config = checkpoint["configuration"]

    if checkpoint["class_names"] != list(CLASS_NAMES):
        raise ValueError(
            "Checkpoint class order is not benign/malicious."
        )

    if (
        checkpoint["input_conversion"]
        != "RGB uint8 -> float32 / 255"
    ):
        raise ValueError(
            "Checkpoint input conversion does not match evaluator."
        )

    if (
        config.get("step05_configuration_hash")
        != summary["configuration_hash"]
    ):
        raise ValueError(
            "Checkpoint and Step-5C dataset configuration hashes differ."
        )

    if config.get("data_fingerprint") != fingerprint:
        raise ValueError(
            "Checkpoint dataset fingerprint does not match this input dataset."
        )

    if (
        config.get("sampling", {}).get("method")
        != "exact_png_hash_balanced"
    ):
        raise ValueError(
            "Checkpoint is not the Step-6D hash-balanced model."
        )

    if config.get("class_weighting") != "none":
        raise ValueError(
            "Unexpected checkpoint: Step 6D should use unweighted loss."
        )

    if config.get("class_weights") != [1.0, 1.0]:
        raise ValueError(
            "Unexpected checkpoint class weights."
        )

    model_config = checkpoint["model_config"]

    if model_config != config.get("model"):
        raise ValueError(
            "Checkpoint model configuration is internally inconsistent."
        )

    model = CZResViTBinary(**model_config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()

    return checkpoint, model


@torch.inference_mode()
def evaluate_test(model, loader, device, amp, log_every):
    labels = []
    probabilities = []
    window_ids = []

    cross_entropy_sum = 0.0
    seen = 0

    began = time.perf_counter()
    last_log = began

    for batch_index, (images, target, ids) in enumerate(loader):
        images = images.to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        ).div_(255.0)

        target = target.to(
            device,
            dtype=torch.long,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
        ):
            logits = model(images)

        logits32 = logits.float()
        loss = F.cross_entropy(
            logits32,
            target,
            reduction="sum",
        )
        probability = logits32.softmax(dim=1)[:, 1]

        if (
            not torch.isfinite(loss)
            or not torch.isfinite(probability).all()
        ):
            raise FloatingPointError(
                "Nonfinite test output encountered."
            )

        cross_entropy_sum += float(loss)
        seen += len(target)

        labels.append(target.cpu().numpy())
        probabilities.append(probability.cpu().numpy())
        window_ids.append(ids.numpy())

        now = time.perf_counter()

        if (
            batch_index == 0
            or (batch_index + 1) % log_every == 0
            or batch_index + 1 == len(loader)
            or now - last_log >= 30
        ):
            speed = seen / max(now - began, 1e-9)
            remaining = len(loader.dataset) - seen
            eta = remaining / max(speed, 1e-9) / 60

            print(
                f"  Test evaluation: {seen:,}/{len(loader.dataset):,} images; "
                f"{speed:.1f} images/s; ETA {eta:.1f} min",
                flush=True,
            )
            last_log = now

    y = np.concatenate(labels)
    p = np.concatenate(probabilities)
    ids = np.concatenate(window_ids)

    metrics = probability_metrics(y, p)
    metrics["cross_entropy"] = (
        cross_entropy_sum / len(y)
    )
    metrics["elapsed_seconds"] = round(
        time.perf_counter() - began,
        2,
    )

    predictions = (
        p > FIXED_THRESHOLD
    ).astype(np.int64)

    return metrics, ids, y, p, predictions


def save_predictions(path, ids, labels, probability, predictions):
    temporary = path.with_name(path.name + ".partial")

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as destination:
        writer = csv.writer(destination)
        writer.writerow(
            [
                "window_id",
                "target",
                "probability_malicious",
                "predicted_target",
                "correct",
            ]
        )

        for window, label, prob, prediction in zip(
            ids.tolist(),
            labels.tolist(),
            probability.tolist(),
            predictions.tolist(),
        ):
            writer.writerow(
                [
                    int(window),
                    int(label),
                    float(prob),
                    int(prediction),
                    int(label == prediction),
                ]
            )

    temporary.replace(path)


def save_confusion(path, metrics):
    matrix = metrics[
        "confusion_matrix_actual_rows_predicted_columns"
    ]

    temporary = path.with_name(path.name + ".partial")

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as destination:
        writer = csv.writer(destination)
        writer.writerow(
            [
                "actual_class",
                "predicted_benign",
                "predicted_malicious",
            ]
        )
        writer.writerow(["benign", *matrix[0]])
        writer.writerow(["malicious", *matrix[1]])

    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Step 7: one-shot evaluation of frozen Step-6D model "
            "on Dataset-7 test split."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=250,
    )

    args = parser.parse_args()

    if (
        args.batch_size < 1
        or args.workers < 0
        or args.torch_threads < 1
        or args.log_every < 1
    ):
        parser.error(
            "Use positive batch/thread/log sizes and nonnegative workers."
        )

    root = args.input_dir.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {root}"
        )
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Frozen model does not exist: {model_path}"
        )

    # Intentionally refuse to overwrite/reuse a completed evaluation folder.
    if out.exists():
        if any(out.iterdir()):
            raise FileExistsError(
                f"Evaluation output already exists and is not empty:\n{out}\n"
                "Do not overwrite a final test evaluation. "
                "Inspect the existing result instead."
            )
    else:
        out.mkdir(parents=True)

    torch.set_num_threads(args.torch_threads)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = choose_device(args)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False

    amp = device.type == "cuda" and not args.no_amp
    environment = runtime_info(device, amp, args)

    print("STEP 7 — ONE-SHOT TEST EVALUATION", flush=True)
    print(f"Input : {root}", flush=True)
    print(f"Model : {model_path}", flush=True)
    print(f"Output: {out}", flush=True)
    print(
        f"Device: {environment['gpu'] or str(device)}; "
        f"PyTorch {environment['torch']}; AMP={amp}",
        flush=True,
    )
    print(
        f"Decision threshold is FIXED at {FIXED_THRESHOLD:.1f}. "
        "No threshold search is performed.",
        flush=True,
    )
    print(
        "No optimizer, backward pass, training, or checkpoint update exists in this step.",
        flush=True,
    )

    print("\n1/4 Verifying Step-5C dataset metadata...", flush=True)
    summary, labels, splits, fingerprint = inspect_inputs(root)

    counts = class_counts(labels, splits)
    test_ids = np.flatnonzero(splits == 2)

    print(f"Image counts: {counts}", flush=True)
    print(
        f"Frozen test set: {len(test_ids):,} images "
        f"({counts['test']['benign']:,} benign + "
        f"{counts['test']['malicious']:,} malicious)",
        flush=True,
    )

    print("\n2/4 Loading and verifying frozen Step-6D checkpoint...", flush=True)
    checkpoint, model = load_frozen_model(
        model_path,
        summary,
        fingerprint,
        device,
    )

    validation = checkpoint["validation_metrics"]

    print(
        f"Frozen checkpoint epoch: {checkpoint['epoch']}",
        flush=True,
    )
    print(
        f"Frozen validation macro F1: {validation['macro_f1']:.6f}",
        flush=True,
    )
    print(
        f"Frozen validation balanced accuracy: "
        f"{validation['balanced_accuracy']:.6f}",
        flush=True,
    )
    print(
        "Checkpoint verification passed. Model is now inference-only.",
        flush=True,
    )

    baseline_probability = np.ones(
        len(test_ids),
        dtype=np.float32,
    )
    always_malicious_baseline = probability_metrics(
        labels[test_ids],
        baseline_probability,
    )

    print(
        "Always-malicious TEST baseline: "
        f"accuracy={always_malicious_baseline['accuracy']:.6f}; "
        f"macro F1={always_malicious_baseline['macro_f1']:.6f}; "
        "benign recall=0.000000",
        flush=True,
    )

    print(
        "\n3/4 Decoding the held-out test images for the FIRST evaluation...",
        flush=True,
    )

    test_data = ContourZipDataset(
        root,
        test_ids,
        labels,
        summary["shard_size"],
    )

    try:
        loader = make_loader(
            test_data,
            args,
            device,
        )

        metrics, ids, y, probability, predictions = evaluate_test(
            model,
            loader,
            device,
            amp,
            args.log_every,
        )
    finally:
        test_data.close()

    print("\n4/4 Saving immutable test-evaluation outputs...", flush=True)

    save_predictions(
        out / "test_predictions.csv",
        ids,
        y,
        probability,
        predictions,
    )

    save_confusion(
        out / "confusion_matrix.csv",
        metrics,
    )

    checkpoint_sha256 = file_digest(model_path)

    manifest = {
        "step": "07_one_shot_test_evaluation",
        "script_version": VERSION,
        "input_dir": str(root),
        "dataset_fingerprint": fingerprint,
        "model_path": str(model_path),
        "model_sha256": checkpoint_sha256,
        "frozen_checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_config_hash": checkpoint["config_hash"],
        "decision_threshold": FIXED_THRESHOLD,
        "threshold_tuned_on_test": False,
        "training_performed": False,
        "checkpoint_modified": False,
        "test_images_evaluated": int(len(test_ids)),
        "test_class_counts": counts["test"],
        "runtime": environment,
    }

    atomic_json(
        out / "evaluation_manifest.json",
        manifest,
    )

    report = {
        "step": "07_one_shot_dataset7_test_evaluation",
        "model_status": "frozen",
        "frozen_checkpoint_epoch": int(checkpoint["epoch"]),
        "frozen_validation_metrics": validation,
        "test_metrics": metrics,
        "always_malicious_test_baseline": always_malicious_baseline,
        "test_class_counts": counts["test"],
        "decision_threshold": FIXED_THRESHOLD,
        "threshold_tuned_on_test": False,
        "test_evaluated": True,
        "zero_day_evaluated": False,
        "model_sha256": checkpoint_sha256,
        "checkpoint_config_hash": checkpoint["config_hash"],
        "dataset_fingerprint": fingerprint,
        "notes": [
            "This is the first final evaluation of the frozen Step-6D best checkpoint on the Dataset-7 test split.",
            "No optimizer, gradient calculation, training, checkpoint update or threshold search was performed.",
            "The classification threshold was fixed at 0.5 before test inference.",
            "ROC AUC and average precision are ranking metrics and were computed without threshold tuning.",
            "Because the test set is extremely imbalanced, accuracy alone must not be used to judge the classifier.",
            "Do not alter the frozen methodology based on this test result and then report the same test set as an untouched final holdout.",
        ],
    }

    atomic_json(
        out / "step07_test_summary.json",
        report,
    )

    print("\nSTEP 7 COMPLETE", flush=True)
    print(
        f"Test accuracy: {metrics['accuracy']:.6f}",
        flush=True,
    )
    print(
        f"Test macro F1: {metrics['macro_f1']:.6f}",
        flush=True,
    )
    print(
        f"Test balanced accuracy: {metrics['balanced_accuracy']:.6f}",
        flush=True,
    )
    print(
        f"Test ROC-AUC (malicious): {metrics['roc_auc_malicious']:.6f}",
        flush=True,
    )
    print(
        f"Test AP (malicious): {metrics['average_precision_malicious']:.6f}",
        flush=True,
    )
    print(
        f"Test AP (benign): {metrics['average_precision_benign']:.6f}",
        flush=True,
    )

    for name in CLASS_NAMES:
        values = metrics["per_class"][name]
        print(
            f"  {name}: "
            f"precision={values['precision']:.6f}; "
            f"recall={values['recall']:.6f}; "
            f"F1={values['f1']:.6f}; "
            f"support={values['support']:,}",
            flush=True,
        )

    matrix = metrics[
        "confusion_matrix_actual_rows_predicted_columns"
    ]

    print(
        "Confusion matrix "
        "(actual rows [benign, malicious], "
        "predicted columns [benign, malicious]):",
        flush=True,
    )
    print(f"  {matrix[0]}", flush=True)
    print(f"  {matrix[1]}", flush=True)

    print(
        f"Predictions: {out / 'test_predictions.csv'}",
        flush=True,
    )
    print(
        f"Summary: {out / 'step07_test_summary.json'}",
        flush=True,
    )
    print(
        "\nIMPORTANT: Do not tune this frozen model using these test results.",
        flush=True,
    )
    print(
        "Next research step after reviewing this output: "
        "design the separate zero-day evaluation.",
        flush=True,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()

    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nSTEP 7 INTERRUPTED. No model was modified. "
            "Delete the partial output folder and rerun the same frozen evaluation.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as error:
        print(
            f"\nSTEP 7 FAILED: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
