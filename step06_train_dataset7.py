"""Step 6: train a documented binary reconstruction of the CZ-ResViT base model.

Run from your iot_zero_day_research project folder:
    python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126
    python step06_train_dataset7.py --smoke-test
    python step06_train_dataset7.py --epochs 1

After reviewing that epoch, extend the SAME run:
    python step06_train_dataset7.py --epochs 10

--epochs is the TOTAL target, not the number of additional epochs. A matching
run automatically resumes the last completed epoch, including optimizer state.
An interrupted, incomplete epoch is replayed from its beginning. The smoke test
uses a small train/validation sample and saves no model or benchmark metrics.

Inputs:  prepared/dataset7_step05/
Outputs: trained/dataset7_step06/
The ZIP files stay compressed. Images are decoded on demand, not all at once.

WHAT FOLLOWS THE SUPPLIED PDF (Section III, Figures 7 and 9):
    [B,3,224,224] RGB contours
      -> ResNet50 convolutional stem and residual stages [3,4,6,3]
      -> [B,2048,7,7] feature map
      -> [B,49,2048] spatial tokens
      -> linear projection, learnable CLS and positional embeddings
      -> Transformer encoders -> final CLS -> MLP classification head.
ResNet global average pooling and its original classifier are excluded: the
spatial 7x7 grid must reach the Transformer. All backbone layers are trainable.
Forward returns logits; cross-entropy applies its own log-softmax. Inference
uses two-way softmax. No additional patchification of the RGB image is used.

EXPLICIT IMPLEMENTATION CHOICES / LIMITS (not asserted author settings):
* Dataset7 uses the agreed TWO outputs: 0=benign, 1=malicious. The paper's base
  model used five outputs and a much smaller, class-balanced image selection.
* Reuse every Step 5 train/validation window without changing labels, splits,
  preprocessing, image colors, or feature order. Test image pixels are never
  read by this script. Test labels are used only to verify saved split counts.
* Transformer: dimension 256, four independent encoder layers, eight attention
  heads, feed-forward width 1024, GELU, pre-layer normalization, dropout 0.1.
  Final head: LayerNorm on the tokens, then Linear(256,256), GELU, Dropout,
  Linear(256,2) on CLS. These sizes are not specified for the hybrid in the PDF.
* Torchvision's ResNet50 implementation, initialized from scratch; no external
  pretrained checkpoint. Model input is RGB / 255 in [0,1], exactly as Step 5
  defines. No ImageNet normalization, random rotations/flips, or augmentation.
* AdamW, learning rate 1e-4, weight decay 1e-4, batch size 8, four accumulated
  microbatches (normally 32 images per update), global gradient-norm limit 1.
  ResNet BatchNorm still uses the actual microbatch of eight, not 32 images.
* Default loss: unweighted cross-entropy, keeping this initial baseline free
  of an extra imbalance-treatment change. For a SEPARATE experiment in a NEW
  output folder, --class-weighting balanced enables training-only weights
  weight[c] = N_train / (2 * N_train_class[c]). Apply any such weights to
  per-example cross-entropy and divide by the NUMBER OF EXAMPLES per optimizer
  update, never by the weight sum of each small batch. Every train image is
  visited once each epoch; no over/undersampling is performed.
* Validate on the complete fixed validation partition, with its original class
  proportions. Select the best epoch by macro F1; break exact ties by lower
  unweighted validation cross-entropy. Use softmax argmax (ties -> benign).
  Use patience=2 for validation-macro-F1 learning-rate reduction; stop
  after five epochs without a better checkpoint. Default maximum is 20 epochs.
* CUDA automatic mixed precision saves memory. AMP is NOT integer quantization.
  The paper's quantization recipe is unresolved and is not reproduced here.
  Parameter counts from this reconstruction need not match the paper's table.
* Report per-class precision/recall/F1, macro/weighted F1, balanced accuracy,
  ROC AUC and average precision for each class. Average precision is named
  explicitly; it is not the trapezoidal area under an interpolated PR curve.
* This is ONE fixed validation fold and supervised binary base-model training.
  It does not complete five-fold cross-validation or establish zero-day results.
  The test partition is reserved for a separate final evaluation step.

Reproducibility: seed Python/NumPy/PyTorch; deterministic epoch permutations;
record versions/configuration and RNG state. CUDA kernels are not forced into
strict deterministic mode, so bit-for-bit GPU reproducibility is not promised.

Local smoke test without a GPU:
    python step06_train_dataset7.py --smoke-test --device cpu --workers 0
For a deliberately slower CPU training run, explicitly pass --device cpu.
For GPU memory errors, use a NEW output folder with smaller microbatches:
    python step06_train_dataset7.py --batch-size 4 --accumulation-steps 8 --output-dir trained/dataset7_step06_b4 --epochs 1
Changing methodology/settings requires a new folder. Changing --epochs,
--workers, --log-every or --device is allowed on resume and recorded in history.
"""

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
import sqlite3
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
    from torch.utils.data import DataLoader, Dataset, Sampler
    import torchvision
    from torchvision.models import resnet50
except ImportError as error:
    raise SystemExit(
        "Missing dependency. Install torch/torchvision using the command in this "
        "script's introduction, plus: python -m pip install numpy pillow scikit-learn\n"
        f"Original error: {error}"
    ) from error


VERSION = 1
CLASS_NAMES = ("benign", "malicious")
SPLIT_NAMES = ("train", "validation", "test")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, content):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(path, content):
    # Close the file before replacing it, including on Windows.
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as destination:
        torch.save(content, destination)
    temporary.replace(path)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def class_counts(labels, splits):
    return {name: {CLASS_NAMES[c]: int(np.count_nonzero((splits == s) & (labels == c)))
                   for c in (0, 1)} for s, name in enumerate(SPLIT_NAMES)}


def inspect_inputs(root):
    """Verify Step 5's contract without decoding any test images."""
    summary = read_json(root / "step05_summary.json")
    if (summary.get("step") != "05_contours_single_iot23_csv"
            or summary.get("complete_windows_dropped") != 0
            or summary.get("image_shape_hwc") != [224, 224, 3]
            or summary.get("model_input_range") != [0, 1]):
        raise ValueError("Expected a completed, unchanged Step 5 run with 224x224 RGB images.")
    labels = np.load(root / "labels.npy", allow_pickle=False)
    splits = np.load(root / "splits.npy", allow_pickle=False)
    n, shard_size = int(summary["images"]), int(summary["shard_size"])
    if (shard_size <= 0 or labels.shape != (n,) or splits.shape != (n,)
            or not np.isin(labels, [0, 1]).all() or not np.isin(splits, [0, 1, 2]).all()
            or class_counts(labels, splits) != summary["window_class_counts"]):
        raise ValueError("Step 5 labels/splits do not match its summary.")
    for split in (0, 1):
        if not all(summary["window_class_counts"][SPLIT_NAMES[split]][c] > 0 for c in CLASS_NAMES):
            raise ValueError("Both classes must be present in training and validation.")
    # These file signatures also prevent an accidental mixture of two Step 5
    # runs on resume. Metadata reads do not load held-out image contents.
    archives = []
    for shard, start in enumerate(range(0, n, shard_size)):
        path = root / "images" / f"shard_{shard:05d}.zip"
        marker = read_json(path.with_suffix(".json"))
        stat = path.stat()
        if (marker["configuration_hash"] != summary["configuration_hash"]
                or marker["start"] != start or marker["stop"] != min(start + shard_size, n)
                or marker["archive_bytes"] != stat.st_size
                or len(marker["png_sha256"]) != marker["stop"] - start):
            raise ValueError(f"Incomplete or inconsistent image batch: {path.name}")
        archives.append({"name": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                         "metadata_sha256": file_digest(path.with_suffix(".json"))})
    if len(archives) != int(summary["image_archives"]):
        raise ValueError("Step 5 archive count disagrees with its summary.")
    signature = {"step05_configuration_hash": summary["configuration_hash"],
                 "labels_sha256": file_digest(root / "labels.npy"),
                 "splits_sha256": file_digest(root / "splits.npy"),
                 "archives": archives}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    print(f"Image counts: {class_counts(labels, splits)}", flush=True)
    print("Training/validation use the saved assignments; test image pixels stay unread.", flush=True)
    return summary, labels, splits, fingerprint


class ContourZipDataset(Dataset):
    """Each worker opens its own ZIP handles and decodes only requested windows.

    Return uint8 CHW images to keep transfer buffers small. The training loop
    performs float32 conversion and division by 255 after moving to the device.
    A 256-archive cache can hold all 239 dataset7 archives in each worker.
    """

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
        # File handles cannot be pickled into Windows DataLoader workers.
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
        encoded = self._archives[shard].read(f"window_{window:09d}.png")
        with Image.open(io.BytesIO(encoded)) as picture:
            if picture.size != (224, 224) or picture.mode != "RGB":
                raise ValueError(f"Window {window}: expected a 224x224 RGB PNG.")
            array = np.array(picture, dtype=np.uint8, copy=True)
        image = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return image, int(self.labels[window]), window


class EpochShuffleSampler(Sampler):
    """The epoch permutation depends only on seed and epoch, including resume."""

    def __init__(self, size, seed):
        self.size, self.seed, self.epoch = size, seed, 0

    def __len__(self):
        return self.size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        yield from torch.randperm(self.size, generator=generator).tolist()


def seed_worker(worker_id):
    # There are no random image transforms, but explicitly seed worker RNGs.
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


def make_loader(dataset, args, device, sampler=None):
    kwargs = {}
    if args.workers:
        kwargs.update(multiprocessing_context="spawn", prefetch_factor=2)
    return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                      shuffle=False, num_workers=args.workers, drop_last=False,
                      pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                      timeout=120 if args.workers else 0,
                      generator=torch.Generator().manual_seed(args.seed + 12345), **kwargs)


class CZResViTBinary(nn.Module):
    """ResNet50 spatial backbone followed by a CLS-token Transformer classifier."""

    def __init__(self, embedding_dim=256, depth=4, heads=8, dropout=0.1):
        super().__init__()
        backbone = resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.projection = nn.Linear(2048, embedding_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position = nn.Parameter(torch.zeros(1, 50, embedding_dim))
        self.embedding_dropout = nn.Dropout(dropout)
        # Construct each block independently, avoiding identical initial values
        # from cloning one initialized TransformerEncoderLayer four times.
        self.encoders = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim, nhead=heads, dim_feedforward=4 * embedding_dim,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(embedding_dim, 2))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, images):
        features = self.backbone(images)                 # B x 2048 x 7 x 7
        if features.shape[1:] != (2048, 7, 7):
            raise ValueError(f"Expected ResNet output [B,2048,7,7]; got {tuple(features.shape)}")
        tokens = features.flatten(2).transpose(1, 2)     # B x 49 x 2048
        tokens = self.projection(tokens)                # B x 49 x D
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)         # B x 50 x D
        tokens = self.embedding_dropout(tokens + self.position)
        for encoder in self.encoders:
            tokens = encoder(tokens)
        return self.head(self.final_norm(tokens)[:, 0])  # B x 2 logits


def metrics_from_confusion(confusion):
    matrix = np.asarray(confusion, dtype=np.int64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix).astype(np.float64)
    precision = np.divide(true_positive, predicted, out=np.zeros(2), where=predicted != 0)
    recall = np.divide(true_positive, support, out=np.zeros(2), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(2), where=(precision + recall) != 0)
    n = int(support.sum())
    return {"n": n, "accuracy": float(true_positive.sum() / n),
            "macro_f1": float(f1.mean()), "weighted_f1": float(np.dot(f1, support) / n),
            "balanced_accuracy": float(recall.mean()),
            "confusion_matrix": matrix.tolist(), "confusion_order": list(CLASS_NAMES),
            "per_class": {name: {"precision": float(precision[c]), "recall": float(recall[c]),
                                 "f1": float(f1[c]), "support": int(support[c])}
                          for c, name in enumerate(CLASS_NAMES)}}


def probability_metrics(labels, probability):
    # argmax([P(benign), P(malicious)]): ties choose index 0 (benign).
    prediction = (probability > 0.5).astype(np.int64)
    matrix = np.bincount(labels * 2 + prediction, minlength=4).reshape(2, 2)
    result = metrics_from_confusion(matrix)
    if len(np.unique(labels)) == 2:
        result.update(roc_auc_malicious=float(roc_auc_score(labels, probability)),
                      average_precision_malicious=float(average_precision_score(labels, probability)),
                      average_precision_benign=float(average_precision_score(1 - labels, 1 - probability)))
    return result


def class_weights_for(labels, mode):
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training must contain both classes.")
    return (len(labels) / (2.0 * counts) if mode == "balanced" else np.ones(2)).astype(np.float32)


def update_example_count(batch_index, total_examples, batch_size, accumulation_steps):
    """Correct denominator for a full or final, incomplete accumulation group."""
    group_start = (batch_index // accumulation_steps) * accumulation_steps * batch_size
    return min(batch_size * accumulation_steps, total_examples - group_start)


def move_batch(images, labels, device):
    images = images.to(device, dtype=torch.float32, non_blocking=True).div_(255.0)
    labels = labels.to(device, dtype=torch.long, non_blocking=True)
    return images, labels


def train_epoch(model, loader, optimizer, scaler, class_weights, device, amp, args, epoch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, seen, updates, skipped = 0.0, 0, 0, 0
    confusion = np.zeros((2, 2), dtype=np.int64)
    began, last_log = time.perf_counter(), time.perf_counter()
    for batch_index, (images, target, _) in enumerate(loader):
        images, target = move_batch(images, target, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
        # Keep loss/weight arithmetic in float32 even when convolutions use AMP.
        losses = F.cross_entropy(logits.float(), target, reduction="none")
        loss_sum = (losses * class_weights[target]).sum()
        if not torch.isfinite(loss_sum):
            raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, batch {batch_index + 1}.")
        denominator = update_example_count(batch_index, len(loader.dataset), args.batch_size, args.accumulation_steps)
        scaler.scale(loss_sum / denominator).backward()
        boundary = ((batch_index + 1) % args.accumulation_steps == 0 or batch_index + 1 == len(loader))
        if boundary:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not amp and not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite full-precision gradients.")
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                skipped += 1  # GradScaler skipped an overflowing update.
            else:
                updates += 1
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            p = logits.float().softmax(dim=1)[:, 1]
            predicted = (p > 0.5).long()
            confusion += torch.bincount(target * 2 + predicted, minlength=4).reshape(2, 2).cpu().numpy()
        total_loss += float(loss_sum.detach())
        seen += len(target)
        now = time.perf_counter()
        if (batch_index == 0 or (batch_index + 1) % args.log_every == 0
                or batch_index + 1 == len(loader) or now - last_log >= 30):
            speed = seen / max(now - began, 1e-9)
            eta = (len(loader.dataset) - seen) / max(speed, 1e-9) / 60
            print(f"  Train epoch {epoch}: {seen:,}/{len(loader.dataset):,} images; "
                  f"weighted CE={total_loss / seen:.5f}; {speed:.1f} images/s; "
                  f"ETA {eta:.1f} min", flush=True)
            last_log = now
    result = metrics_from_confusion(confusion)
    result.update(weighted_cross_entropy=total_loss / seen, optimizer_updates=updates,
                  amp_skipped_updates=skipped, elapsed_seconds=round(time.perf_counter() - began, 2))
    if updates == 0:
        raise FloatingPointError("No optimizer update succeeded. Check AMP overflow; try --no-amp in a new output folder.")
    return result


@torch.inference_mode()
def validate(model, loader, device, amp, args, epoch):
    model.eval()  # Freeze BatchNorm statistics and disable dropout on validation.
    labels, probabilities, ids = [], [], []
    loss_total, seen, last_log = 0.0, 0, time.perf_counter()
    for batch_index, (images, target, window_ids) in enumerate(loader):
        images, target = move_batch(images, target, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
        loss = F.cross_entropy(logits.float(), target, reduction="sum")
        p = logits.float().softmax(dim=1)[:, 1]
        if not torch.isfinite(loss) or not torch.isfinite(p).all():
            raise FloatingPointError("Nonfinite validation outputs.")
        loss_total += float(loss)
        seen += len(target)
        labels.append(target.cpu().numpy())
        probabilities.append(p.cpu().numpy())
        ids.append(window_ids.numpy())
        now = time.perf_counter()
        if batch_index == 0 or batch_index + 1 == len(loader) or now - last_log >= 30:
            print(f"  Validate epoch {epoch}: {seen:,}/{len(loader.dataset):,} images", flush=True)
            last_log = now
    y = np.concatenate(labels)
    p = np.concatenate(probabilities)
    result = probability_metrics(y, p)
    result["cross_entropy"] = loss_total / seen
    return result, {"window_ids": torch.from_numpy(np.concatenate(ids)),
                    "labels": torch.from_numpy(y), "probability_malicious": torch.from_numpy(p)}


def cpu_copy(value):
    """Snapshot tensors so later optimizer updates cannot mutate a checkpoint."""
    if torch.is_tensor(value):
        return value.detach().to("cpu").clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(v) for v in value)
    return value


def rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


def export_progress(out, checkpoint):
    """The atomic last checkpoint is authoritative; exports can be rebuilt."""
    history = checkpoint["history"]
    atomic_json(out / "history.json", history)
    temporary = out / "metrics.csv.partial"
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(["epoch", "learning_rate", "train_weighted_ce", "train_macro_f1_online",
                         "validation_ce", "validation_accuracy", "validation_macro_f1",
                         "validation_balanced_accuracy", "validation_benign_recall", "best_so_far"])
        for row in history:
            t, v = row["training"], row["validation"]
            writer.writerow([row["epoch"], row["learning_rate"], t["weighted_cross_entropy"], t["macro_f1"],
                             v["cross_entropy"], v["accuracy"], v["macro_f1"], v["balanced_accuracy"],
                             v["per_class"]["benign"]["recall"], row["selected_as_best"]])
    temporary.replace(out / "metrics.csv")
    if checkpoint["best_model_state"] is not None:
        # Export only if needed. A crash during an export is harmless: resume
        # regenerates it from the committed checkpoint before further training.
        marker = out / "best_export.json"
        expected = {"epoch": checkpoint["best_epoch"], "config_hash": checkpoint["config_hash"]}
        current = read_json(marker) if marker.exists() else None
        if (current != expected or not (out / "best_model.pt").exists()
                or not (out / "best_validation_predictions.csv").exists()):
            atomic_torch_save(out / "best_model.pt", {
                "model_config": checkpoint["configuration"]["model"],
                "model_state": checkpoint["best_model_state"],
                "epoch": checkpoint["best_epoch"], "validation_metrics": checkpoint["best_metrics"],
                "configuration": checkpoint["configuration"], "config_hash": checkpoint["config_hash"],
                "class_names": list(CLASS_NAMES), "input_conversion": "RGB uint8 -> float32 / 255",
            })
            temporary = out / "best_validation_predictions.csv.partial"
            values = checkpoint["best_validation_predictions"]
            with temporary.open("w", newline="", encoding="utf-8") as destination:
                writer = csv.writer(destination)
                writer.writerow(["window_id", "target", "probability_malicious", "predicted_target"])
                for window, label, probability in zip(values["window_ids"].tolist(), values["labels"].tolist(),
                                                        values["probability_malicious"].tolist()):
                    writer.writerow([window, label, probability, int(probability > 0.5)])
            temporary.replace(out / "best_validation_predictions.csv")
            atomic_json(marker, expected)


def runtime_info(device, amp, args):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "numpy": np.__version__,
            "pillow": PIL.__version__, "scikit_learn": sklearn.__version__,
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda_runtime": torch.version.cuda, "amp": amp, "workers": args.workers,
            "torch_threads": torch.get_num_threads()}


def choose_device(args):
    if args.device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if args.smoke_test and args.device == "auto":
        return torch.device("cpu")
    raise RuntimeError(
        "CUDA is unavailable in this Python environment. Install the CUDA-enabled "
        "torch/torchvision build from the command at the top of this file and check "
        "your NVIDIA driver. For intentional CPU use, pass --device cpu."
    )


def smoke_indices(labels, splits, split, maximum_per_class):
    # Force both classes into the diagnostic sample. It is not a benchmark.
    indices = [np.flatnonzero((splits == split) & (labels == c))[:maximum_per_class] for c in (0, 1)]
    return np.concatenate(indices)


def run_smoke(root, summary, labels, splits, model, optimizer, scaler, weights, device, amp, args):
    print("\nSMOKE TEST: small train/validation sample; no benchmark or model files are saved.", flush=True)
    train_ids = smoke_indices(labels, splits, 0, 8)
    val_ids = smoke_indices(labels, splits, 1, 4)
    train_data = ContourZipDataset(root, train_ids, labels, summary["shard_size"])
    val_data = ContourZipDataset(root, val_ids, labels, summary["shard_size"])
    # Probe gradients/AMP with unweighted CE, so deliberately balancing this tiny
    # diagnostic sample does not amplify the full-data minority class weights.
    # The real run below still uses the training-only weights it reports.
    probe_weights = torch.ones_like(weights)
    before = model.head[-1].weight.detach().cpu().clone()
    try:
        train_epoch(model, make_loader(train_data, args, device), optimizer, scaler,
                    probe_weights, device, amp, args, 0)
        metrics, _ = validate(model, make_loader(val_data, args, device), device, amp, args, 0)
        if torch.equal(before, model.head[-1].weight.detach().cpu()):
            raise RuntimeError("Smoke test did not change the classifier weights.")
        print("SMOKE TEST PASSED", flush=True)
        print(f"Read {len(train_ids)} training and {metrics['n']} validation images; output shape [B,2].")
        print("Forward pass, loss, backward pass, optimizer update and validation succeeded.")
        print("Test image pixels were not read. Start the real run without --smoke-test.")
    finally:
        train_data.close()
        val_data.close()


def run(args):
    root = args.input_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    print(f"Input: {root}", flush=True)
    device = choose_device(args)
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
    amp = device.type == "cuda" and not args.no_amp
    environment = runtime_info(device, amp, args)
    print(f"Device: {environment['gpu'] or str(device)}; PyTorch {environment['torch']}; AMP={amp}", flush=True)
    summary, labels, splits, fingerprint = inspect_inputs(root)
    train_ids, val_ids = np.flatnonzero(splits == 0), np.flatnonzero(splits == 1)
    weights_np = class_weights_for(labels[train_ids], args.class_weighting)
    weights = torch.as_tensor(weights_np, device=device)
    print(f"Training loss weights: {dict(zip(CLASS_NAMES, weights_np.astype(float).tolist()))}", flush=True)
    baseline = probability_metrics(labels[val_ids], np.ones(len(val_ids)))
    print(f"Always-malicious VALIDATION baseline: accuracy={baseline['accuracy']:.6f}; "
          f"macro F1={baseline['macro_f1']:.6f}; benign recall=0.000000", flush=True)
    model_config = {"embedding_dim": args.embedding_dim, "depth": args.depth,
                    "heads": args.heads, "dropout": args.dropout}
    configuration = {"script_version": VERSION, "input_dir": str(root), "data_fingerprint": fingerprint,
                     "step05_configuration_hash": summary["configuration_hash"],
                     "feature_order": summary["feature_order"], "validation_fold": summary["validation_fold"],
                     "model": model_config, "backbone": "torchvision ResNet50, weights=None, stages 3/4/6/3",
                     "class_names": list(CLASS_NAMES), "class_weighting": args.class_weighting,
                     "class_weights": weights_np.astype(float).tolist(), "seed": args.seed,
                     "batch_size": args.batch_size, "accumulation_steps": args.accumulation_steps,
                     "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                     "amp_requested": not args.no_amp, "input_conversion": "RGB uint8 -> float32 / 255",
                     "checkpoint_selection": "validation macro F1; exact ties -> lower validation CE",
                     "early_stopping_patience": args.patience,
                     "scheduler": "ReduceLROnPlateau(mode=max, factor=0.5, patience=2, threshold=1e-6, threshold_mode=abs)",
                     "loss_reduction": "sum(weight[target] * CE_per_example) / examples_per_update",
                     "window_class_counts": class_counts(labels, splits)}
    config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    model = CZResViTBinary(**model_config).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {parameters:,}; all trainable.", flush=True)
    print("Shapes: [B,3,224,224] -> [B,2048,7,7] -> 49 spatial tokens + CLS -> [B,2].", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)
    if args.smoke_test:
        run_smoke(root, summary, labels, splits, model, optimizer, scaler, weights, device, amp, args)
        return
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, threshold=1e-6, threshold_mode="abs")
    out.mkdir(parents=True, exist_ok=True)
    lock = sqlite3.connect(out / "run_lock.sqlite3", timeout=0)
    train_data = val_data = None
    try:
        lock.execute("CREATE TABLE IF NOT EXISTS writer_lock (id INTEGER PRIMARY KEY)")
        lock.execute("BEGIN EXCLUSIVE")
        config_path = out / "training_config.json"
        if config_path.exists():
            if read_json(config_path) != configuration:
                raise ValueError("Training settings or input files changed. Use a NEW --output-dir for that experiment.")
        elif any(p.name != "run_lock.sqlite3" for p in out.iterdir()):
            raise ValueError("Output folder contains unrelated files. Choose a new --output-dir.")
        else:
            atomic_json(config_path, configuration)
        checkpoint_path = out / "last_checkpoint.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint["config_hash"] != config_hash:
                raise ValueError("Checkpoint configuration does not match this run.")
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            if amp and checkpoint["scaler_state"]:
                scaler.load_state_dict(checkpoint["scaler_state"])
            restore_rng(checkpoint["rng_state"])
            print(f"Resuming after completed epoch {checkpoint['epoch']}.", flush=True)
        else:
            checkpoint = {"configuration": configuration, "config_hash": config_hash, "epoch": 0,
                          "model_state": cpu_copy(model.state_dict()),
                          "optimizer_state": cpu_copy(optimizer.state_dict()), "scheduler_state": scheduler.state_dict(),
                          "scaler_state": scaler.state_dict(), "rng_state": rng_state(), "history": [],
                          "best_epoch": 0, "best_metrics": None, "best_model_state": None,
                          "best_validation_predictions": None, "epochs_without_improvement": 0,
                          "initial_environment": environment}
            atomic_torch_save(checkpoint_path, checkpoint)
        export_progress(out, checkpoint)
        atomic_json(out / "runtime_latest.json", environment)
        train_data = ContourZipDataset(root, train_ids, labels, summary["shard_size"])
        val_data = ContourZipDataset(root, val_ids, labels, summary["shard_size"])
        sampler = EpochShuffleSampler(len(train_data), args.seed)
        train_loader = make_loader(train_data, args, device, sampler)
        val_loader = make_loader(val_data, args, device)
        for epoch in range(checkpoint["epoch"] + 1, args.epochs + 1):
            if checkpoint["epochs_without_improvement"] >= args.patience:
                print("Early-stopping criterion was reached; keeping the selected checkpoint.", flush=True)
                break
            sampler.set_epoch(epoch)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            print(f"\nEpoch {epoch}/{args.epochs}; learning rate={learning_rate:g}", flush=True)
            training = train_epoch(model, train_loader, optimizer, scaler, weights, device, amp, args, epoch)
            validation, predictions = validate(model, val_loader, device, amp, args, epoch)
            best = checkpoint["best_metrics"]
            improved = (best is None or validation["macro_f1"] > best["macro_f1"]
                        or (validation["macro_f1"] == best["macro_f1"] and validation["cross_entropy"] < best["cross_entropy"]))
            current_state = cpu_copy(model.state_dict())
            if improved:
                checkpoint.update(best_epoch=epoch, best_metrics=validation, best_model_state=current_state,
                                  best_validation_predictions=predictions, epochs_without_improvement=0)
            else:
                checkpoint["epochs_without_improvement"] += 1
            scheduler.step(validation["macro_f1"])
            checkpoint["history"].append({"epoch": epoch, "learning_rate": learning_rate,
                                          "training": training, "validation": validation,
                                          "selected_as_best": improved, "environment": environment})
            checkpoint.update(epoch=epoch, model_state=current_state,
                              optimizer_state=cpu_copy(optimizer.state_dict()), scheduler_state=scheduler.state_dict(),
                              scaler_state=scaler.state_dict(), rng_state=rng_state())
            atomic_torch_save(checkpoint_path, checkpoint)
            export_progress(out, checkpoint)
            print(f"EPOCH {epoch} COMPLETE: validation accuracy={validation['accuracy']:.6f}; "
                  f"macro F1={validation['macro_f1']:.6f}; balanced accuracy={validation['balanced_accuracy']:.6f}", flush=True)
            for name in CLASS_NAMES:
                c = validation["per_class"][name]
                print(f"  {name}: precision={c['precision']:.6f}; recall={c['recall']:.6f}; "
                      f"F1={c['f1']:.6f}; support={c['support']:,}", flush=True)
            print(f"  Checkpoint saved. Best epoch: {checkpoint['best_epoch']}; "
                  f"successful updates: {training['optimizer_updates']}; AMP-skipped updates: {training['amp_skipped_updates']}", flush=True)
        best = checkpoint["best_metrics"]
        report = {"step": "06_binary_resnet50_transformer_training", "output_dir": str(out),
                  "epochs_completed": checkpoint["epoch"], "requested_total_epochs": args.epochs,
                  "early_stopped": checkpoint["epochs_without_improvement"] >= args.patience,
                  "best_epoch": checkpoint["best_epoch"], "best_validation_metrics": best,
                  "always_malicious_validation_baseline": baseline, "parameters": parameters,
                  "model_config": model_config, "class_weighting": args.class_weighting,
                  "window_class_counts": class_counts(labels, splits),
                  "test_evaluated": False, "zero_day_evaluated": False,
                  "validation_fold": summary["validation_fold"], "five_fold_experiment_complete": False,
                  "step05_image_diagnostics": summary.get("image_diagnostics", {}),
                  "step05_correlation_diagnostics": summary.get("correlations", {}),
                  "configuration_hash": config_hash,
                  "notes": [
                      "All training images are visited once per completed epoch; no sampling or augmentation.",
                      ("Class-balanced loss is an explicitly enabled adaptation for dataset7's imbalance."
                       if args.class_weighting == "balanced" else "This baseline uses unweighted cross-entropy."),
                      "Reported model metrics are validation metrics used for checkpoint selection.",
                      "Only 24 benign validation images exist in the full dataset7 run; precision of minority-class estimates is limited.",
                      "The architecture's Transformer settings, optimizer settings and initialization are reconstruction choices.",
                      "AMP is not quantization; the paper's quantization procedure remains unresolved.",
                  ]}
        atomic_json(out / "step06_summary.json", report)
        print("\nSTEP 6 COMPLETE", flush=True)
        print(f"Epochs completed: {checkpoint['epoch']}; best validation epoch: {checkpoint['best_epoch']}")
        if best is not None:
            print(f"Best validation macro F1: {best['macro_f1']:.6f}")
            print(f"Best model: {out / 'best_model.pt'}")
        print(f"Resume checkpoint: {checkpoint_path}")
        print(f"Summary: {out / 'step06_summary.json'}")
        print("Test evaluation and zero-day evaluation have not been performed.")
    finally:
        if train_data is not None:
            train_data.close()
        if val_data is not None:
            val_data.close()
        lock.rollback()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description="Step 6: binary CZ-ResViT reconstruction with resumable training.")
    parser.add_argument("--input-dir", type=Path, default=Path("prepared/dataset7_step05"))
    parser.add_argument("--output-dir", type=Path, default=Path("trained/dataset7_step06"))
    parser.add_argument("--epochs", type=int, default=20, help="Total target epochs, including those already saved.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weighting", choices=("balanced", "none"), default="none")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    positive = (args.epochs, args.batch_size, args.accumulation_steps, args.embedding_dim,
                args.depth, args.heads, args.torch_threads, args.patience, args.log_every)
    if (any(value < 1 for value in positive) or args.workers < 0 or args.seed < 0
            or args.embedding_dim % args.heads != 0 or not 0 <= args.dropout < 1
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        parser.error("Use positive sizes/rates, nonnegative workers/weight-decay/seed, dropout in [0,1), and embedding-dim divisible by heads.")
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nInterrupted. The last completed epoch is retained. Rerun the same training command to resume.", file=sys.stderr)
        return 130
    except torch.cuda.OutOfMemoryError:
        print("\nCUDA memory exhausted. Try --batch-size 4 --accumulation-steps 8 with a NEW --output-dir.", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"\nSTEP 6 FAILED: {error}", file=sys.stderr)
        if "DataLoader" in str(error) or "worker" in str(error).lower():
            print("For a loader worker failure, rerun with --workers 0. Completed epochs are retained.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
