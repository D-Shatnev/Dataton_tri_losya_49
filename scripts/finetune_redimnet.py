"""
Fine-tuning script for ReDimNet-b6 on organizer's training data.

3-stage training strategy:
  Stage 1 (head_warmup):    freeze backbone, train AAM-Softmax head only
  Stage 2 (full_finetune):  unfreeze backbone, different LR for backbone vs head
  Stage 3 (large_margin):   large margin (0.5), long crops (6s), no augmentations

Usage:
    python scripts/finetune_redimnet.py --config configs/train/finetune_redimnet.json

Saves only backbone weights (compatible with torch.hub.load + load_state_dict).
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import torchaudio
except ImportError as e:
    raise ImportError("torchaudio is required: pip install torchaudio") from e

from datetime import timedelta

from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs

from dataton_tri_losya_49.pipeline.components.evaluators.precision_at_k import PrecisionAtKEvaluator
from dataton_tri_losya_49.pipeline.components.indexers.faiss_ip import FaissInnerProductIndexer

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# AAM-Softmax Loss
# ---------------------------------------------------------------------------


class AAMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax (ArcFace-style) loss.

    Args:
        embed_dim:    Dimensionality of input embeddings.
        num_classes:  Number of speaker classes.
        scale:        Logit scale factor (s).
        margin:       Angular margin (m) in radians.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.2,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def set_margin(self, margin: float) -> None:
        """Update margin in-place (used between stages)."""
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute AAM-Softmax loss.

        Args:
            embeddings: [B, embed_dim] — raw (unnormalized) embeddings from backbone.
            labels:     [B] — integer class indices.

        Returns:
            Scalar cross-entropy loss.
        """
        # Normalize embeddings and weight matrix
        emb_norm = F.normalize(embeddings, p=2, dim=1)  # [B, D]
        w_norm = F.normalize(self.weight, p=2, dim=1)  # [C, D]

        # cos(θ) for all classes
        cos_theta = F.linear(emb_norm, w_norm)  # [B, C]
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if self.margin == 0.0:
            # No margin — plain cosine softmax
            logits = self.scale * cos_theta
        else:
            # cos(θ + m) for target class
            theta = torch.acos(cos_theta)
            # Build one-hot mask for target positions
            one_hot = torch.zeros_like(cos_theta)
            one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

            # Apply margin only to target class angles
            target_logits = torch.cos(theta + self.margin)
            logits = self.scale * (one_hot * target_logits + (1.0 - one_hot) * cos_theta)

        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Augmentation (per-sample time masking + additive noise)
# ---------------------------------------------------------------------------


class SimpleAugment(nn.Module):
    """
    Lightweight waveform augmentation for speaker recognition training.

    Applies four independent augmentations:
    1. **Speed perturbation** — with 50% probability resample to 0.9x or 1.1x speed,
       then crop/pad back to original length. Effective domain augmentation for
       speaker recognition (changes pitch + tempo).
    2. **Volume perturbation** — with 50% probability apply random gain ±6 dB.
    3. **Per-sample time masking** — zeros out 0–5 random contiguous segments
       of 10–30 ms each. Applied independently per sample in the batch.
    4. **Additive Gaussian noise** — with 50% probability adds white noise
       at a random SNR between 5 and 25 dB.

    No STFT/ISTFT roundtrip — avoids reconstruction artifacts and GPU overhead.

    Args:
        sample_rate:     Audio sample rate (needed for speed perturbation).
        max_time_masks:  Maximum number of time-mask regions per sample (default 5).
        speed_prob:      Probability of applying speed perturbation (default 0.5).
        volume_prob:     Probability of applying volume perturbation (default 0.5).
        noise_prob:      Probability of applying additive noise (default 0.5).
        snr_min_db:      Minimum SNR in dB for additive noise (default 5).
        snr_max_db:      Maximum SNR in dB for additive noise (default 25).
        gain_min_db:     Minimum gain in dB for volume perturbation (default -6).
        gain_max_db:     Maximum gain in dB for volume perturbation (default 6).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_time_masks: int = 5,
        speed_prob: float = 0.5,
        volume_prob: float = 0.5,
        noise_prob: float = 0.5,
        snr_min_db: float = 5.0,
        snr_max_db: float = 25.0,
        gain_min_db: float = -6.0,
        gain_max_db: float = 6.0,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.max_time_masks = max_time_masks
        self.speed_prob = speed_prob
        self.volume_prob = volume_prob
        self.noise_prob = noise_prob
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db
        self.gain_min_db = gain_min_db
        self.gain_max_db = gain_max_db

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to a batch of waveforms.

        Args:
            waveform: [B, T] float32 waveforms.

        Returns:
            Augmented waveforms [B, T] of the same shape.
        """
        if not self.training:
            return waveform

        x = waveform.clone()
        B, T = x.shape

        # --- Speed perturbation (per-batch) ---
        # Changes pitch + tempo — one of the most effective augmentations for SV
        if random.random() < self.speed_prob:
            speed = random.choice([0.9, 1.1])
            # torchaudio.functional.speed returns (waveform, new_sample_rate)
            x_speed, _ = torchaudio.functional.speed(x, orig_freq=self.sample_rate, factor=speed)
            # Crop or pad back to original length T
            T_new = x_speed.shape[1]
            if T_new >= T:
                x = x_speed[:, :T]
            else:
                x = F.pad(x_speed, (0, T - T_new))

        # --- Volume perturbation (per-batch) ---
        if random.random() < self.volume_prob:
            gain_db = random.uniform(self.gain_min_db, self.gain_max_db)
            x = x * (10.0 ** (gain_db / 20.0))

        # --- Per-sample time masking ---
        if self.max_time_masks > 0:
            for b in range(B):
                num_masks = random.randint(0, self.max_time_masks)
                for _ in range(num_masks):
                    # 10–30 ms at 16 kHz = 160–480 samples
                    mask_len = random.randint(160, 480)
                    start = random.randint(0, max(0, T - mask_len))
                    x[b, start : start + mask_len] = 0.0

        # --- Additive Gaussian noise (per-batch, random SNR) ---
        if random.random() < self.noise_prob:
            snr_db = random.uniform(self.snr_min_db, self.snr_max_db)
            signal_power = (x**2).mean(dim=1, keepdim=True).clamp(min=1e-9)
            noise_power = signal_power / (10.0 ** (snr_db / 10.0))
            noise = torch.randn_like(x) * noise_power.sqrt()
            x = x + noise

        return x


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SpeakerDataset(Dataset):
    """
    PyTorch Dataset for speaker verification training.

    Loads FLAC (or any torchaudio-supported) audio files, applies random crop
    or zero-padding to a fixed length, and returns (waveform, label) pairs.

    Args:
        filepaths:    List of absolute or relative audio file paths.
        labels:       Integer speaker labels aligned with filepaths.
        crop_samples: Target waveform length in samples.
        augment:      SpecAugment module or None.
        data_root:    Root directory prepended to relative paths.
        sample_rate:  Expected sample rate (files are resampled if needed).
    """

    def __init__(
        self,
        filepaths: list[str],
        labels: list[int],
        crop_samples: int,
        augment: SimpleAugment | None,
        data_root: Path,
        sample_rate: int = 16000,
    ) -> None:
        self.filepaths = filepaths
        self.labels = labels
        self.crop_samples = crop_samples
        self.augment = augment
        self.data_root = data_root
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        try:
            fp = self.filepaths[idx]
            label = self.labels[idx]

            path = Path(fp)
            if not path.is_absolute():
                path = self.data_root / path

            waveform, sr = torchaudio.load(str(path))  # [C, T]
            waveform = waveform.mean(dim=0)  # mono [T]

            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

            waveform = self._crop_or_pad(waveform)  # [crop_samples]

            if self.augment is not None:
                # augment expects [B, T]; add/remove batch dim
                waveform = self.augment(waveform.unsqueeze(0)).squeeze(0)

            return waveform, label

        except Exception:
            # Corrupted or missing file — fallback to a random valid sample
            fallback_idx = random.randint(0, len(self) - 1)
            if fallback_idx == idx:
                fallback_idx = (idx + 1) % len(self)
            return self.__getitem__(fallback_idx)

    def _crop_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        """Random crop if longer, zero-pad if shorter."""
        T = waveform.shape[0]
        target = self.crop_samples

        if T >= target:
            start = random.randint(0, T - target)
            return waveform[start : start + target]
        else:
            pad = target - T
            return F.pad(waveform, (0, pad))


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_embeddings(
    backbone: nn.Module,
    filepaths: list[str],
    data_root: Path,
    sample_rate: int,
    device: torch.device,
    val_chunk_seconds: float | None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> np.ndarray:
    """
    Extract L2-normalized embeddings for a list of audio files.

    Args:
        backbone:          ReDimNet model in eval mode.
        filepaths:         List of audio file paths.
        data_root:         Root directory for relative paths.
        sample_rate:       Target sample rate.
        device:            Compute device.
        val_chunk_seconds: If None — use full audio; if float — crop to that length.
        batch_size:        Inference batch size.
        num_workers:       DataLoader workers.

    Returns:
        float32 numpy array [N, embed_dim], L2-normalized.
    """
    backbone.eval()

    if val_chunk_seconds is not None:
        crop_samples = int(val_chunk_seconds * sample_rate)
    else:
        crop_samples = None  # variable length — handled per-sample

    all_embs = []

    # Process in batches; for variable-length we use batch_size=1
    if crop_samples is None:
        # Variable length: process one by one
        for fp in tqdm(filepaths, desc="Val embeddings (full-length)", leave=False):
            path = Path(fp)
            if not path.is_absolute():
                path = data_root / path
            wav, sr = torchaudio.load(str(path))
            wav = wav.mean(dim=0)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            wav = wav.unsqueeze(0).to(device)  # [1, T]
            emb = backbone(wav)  # [1, D]
            emb = F.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu().float().numpy())
        return np.concatenate(all_embs, axis=0)
    else:
        # Fixed length: use DataLoader for efficiency
        dummy_labels = [0] * len(filepaths)
        ds = SpeakerDataset(
            filepaths=filepaths,
            labels=dummy_labels,
            crop_samples=crop_samples,
            augment=None,
            data_root=data_root,
            sample_rate=sample_rate,
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        for wavs, _ in loader:
            wavs = wavs.to(device)
            emb = backbone(wavs)
            emb = F.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu().float().numpy())
        return np.concatenate(all_embs, axis=0)


def compute_precision_at_k_faiss(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ks: list[int],
) -> dict[str, float]:
    """
    Compute Precision@K using FAISS for efficient nearest-neighbor search.

    Uses FaissInnerProductIndexer (cosine similarity via L2-normalized IP)
    and PrecisionAtKEvaluator from the project pipeline.

    Args:
        embeddings: [N, D] float32 array (will be L2-normalized inside FAISS indexer).
        labels:     [N] integer speaker labels.
        ks:         List of K values.

    Returns:
        Dict {"precision@K": value} (keys from PrecisionAtKEvaluator).
    """
    k_max = max(ks)
    indexer = FaissInnerProductIndexer()
    neighbors = indexer.neighbors(embeddings, topk=k_max)
    evaluator = PrecisionAtKEvaluator()
    raw = evaluator.evaluate(neighbors=neighbors, labels=labels, ks=ks)
    # Remap keys from "precision@K" → "P@K" for consistent log output
    return {f"P@{k}": raw[f"precision@{k}"] for k in ks}


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def build_optimizer(
    backbone: nn.Module,
    head: AAMSoftmax,
    stage_cfg: dict,
) -> torch.optim.Optimizer:
    """
    Build optimizer with (optionally) separate LR groups for backbone and head.

    Stage 1 (freeze_backbone=True): only head parameters, SGD with lr=stage_cfg['lr'].
    Stage 2/3 (freeze_backbone=False): two param groups with lr_backbone / lr_head.
    """
    if stage_cfg.get("freeze_backbone", False):
        # Only head parameters
        optimizer = torch.optim.SGD(
            head.parameters(),
            lr=stage_cfg["lr"],
            momentum=0.9,
            weight_decay=stage_cfg.get("weight_decay", 1e-4),
        )
    else:
        lr_backbone = stage_cfg["lr_backbone"]
        lr_head = stage_cfg["lr_head"]
        wd = stage_cfg.get("weight_decay", 1e-5)
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone.parameters(), "lr": lr_backbone},
                {"params": head.parameters(), "lr": lr_head},
            ],
            weight_decay=wd,
        )
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    stage_cfg: dict,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    """Build cosine LR scheduler for full_finetune and large_margin stages."""
    if stage_cfg.get("freeze_backbone", False):
        return None  # Stage 1: no scheduler
    total_steps = stage_cfg["epochs"] * steps_per_epoch
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-7,
    )


def train_one_epoch(
    backbone: nn.Module,
    head: AAMSoftmax,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    freeze_backbone: bool,
    accelerator: Accelerator,
    raw_head: AAMSoftmax,
) -> tuple[float, float]:
    """
    Run one training epoch.

    Returns:
        (avg_loss, avg_accuracy) over the epoch.
    """
    backbone.train() if not freeze_backbone else backbone.eval()
    head.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Training", leave=False, dynamic_ncols=True, disable=not accelerator.is_main_process)
    for wavs, labels in pbar:
        # accelerate.prepare() already moved data to the right device,
        # but DataLoader items still need explicit placement when not using
        # accelerate's DataLoader wrapping for variable-length batches.
        wavs = wavs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if freeze_backbone:
            with torch.no_grad():
                emb = backbone(wavs)
        else:
            emb = backbone(wavs)

        loss = head(emb, labels)
        accelerator.backward(loss)

        # Gradient clipping for stability
        if not freeze_backbone:
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Accuracy (no-margin cosine similarity argmax)
        # Use raw_head (unwrapped from DDP) to access .weight directly
        with torch.no_grad():
            emb_norm = F.normalize(emb.detach(), p=2, dim=1)
            w_norm = F.normalize(raw_head.weight.detach(), p=2, dim=1)
            cos_sim = F.linear(emb_norm, w_norm)
            preds = cos_sim.argmax(dim=1)
            total_correct += (preds == labels).sum().item()

        total_loss += loss.item() * wavs.size(0)
        total_samples += wavs.size(0)

        # Update progress bar with running stats
        running_loss = total_loss / max(total_samples, 1)
        running_acc = total_correct / max(total_samples, 1)
        pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def run_training(config: dict) -> None:
    set_seed(42)

    # --- Paths ---
    exp_dir = Path(config["exp_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(config["data_base_dir"])
    train_csv = Path(config["train_csv"])
    best_ckpt_path = exp_dir / "best_backbone.pt"

    # --- Accelerator (multi-GPU via `accelerate launch`) ---
    # find_unused_parameters=True is required for Stage 1 where backbone is frozen
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=24))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, init_kwargs])
    device = accelerator.device
    accelerator.print(f"Using device: {device}  |  num_processes={accelerator.num_processes}")

    # --- Load CSV and split ---
    accelerator.print(f"Loading CSV: {train_csv}")
    df = pd.read_csv(train_csv)
    assert "filepath" in df.columns, "CSV must have 'filepath' column"
    assert "speaker_id" in df.columns, "CSV must have 'speaker_id' column"

    # Encode speaker_id → integer
    speakers = sorted(df["speaker_id"].astype(str).unique())
    spk2id = {s: i for i, s in enumerate(speakers)}
    num_classes = len(speakers)
    accelerator.print(f"Total speakers: {num_classes}, total files: {len(df)}")

    df["label"] = df["speaker_id"].astype(str).map(spk2id)

    filepaths = df["filepath"].tolist()
    labels = df["label"].tolist()

    # Stratified split — speakers with only 1 sample go to train only
    # (train_test_split with stratify requires ≥2 samples per class)
    split_ratio = config.get("split_ratio", 0.9)
    label_counts = Counter(labels)
    single_mask = [label_counts[lb] == 1 for lb in labels]

    # Single-sample speakers → train only
    train_fps: list[str] = [fp for fp, s in zip(filepaths, single_mask, strict=False) if s]
    train_lbs: list[int] = [lb for lb, s in zip(labels, single_mask, strict=False) if s]

    # Multi-sample speakers → stratified split
    multi_fps = [fp for fp, s in zip(filepaths, single_mask, strict=False) if not s]
    multi_lbs = [lb for lb, s in zip(labels, single_mask, strict=False) if not s]

    n_multi_classes = len(set(multi_lbs))
    val_size_multi = max(1, int(len(multi_fps) * (1.0 - split_ratio)))
    use_stratify = val_size_multi >= n_multi_classes

    tr_fps, val_fps, tr_lbs, val_lbs = train_test_split(
        multi_fps,
        multi_lbs,
        train_size=split_ratio,
        stratify=multi_lbs if use_stratify else None,
        random_state=42,
    )
    train_fps.extend(tr_fps)
    train_lbs.extend(tr_lbs)

    n_single = sum(single_mask)
    accelerator.print(f"Train: {len(train_fps)} (incl. {n_single} single-sample speakers), " f"Val: {len(val_fps)}")

    # --- Load backbone ---
    hub_repo = config["hub_repo"]
    model_name = config["model_name"]
    train_type = config["train_type"]
    dataset = config["dataset"]
    sample_rate = config.get("sample_rate", 16000)
    embed_dim = config.get("embed_dim", 192)

    accelerator.print(f"Loading backbone: {hub_repo} / {model_name} / {train_type} / {dataset}")
    torch.hub.set_dir(str(exp_dir.parent.parent / "models" / "torch_hub"))
    backbone = torch.hub.load(
        hub_repo,
        "ReDimNet",
        model_name=model_name,
        train_type=train_type,
        dataset=dataset,
        trust_repo=True,
    )
    backbone = backbone.to(device)
    accelerator.print("Backbone loaded.")

    # --- AAM-Softmax head ---
    aam_scale = config.get("aam_scale", 30.0)
    head = AAMSoftmax(
        embed_dim=embed_dim,
        num_classes=num_classes,
        scale=aam_scale,
        margin=0.0,  # will be set per stage
    ).to(device)

    # --- Augmentation ---
    augment = SimpleAugment(
        sample_rate=sample_rate,
        max_time_masks=5,
        speed_prob=0.5,
        volume_prob=0.5,
        noise_prob=0.5,
        snr_min_db=5.0,
        snr_max_db=25.0,
        gain_min_db=-6.0,
        gain_max_db=6.0,
    ).to(device)

    val_ks = config.get("val_ks", [1, 5, 10])
    val_chunk_seconds = config.get("val_chunk_seconds", None)
    num_workers = config.get("num_workers", 4)

    best_p10 = -1.0
    stages = config["stages"]
    total_stages = len(stages)

    # Prepare backbone and head once — avoids double DDP wrapping across stages
    backbone, head = accelerator.prepare(backbone, head)

    # References to previous-stage objects for explicit cleanup
    _prev_optimizer = None
    _prev_loader = None
    _prev_scheduler = None

    for stage_idx, stage_cfg in enumerate(stages):
        # --- Free GPU memory from previous stage ---
        # Nullify local variables first so gc.collect() can reclaim them
        if stage_idx > 0:
            optimizer = None
            train_loader = None
            scheduler = None
        if _prev_optimizer is not None:
            del _prev_optimizer
            _prev_optimizer = None
        if _prev_loader is not None:
            del _prev_loader
            _prev_loader = None
        if _prev_scheduler is not None:
            del _prev_scheduler
            _prev_scheduler = None
        gc.collect()
        torch.cuda.empty_cache()

        stage_name = stage_cfg["name"]
        num_epochs = stage_cfg["epochs"]
        freeze_backbone = stage_cfg.get("freeze_backbone", False)
        batch_size = stage_cfg["batch_size"]
        crop_seconds = stage_cfg["crop_seconds"]
        margin = stage_cfg.get("margin", 0.2)
        use_aug = stage_cfg.get("use_augmentations", True)

        crop_samples = int(crop_seconds * sample_rate)

        accelerator.print(f"\n{'='*70}")
        accelerator.print(f"Stage {stage_idx+1}/{total_stages}: [{stage_name}]")
        accelerator.print(
            f"  epochs={num_epochs}, freeze_backbone={freeze_backbone}, "
            f"batch={batch_size}, crop={crop_seconds}s, margin={margin}, aug={use_aug}"
        )
        accelerator.print(f"{'='*70}")

        # Update margin via unwrapped head (DDP wrapper has no set_margin)
        accelerator.unwrap_model(head).set_margin(margin)

        # Freeze / unfreeze backbone via unwrapped model
        for p in accelerator.unwrap_model(backbone).parameters():
            p.requires_grad = not freeze_backbone

        # Build dataset & loader
        train_ds = SpeakerDataset(
            filepaths=train_fps,
            labels=train_lbs,
            crop_samples=crop_samples,
            augment=augment if use_aug else None,
            data_root=data_root,
            sample_rate=sample_rate,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
        )

        # Build optimizer & scheduler (backbone/head already DDP-wrapped)
        optimizer = build_optimizer(backbone, head, stage_cfg)
        scheduler = build_scheduler(optimizer, stage_cfg, len(train_loader))

        # Prepare only optimizer and loader each stage (backbone/head prepared once above)
        optimizer, train_loader = accelerator.prepare(optimizer, train_loader)
        if scheduler is not None:
            scheduler = accelerator.prepare(scheduler)

        # Keep references for cleanup at next stage start
        _prev_optimizer = optimizer
        _prev_loader = train_loader
        _prev_scheduler = scheduler

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()

            avg_loss, avg_acc = train_one_epoch(
                backbone=backbone,
                head=head,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                freeze_backbone=freeze_backbone,
                accelerator=accelerator,
                raw_head=accelerator.unwrap_model(head),
            )

            # Save after every training epoch (BEFORE validation, in case val crashes)
            if accelerator.is_main_process:
                epoch_ckpt = exp_dir / f"backbone_stage{stage_idx+1}_epoch{epoch}.pt"
                torch.save(accelerator.unwrap_model(backbone).state_dict(), epoch_ckpt)

            # --- Validation (main process only) ---
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_backbone = accelerator.unwrap_model(backbone)
                val_embs = extract_embeddings(
                    backbone=unwrapped_backbone,
                    filepaths=val_fps,
                    data_root=data_root,
                    sample_rate=sample_rate,
                    device=device,
                    val_chunk_seconds=val_chunk_seconds,
                    batch_size=batch_size * 2,
                    num_workers=num_workers,
                )
                val_labels_arr = np.array(val_lbs, dtype=np.int64)
                metrics = compute_precision_at_k_faiss(val_embs, val_labels_arr, val_ks)

                elapsed = int(time.time() - t0)
                p10 = metrics.get("P@10", 0.0)

                # Format metrics string
                metrics_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())

                accelerator.print(
                    f"Stage {stage_idx+1}/{total_stages} [{stage_name}] "
                    f"Epoch {epoch}/{num_epochs} | "
                    f"loss={avg_loss:.4f} acc={avg_acc:.4f} | "
                    f"val {metrics_str} | "
                    f"time={elapsed}s | "
                    f"best_P@10={best_p10:.4f}"
                )

                # Save best backbone
                if p10 > best_p10:
                    best_p10 = p10
                    torch.save(unwrapped_backbone.state_dict(), best_ckpt_path)
                    accelerator.print(f"  ✓ New best P@10={best_p10:.4f} → saved to {best_ckpt_path}")

            accelerator.wait_for_everyone()  # sync all ranks after validation

        # Also save stage checkpoint (main process only)
        if accelerator.is_main_process:
            stage_ckpt = exp_dir / f"backbone_stage{stage_idx+1}_{stage_name}.pt"
            unwrapped_backbone = accelerator.unwrap_model(backbone)
            torch.save(unwrapped_backbone.state_dict(), stage_ckpt)
            accelerator.print(f"Stage {stage_idx+1} checkpoint saved: {stage_ckpt}")

    accelerator.print(f"\nTraining complete. Best P@10={best_p10:.4f}")
    accelerator.print(f"Best backbone saved to: {best_ckpt_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ReDimNet-b6 with AAM-Softmax on organizer's training data.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune_redimnet.json",
        help="Path to JSON config file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"Config loaded from: {config_path}")
    print(json.dumps(config, indent=2, ensure_ascii=False))

    run_training(config)


if __name__ == "__main__":
    main()
