"""Explicit, recording-group-disjoint splits prevent overlapping-window leakage."""
import csv
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .model import SAMPLE_RATE, WINDOW_SAMPLES


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"path", "label", "split", "group"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest requires path,label,split,group columns")
        rows = list(reader)
    groups, files = {}, {}
    for row in rows:
        if row["label"] not in ("0", "1") or row["split"] not in ("train", "val", "test"):
            raise ValueError("Labels must be 0/1; splits must be train/val/test")
        if not row["group"].strip():
            raise ValueError("Each row needs a source recording/session group")
        row["path"] = str((path.parent / row["path"]).resolve())
        if not Path(row["path"]).is_file():
            raise ValueError(f"Missing audio: {row['path']}")
        for key, registry in ((row["group"], groups), (row["path"], files)):
            if key in registry and registry[key] != row["split"]:
                raise ValueError(f"Split leakage: {key}")
            registry[key] = row["split"]
    for split in ("train", "val", "test"):
        if {r["label"] for r in rows if r["split"] == split} != {"0", "1"}:
            raise ValueError(f"{split} must contain both classes")
    return rows


def load_window(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    # Use labeled windows, never randomly crop a long clip whose honk may be elsewhere.
    if not 0.1 <= len(audio) / rate <= 1.05:
        raise ValueError(f"{path}: expected a labeled 0.1–1.05 second window")
    audio = audio.mean(axis=1)
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite audio: {path}")
    divisor = math.gcd(rate, SAMPLE_RATE)
    audio = resample_poly(audio, SAMPLE_RATE // divisor, rate // divisor)
    if len(audio) > WINDOW_SAMPLES:
        audio = audio[:WINDOW_SAMPLES]
    return np.pad(audio, (0, WINDOW_SAMPLES - len(audio))).astype(np.float32)


class Windows(Dataset):
    def __init__(self, rows, split):
        self.rows = [r for r in rows if r["split"] == split]
        self.augment = split == "train"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        wave = torch.from_numpy(load_window(row["path"]))
        if self.augment:
            wave = wave * 10 ** (float(torch.empty(1).uniform_(-9, 3).item()) / 20)
            wave += torch.randn_like(wave) * float(torch.empty(1).uniform_(0, 0.005).item())
            wave = wave.clamp(-1, 1)
        return wave, torch.tensor(float(row["label"]))
