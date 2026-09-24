import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Windows, read_manifest
from .model import HonkNet, SAMPLE_RATE, WINDOW_SAMPLES, HOP_SAMPLES


def metrics(labels, probabilities, threshold):
    positive = probabilities >= threshold
    actual = labels == 1
    tp = int((positive & actual).sum())
    fp = int((positive & ~actual).sum())
    fn = int((~positive & actual).sum())
    tn = int((~positive & ~actual).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1),
                false_positive_rate=fp / max(fp + tn, 1))


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for wave, target in loader:
        labels.extend(target.tolist())
        probabilities.extend(model(wave.to(device)).sigmoid().cpu().tolist())
    return np.array(labels), np.array(probabilities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_manifest(args.manifest)
    loaders = {s: DataLoader(Windows(rows, s), batch_size=args.batch_size, shuffle=s == "train")
               for s in ("train", "val", "test")}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = HonkNet().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_rows = loaders["train"].dataset.rows
    positives = sum(r["label"] == "1" for r in train_rows)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        (len(train_rows) - positives) / positives, device=args.device))
    best = -1
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for wave, labels in loaders["train"]:
            optimizer.zero_grad()
            loss = criterion(model(wave.to(args.device)), labels.to(args.device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        labels, probabilities = predict(model, loaders["val"], args.device)
        candidates = [(metrics(labels, probabilities, float(t))["f1"], float(t))
                      for t in np.linspace(0.05, 0.95, 91)]
        score, threshold = max(candidates)
        print(f"epoch={epoch+1} loss={total/len(train_rows):.4f} val_f1={score:.4f}", flush=True)
        if score > best:
            best = score
            torch.save(dict(state_dict=model.state_dict(), threshold=threshold,
                            sample_rate=SAMPLE_RATE, window_samples=WINDOW_SAMPLES,
                            hop_samples=HOP_SAMPLES, schema=1), output / "honk.pt")
    checkpoint = torch.load(output / "honk.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    report = {"seed": args.seed, "threshold": checkpoint["threshold"],
              "selection": "Best validation F1; test set evaluated only after selection",
              "counts": {s: len(loader.dataset) for s, loader in loaders.items()}}
    for split in ("val", "test"):
        labels, probabilities = predict(model, loaders[split], args.device)
        report[split] = metrics(labels, probabilities, checkpoint["threshold"])
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
