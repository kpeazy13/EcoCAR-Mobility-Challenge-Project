import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

HAS_ML = all(importlib.util.find_spec(m) for m in ("torch", "numpy", "scipy", "soundfile"))


@unittest.skipUnless(HAS_ML, "Install .[train] for model tests")
class ModelTests(unittest.TestCase):
    def test_training_cli_creates_loadable_artifact(self):
        import contextlib
        import io
        import json
        from unittest.mock import patch
        import numpy as np
        import soundfile as sf
        import torch
        from ecocar.train import main
        from ecocar.model import HonkNet
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["path", "label", "split", "group"])
                for split in ("train", "val", "test"):
                    for label in (0, 1):
                        name = f"{split}-{label}.wav"
                        wave = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * label * 0.1
                        sf.write(root / name, wave, 16000)
                        writer.writerow([name, label, split, name])
            output = root / "artifacts"
            args = ["train", str(manifest), "--epochs", "1", "--device", "cpu", "--output", str(output)]
            with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                main()
            report = json.loads((output / "metrics.json").read_text())
            self.assertEqual(report["counts"], dict(train=2, val=2, test=2))
            self.assertIn("recall", report["test"])
            checkpoint = torch.load(output / "honk.pt", weights_only=True)
            model = HonkNet().eval()
            model.load_state_dict(checkpoint["state_dict"])
            with torch.inference_mode():
                self.assertTrue(torch.isfinite(model(torch.zeros(1, 16000))).all())

    def test_forward_backward_checkpoint(self):
        import torch
        from ecocar.model import HonkNet, WINDOW_SAMPLES
        torch.set_num_threads(2)
        model = HonkNet()
        audio = torch.randn(2, WINDOW_SAMPLES) * 0.1
        logits = model(audio)
        self.assertEqual(tuple(logits.shape), (2,))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            torch.save(model.state_dict(), path)
            restored = HonkNet().eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
            with torch.inference_mode():
                torch.testing.assert_close(model(audio), restored(audio))

    def test_resampling_and_leakage_rejection(self):
        import numpy as np
        import soundfile as sf
        from ecocar.data import load_window, read_manifest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for split in ("train", "val", "test"):
                for label in (0, 1):
                    name = f"{split}-{label}.wav"
                    sf.write(root / name, np.zeros((44100, 2)), 44100)
                    rows.append(dict(path=name, label=label, split=split, group=name))
            manifest = root / "manifest.csv"
            def write():
                with manifest.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["path", "label", "split", "group"])
                    writer.writeheader()
                    writer.writerows(rows)
            write()
            self.assertEqual(len(read_manifest(manifest)), 6)
            wave = load_window(root / rows[0]["path"])
            self.assertEqual(wave.shape, (16000,))
            self.assertTrue(np.isfinite(wave).all())
            rows[2]["group"] = rows[0]["group"]
            write()
            with self.assertRaisesRegex(ValueError, "leakage"):
                read_manifest(manifest)
            sf.write(root / "long.wav", np.zeros(32000), 16000)
            with self.assertRaisesRegex(ValueError, "labeled"):
                load_window(root / "long.wav")


if __name__ == "__main__":
    unittest.main()
