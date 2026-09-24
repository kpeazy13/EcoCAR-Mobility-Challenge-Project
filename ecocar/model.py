"""Identical waveform-to-log-mel frontend in training and on the Jetson."""
import math

import torch
from torch import nn

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000
HOP_SAMPLES = 3200


def mel_filters(n_fft=512, bins=64):
    low = 2595 * math.log10(1 + 60 / 700)
    high = 2595 * math.log10(1 + 7600 / 700)
    points = 700 * (10 ** (torch.linspace(low, high, bins + 2) / 2595) - 1)
    hz = torch.linspace(0, SAMPLE_RATE / 2, n_fft // 2 + 1)
    rising = (hz[None, :] - points[:-2, None]) / (points[1:-1] - points[:-2])[:, None]
    falling = (points[2:, None] - hz[None, :]) / (points[2:] - points[1:-1])[:, None]
    return torch.minimum(rising, falling).clamp(min=0)


class HonkNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(512))
        self.register_buffer("mel", mel_filters())
        layers = []
        channels = 1
        for out in (16, 32, 64):
            layers += [nn.Conv2d(channels, out, 3, padding=1), nn.BatchNorm2d(out),
                       nn.ReLU(), nn.MaxPool2d(2)]
            channels = out
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                     nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, waveform):
        # Preserve absolute level; normalization parameters are fixed across devices.
        spectrum = torch.stft(waveform, n_fft=512, hop_length=160,
                              window=self.window, return_complex=True).abs().square()
        features = torch.matmul(self.mel, spectrum).clamp(min=1e-8).log10()
        return self.encoder(((features + 4.0) / 4.0).unsqueeze(1)).squeeze(-1)
