"""Interpretable features from total acceleration in g, sampled at 50 Hz."""

import numpy as np

HZ, LENGTH = 50, 128
# Static labels describe a posture throughout the window, not a transition.
LABELS = ("caminhando", "subindo escadas", "descendo escadas", "sentado", "em pé", "deitado")
CHANNELS = ("x", "y", "z", "norm")
STATS = (
    "mean_g",
    "sd_g",
    "iqr_g",
    "mean_abs_step_g",
    "lag4_corr",
    "peak_hz",
    "power_0p4_2",
    "power_2_5",
    "power_5_12",
)
NAMES = [f"{axis}_{stat}" for stat in STATS for axis in CHANNELS] + [
    "corr_xy",
    "corr_xz",
    "corr_yz",
]


def safe_ratio(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b > 1e-12)


def extract(windows):
    raw = np.asarray(windows, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (LENGTH, 3) or not np.isfinite(raw).all():
        raise ValueError("Cada janela precisa ter 128 amostras finitas em três eixos.")
    signals = np.concatenate((raw, np.linalg.norm(raw, axis=2, keepdims=True)), axis=2)
    mean = signals.mean(axis=1)
    centered = signals - mean[:, None, :]
    sd = signals.std(axis=1)
    q25, q75 = np.percentile(signals, [25, 75], axis=1)
    step = np.abs(np.diff(signals, axis=1)).mean(axis=1)
    a, b = centered[:, :-4], centered[:, 4:]
    lag = safe_ratio((a * b).sum(axis=1), np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1)))
    # Remove DC and taper to reduce spectral leakage at the window boundaries.
    power = np.abs(np.fft.rfft(centered * np.hanning(LENGTH)[None, :, None], axis=1)) ** 2
    freq = np.fft.rfftfreq(LENGTH, 1 / HZ)
    total = power[:, 1:].sum(axis=1)
    useful = (freq > 0) & (freq < 12)
    peak = np.where(total > 1e-12, freq[useful][power[:, useful].argmax(axis=1)], 0)
    blocks = [mean, sd, q75 - q25, step, lag, peak]
    for low, high in ((0.4, 2), (2, 5), (5, 12)):
        band = (freq >= low) & (freq < high)
        blocks.append(safe_ratio(power[:, band].sum(axis=1), total))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        cov = (centered[:, :, i] * centered[:, :, j]).mean(axis=1)
        blocks.append(safe_ratio(cov, sd[:, i] * sd[:, j])[:, None])
    return np.column_stack(blocks).astype(np.float32)
