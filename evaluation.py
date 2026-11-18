"""Metrics at window level with participant-level uncertainty."""

import numpy as np

from features import LABELS


def confusion(truth, predicted):
    truth = np.asarray(truth, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    k = len(LABELS)
    if truth.shape != predicted.shape or np.any(
        (truth < 1) | (truth > k) | (predicted < 1) | (predicted > k)
    ):
        raise ValueError("Rótulos inválidos para a matriz de confusão.")
    return np.bincount((truth - 1) * k + (predicted - 1), minlength=k * k).reshape(k, k)


def scores(matrix):
    matrix = np.asarray(matrix)
    hits = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1)
    called = matrix.sum(axis=0)
    precision = np.divide(hits, called, out=np.zeros_like(hits), where=called > 0)
    recall = np.divide(hits, support, out=np.zeros_like(hits), where=support > 0)
    f1 = np.divide(
        2 * hits, support + called, out=np.zeros_like(hits), where=(support + called) > 0
    )
    return {
        "accuracy": float(hits.sum() / matrix.sum()) if matrix.sum() else None,
        "macro_f1": float(f1.mean()) if matrix.sum() else None,
        "per_class": {
            label: {
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
            for i, label in enumerate(LABELS)
        },
    }


def participant_interval(truth, predicted, groups, repeats=2000, seed=42):
    matrices = np.stack(
        [confusion(truth[groups == p], predicted[groups == p]) for p in np.unique(groups)]
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(matrices), size=(repeats, len(matrices)))
    samples = [scores(cm)["macro_f1"] for cm in matrices[indices].sum(axis=1)]
    return np.quantile(samples, [0.025, 0.975]).tolist()
