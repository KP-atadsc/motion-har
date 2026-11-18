"""Compare models on out-of-fold participants, then run one locked test evaluation."""

import argparse
import csv
import gzip
import json
import tempfile
from pathlib import Path

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from dataset import DATA, load_dataset
from evaluation import confusion, participant_interval, scores
from features import LABELS, NAMES
from inference import load_model, probabilities

SEED = 42
TARGET_COVERAGE = 0.90


def candidates():
    linear = Pipeline(
        [("scale", StandardScaler()), ("head", LogisticRegression(C=1, max_iter=700))]
    )
    fourier = Pipeline(
        [
            ("scale", StandardScaler()),
            ("map", RBFSampler(gamma=1 / len(NAMES), n_components=384, random_state=SEED)),
            ("head", LogisticRegression(C=8, max_iter=700)),
        ]
    )
    return {"linear": linear, "fourier_logistic": fourier}


def cross_validate(x, y, people):
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(x, y, people))
    results, pooled = {}, {}
    for name, candidate in candidates().items():
        oof = np.empty((len(y), len(LABELS)))
        fold_scores = []
        for train, valid in folds:
            if np.intersect1d(people[train], people[valid]).size:
                raise ValueError("Participante compartilhado entre folds.")
            model = clone(candidate).fit(x[train], y[train])
            oof[valid] = model.predict_proba(x[valid])
            fold_scores.append(
                float(f1_score(y[valid], oof[valid].argmax(axis=1) + 1, average="macro"))
            )
        results[name] = {
            "pooled_macro_f1": float(f1_score(y, oof.argmax(axis=1) + 1, average="macro")),
            "fold_macro_f1": fold_scores,
        }
        pooled[name] = oof
    return results, pooled, [np.unique(people[valid]).tolist() for _, valid in folds]


def export(model, threshold, path):
    scaler, head = model.named_steps["scale"], model.named_steps["head"]
    if head.classes_.tolist() != list(range(1, 7)):
        raise ValueError("Ordem de classes incompatível.")
    params = {
        "feature_names": np.array(NAMES),
        "labels": np.array(LABELS),
        "threshold": threshold,
        "mean": scaler.mean_,
        "scale": scaler.scale_,
        "weights": head.coef_,
        "intercept": head.intercept_,
    }
    if "map" in model.named_steps:
        projection = model.named_steps["map"]
        params.update(
            fourier_weights=projection.random_weights_, fourier_offset=projection.random_offset_
        )
    np.savez_compressed(path, **params)


def fit_and_evaluate(data_dir, output):
    ds = load_dataset(data_dir)
    x, y, people = ds["x_train"].astype(float), ds["y_train"], ds["subject_train"]
    cv, pooled, folds = cross_validate(x, y, people)
    name = max(cv, key=lambda key: cv[key]["pooled_macro_f1"])
    threshold = float(np.quantile(pooled[name].max(axis=1), 1 - TARGET_COVERAGE))
    final = candidates()[name].fit(x, y)
    output.mkdir(parents=True, exist_ok=True)
    export(final, threshold, output / "model.npz")

    xt, yt, gt = ds["x_test"].astype(float), ds["y_test"], ds["subject_test"]
    prob = probabilities(xt, load_model(output / "model.npz"))
    reference = final.predict_proba(xt)
    if not np.allclose(prob, reference, atol=1e-9, rtol=1e-8):
        raise ValueError("A inferência portátil divergiu do modelo de referência.")
    pred = prob.argmax(axis=1) + 1
    matrix = confusion(yt, pred)
    full = scores(matrix)
    if not np.isclose(full["macro_f1"], f1_score(yt, pred, average="macro")):
        raise ValueError("Divergência no cálculo independente do F1.")
    kept = prob.max(axis=1) >= threshold
    majority = np.bincount(y).argmax()
    baseline = scores(confusion(yt, np.full_like(yt, majority)))
    report = {
        "dataset_doi": "10.24432/C54S4K",
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
        "feature_count": len(NAMES),
        "train_windows": len(x),
        "test_windows": len(xt),
        "train_subjects": np.unique(people).tolist(),
        "test_subjects": np.unique(gt).tolist(),
        "cv_validation_subjects": folds,
        "cv_models": cv,
        "selected_model": name,
        "baseline_most_frequent": {
            "label": LABELS[majority - 1],
            "macro_f1": baseline["macro_f1"],
            "accuracy": baseline["accuracy"],
        },
        "test": full,
        "confusion_true_by_pred": matrix.tolist(),
        "test_macro_f1_subject_bootstrap_95": participant_interval(yt, pred, gt),
        "selective": {
            "threshold": threshold,
            "oof_target_coverage": TARGET_COVERAGE,
            "test_kept": int(kept.sum()),
            "test_total": len(kept),
            "test_coverage": float(kept.mean()),
            "accepted_only": scores(confusion(yt[kept], pred[kept])),
        },
        "by_subject": {
            str(p): {
                "windows": int((gt == p).sum()),
                "accuracy": float(np.mean(yt[gt == p] == pred[gt == p])),
            }
            for p in np.unique(gt)
        },
    }
    (output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with gzip.open(output / "test-predictions.csv.gz", "wt", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["official_test_row", "subject", "truth_id", "pred_id", "top_score", "accepted"]
        )
        for i in range(len(yt)):
            writer.writerow([i, gt[i], yt[i], pred[i], f"{prob[i].max():.10f}", int(kept[i])])
    return report


def same_release(existing, staged):
    names = ("model.npz", "evaluation.json", "test-predictions.csv.gz")
    if not all((existing / name).is_file() for name in names):
        return False
    for name in names:
        first = (existing / name).read_bytes()
        second = (staged / name).read_bytes()
        if name.endswith(".gz"):
            first, second = gzip.decompress(first), gzip.decompress(second)
        if first != second:
            return False
    return True


def train(data_dir, output):
    # Keep the previous release readable if fitting, verification or disk writes fail.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        staged = Path(scratch) / "bundle"
        report = fit_and_evaluate(data_dir, staged)
        try:
            staged.rename(output)
        except OSError:
            if not output.exists():
                raise
            if not same_release(output, staged):
                raise FileExistsError(
                    f"Já existe outra versão em {output}. Escolha um novo --out."
                ) from None
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar e avaliar por participante")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=Path(".local/training"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = train(args.data, args.out)
    print(f"{result['selected_model']} | F1 macro: {result['test']['macro_f1']:.4f} | {args.out}")
