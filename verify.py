"""Independently replay the checked-in model and compare every held-out prediction."""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

from dataset import load_dataset
from evaluation import confusion, scores
from inference import load_model, probabilities

ROOT = Path(__file__).resolve().parent


def verify():
    ds = load_dataset()
    path = ROOT / "release"
    reference = json.loads((path / "evaluation.json").read_text(encoding="utf-8"))
    model = load_model(path / "model.npz")
    prob = probabilities(ds["x_test"], model)
    predicted = prob.argmax(axis=1) + 1
    kept = prob.max(axis=1) >= model["threshold"]
    matrix = confusion(ds["y_test"], predicted)
    with gzip.open(path / "test-predictions.csv.gz", "rt", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != len(predicted):
        raise ValueError("Quantidade de previsões divergente.")
    for i, row in enumerate(rows):
        if (
            int(row["official_test_row"]) != i
            or int(row["pred_id"]) != predicted[i]
            or int(row["truth_id"]) != ds["y_test"][i]
            or int(row["subject"]) != ds["subject_test"][i]
            or int(row["accepted"]) != int(kept[i])
            or not np.isclose(float(row["top_score"]), prob[i].max(), rtol=0, atol=1e-8)
        ):
            raise ValueError(f"Previsão divergente na linha {i}.")
    actual = scores(matrix)
    if matrix.tolist() != reference["confusion_true_by_pred"] or actual != reference["test"]:
        raise ValueError("Métricas agregadas divergentes.")
    return {
        "windows": len(rows),
        "macro_f1": actual["macro_f1"],
        "accuracy": actual["accuracy"],
        "coverage": float(kept.mean()),
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
