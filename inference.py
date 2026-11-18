"""Portable inference; the release contains only numerical arrays, no pickled estimators."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from features import HZ, LABELS, LENGTH, NAMES, extract

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "release" / "model.npz"


def load_model(path=MODEL):
    with np.load(path, allow_pickle=False) as data:
        model = {name: data[name] for name in data.files}
    if model["feature_names"].tolist() != NAMES or model["labels"].tolist() != list(LABELS):
        raise ValueError("Versão do modelo incompatível com os atributos ou classes.")
    return model


def probabilities(features, model):
    x = (np.asarray(features, dtype=np.float64) - model["mean"]) / model["scale"]
    if "fourier_weights" in model:
        angle = x @ model["fourier_weights"] + model["fourier_offset"]
        x = np.cos(angle) * np.sqrt(2 / angle.shape[1])
    logits = x @ model["weights"].T + model["intercept"]
    exps = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exps / exps.sum(axis=1, keepdims=True)


def read_window(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = ["t_ms", "ax_g", "ay_g", "az_g"]
        if reader.fieldnames != fields:
            raise ValueError("Cabeçalho esperado: " + ",".join(fields))
        rows = []
        for row in reader:
            if None in row or any(row[k] is None for k in fields):
                raise ValueError("Amostra com número incorreto de colunas.")
            rows.append([float(row[k]) for k in fields])
    values = np.asarray(rows)
    if values.shape != (LENGTH, 4) or not np.isfinite(values).all():
        raise ValueError("São necessárias 128 amostras finitas.")
    if not np.allclose(np.diff(values[:, 0]), 1000 / HZ, atol=2, rtol=0):
        raise ValueError("Cadência fora de 50 Hz (tolerância de 2 ms por intervalo).")
    return values[:, 1:]


def classify(path, model):
    score = probabilities(extract(read_window(path)[None]), model)[0]
    best = int(score.argmax())
    return {
        "classe_mais_provavel": LABELS[best],
        "escore": round(float(score[best]), 4),
        "decisao": "classificar" if score[best] >= model["threshold"] else "revisar",
        "limiar": round(float(model["threshold"]), 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classificar uma janela de acelerômetro")
    parser.add_argument("csv", nargs="?", type=Path, default=ROOT / "data/examples/activity-1.csv")
    parser.add_argument("--model", type=Path, default=MODEL)
    args = parser.parse_args()
    print(json.dumps(classify(args.csv, load_model(args.model)), ensure_ascii=False, indent=2))
