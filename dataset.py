import hashlib
import json
from pathlib import Path

import numpy as np

from features import LABELS, NAMES

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def load_dataset(data_dir=DATA):
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    path = data_dir / "features.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["features_sha256"]:
        raise ValueError("SHA-256 divergente na tabela de atributos.")
    with np.load(path, allow_pickle=False) as saved:
        sets = {key: saved[key] for key in saved.files}
    if manifest["feature_names"] != NAMES:
        raise ValueError("Atributos em ordem ou versão incompatível.")
    for split in ("train", "test"):
        x, y, groups = (sets[f"{name}_{split}"] for name in ("x", "y", "subject"))
        if (
            x.shape != (len(y), len(NAMES))
            or groups.shape != y.shape
            or not np.isfinite(x).all()
            or set(y) != set(range(1, len(LABELS) + 1))
        ):
            raise ValueError(f"Dimensões, classes ou valores inválidos em {split}.")
    if np.intersect1d(sets["subject_train"], sets["subject_test"]).size:
        raise ValueError("Participante compartilhado entre treino e teste.")
    return sets
