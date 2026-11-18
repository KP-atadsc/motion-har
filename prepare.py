"""Rebuild the compact feature table from the checksum-pinned original UCI ZIP."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np

from dataset import DATA
from features import HZ, LABELS, LENGTH, NAMES, extract

URL = (
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
)
SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"


def load_part(archive, split):
    def numbers(relative, dtype=float):
        with archive.open(f"UCI HAR Dataset/{split}/{relative}") as file:
            return np.loadtxt(file, dtype=dtype)

    signal = np.stack(
        [numbers(f"Inertial Signals/total_acc_{axis}_{split}.txt") for axis in "xyz"], axis=2
    )
    return signal, numbers(f"y_{split}.txt", int), numbers(f"subject_{split}.txt", int)


def prepare(raw, destination):
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Arquivo UCI diferente da versão auditada; confira a origem.")
    destination.mkdir(parents=True, exist_ok=True)
    examples = destination / "examples"
    examples.mkdir(exist_ok=True)
    arrays, example_index = {}, []
    with ZipFile(io.BytesIO(raw)) as outer:
        inner = outer.read("UCI HAR Dataset.zip")
    with ZipFile(io.BytesIO(inner)) as archive:
        for split in ("train", "test"):
            signal, y, groups = load_part(archive, split)
            if len(signal) != len(y) or len(y) != len(groups):
                raise ValueError("Contagem de rótulos incompatível com os sinais.")
            arrays[f"x_{split}"] = np.concatenate(
                [extract(signal[i : i + 256]) for i in range(0, len(y), 256)]
            )
            arrays[f"y_{split}"] = y.astype(np.int16)
            arrays[f"subject_{split}"] = groups.astype(np.int16)
            if split == "test":
                for label, name in enumerate(LABELS, 1):
                    index = int(np.flatnonzero(y == label)[0])
                    fname = f"activity-{label}.csv"
                    with (examples / fname).open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(["t_ms", "ax_g", "ay_g", "az_g"])
                        for t, axes in enumerate(signal[index]):
                            writer.writerow([t * (1000 // HZ), *axes])
                    example_index.append(
                        {
                            "file": fname,
                            "label": name,
                            "subject": int(groups[index]),
                            "official_test_row_zero_based": index,
                        }
                    )
    path = destination / "features.npz"
    np.savez_compressed(path, **arrays)
    manifest = {
        "source": URL,
        "doi": "10.24432/C54S4K",
        "source_sha256": SHA256,
        "features_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sample_rate_hz": HZ,
        "window_samples": LENGTH,
        "feature_names": NAMES,
        "examples": example_index,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preparar os atributos a partir do arquivo oficial"
    )
    parser.add_argument("--archive", type=Path, help="ZIP externo da UCI; sem ele, baixar da UCI")
    parser.add_argument("--out", type=Path, default=DATA)
    args = parser.parse_args()
    if args.archive:
        raw = args.archive.read_bytes()
    else:
        with urlopen(URL, timeout=120) as response:
            raw = response.read()
    prepare(raw, args.out)
    print(f"Base e seis janelas de exemplo gravadas em {args.out}.")
