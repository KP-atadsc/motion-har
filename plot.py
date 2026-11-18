"""Plot the held-out confusion matrix. Optional dependency: matplotlib."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
r = json.loads((ROOT / "release/evaluation.json").read_text(encoding="utf-8"))
matrix = np.asarray(r["confusion_true_by_pred"])
labels = ["Caminhando", "Subindo\nescadas", "Descendo\nescadas", "Sentado", "Em pé", "Deitado"]
rows = matrix.sum(axis=1)
share = matrix / rows[:, None]
navy, teal, muted = "#142D3D", "#006D77", "#425B68"
fig = plt.figure(figsize=(11.8, 6.2), facecolor="white")
ax = fig.add_axes([0.14, 0.16, 0.50, 0.63])
cmap = LinearSegmentedColormap.from_list("signal", ["#F3F6F6", "#AED8D3", teal, "#154454"])
ax.imshow(share, cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(6), labels, fontsize=9)
ax.set_yticks(range(6), [s.replace("\n", " ") for s in labels], fontsize=10)
ax.tick_params(length=0, pad=8, colors=navy)
ax.set_xticks(np.arange(6) - 0.5, minor=True)
ax.set_yticks(np.arange(6) - 0.5, minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)
ax.tick_params(which="minor", length=0)
for (i, j), v in np.ndenumerate(matrix):
    ax.text(
        j,
        i,
        str(v) if v else "·",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="semibold" if v else "normal",
        color="white" if share[i, j] > 0.60 else muted,
    )
for spine in ax.spines.values():
    spine.set_visible(False)
fig.text(
    0.055, 0.92, "Quais atividades o acelerômetro confunde?", fontsize=23, color=navy, weight="bold"
)
fig.text(
    0.056,
    0.86,
    "Atividade real nas linhas e previsão nas colunas. Cada número conta janelas de 2,56 s.",
    fontsize=11,
    color=muted,
)


def pct(value):
    return f"{100 * value:.1f}%".replace(".", ",")


fig.text(0.69, 0.70, pct(r["test"]["macro_f1"]), size=30, color=teal, weight="bold")
fig.text(0.69, 0.645, "F1 macro nas seis classes", size=11, color=navy)
fig.text(0.69, 0.53, str(int(matrix[3, 4])), size=28, color=navy, weight="bold")
fig.text(
    0.69,
    0.465,
    "janelas de pessoas sentadas\nclassificadas como em pé",
    fontsize=10.5,
    color=muted,
    linespacing=1.4,
)
review = r["test_windows"] - r["selective"]["test_kept"]
fig.text(0.69, 0.34, str(review), size=28, color=navy, weight="bold")
fig.text(
    0.69,
    0.275,
    "janelas encaminhadas à revisão\nno modo de classificação seletiva",
    fontsize=10,
    color=muted,
    linespacing=1.4,
)
fig.text(
    0.056,
    0.038,
    "UCI HAR · teste oficial · 2.947 janelas de 9 participantes ausentes do treino · cor por proporção dentro da linha",
    fontsize=9.5,
    color=muted,
)
fig.savefig(ROOT / "release/confusion.png", dpi=145)
plt.close(fig)
