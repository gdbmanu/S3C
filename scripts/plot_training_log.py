#!/usr/bin/env python3
"""Visualise rapidement un *_training_log.csv : une sous-figure par loss.

Usage :
    python plot_training_log.py chemin/vers/training_log.csv
    python plot_training_log.py log.csv --smooth 10 --log
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LOSSES = [
    ("1", "Loss 1 : alignement  (z2_pred vs z2)"),
    ("2", "Loss 2 : shift  (y_pred vs y_cible)"),
    ("3", "Loss 3 : régularisation  (z1 vs z1 teacher)"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="fichier *_training_log.csv")
    parser.add_argument("--smooth", type=int, default=1,
                        help="fenêtre de moyenne glissante pour la courbe train (défaut : 1 = aucune)")
    parser.add_argument("--log", action="store_true", help="axe y en échelle logarithmique")
    parser.add_argument("-o", "--output", default=None,
                        help="image de sortie (défaut : à côté du CSV, en .png)")
    parser.add_argument("--no-show", action="store_true", help="n'ouvre pas la fenêtre")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    needed = ["epoch", "batch"] + [f"{s}_loss_{k}" for k, _ in LOSSES for s in ("train", "val")]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Colonnes manquantes dans {csv_path.name} : {missing}")

    # Itération globale : 'batch' repart de 0 à chaque epoch, on cumule les epochs précédentes.
    epoch_len = df.groupby("epoch")["batch"].max()
    offset = epoch_len.cumsum().shift(fill_value=0)
    df["iter"] = df["epoch"].map(offset) + df["batch"]

    fig, axes = plt.subplots(len(LOSSES), 1, figsize=(10, 9), sharex=True)
    for ax, (k, title) in zip(axes, LOSSES):
        train = df[f"train_loss_{k}"]
        if args.smooth > 1:
            ax.plot(df["iter"], train, color="C0", alpha=0.25)
            ax.plot(df["iter"], train.rolling(args.smooth, min_periods=1).mean(),
                    color="C0", label=f"train (moyenne sur {args.smooth})")
        else:
            ax.plot(df["iter"], train, color="C0", label="train")
        ax.plot(df["iter"], df[f"val_loss_{k}"], color="C1", label="val")

        for start in offset.iloc[1:]:                      # frontières d'epochs
            ax.axvline(start, color="gray", linestyle=":", linewidth=0.8)
        if args.log:
            ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("MSE")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("itération (les pointillés marquent les débuts d'epoch)")
    fig.suptitle(csv_path.name)
    fig.tight_layout()

    out = Path(args.output) if args.output else csv_path.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"Figure enregistrée : {out}")

    last = df.iloc[-1]
    print(f"Dernier point : epoch {int(last['epoch'])}, batch {int(last['batch'])}")
    for k, _ in LOSSES:
        print(f"  loss {k} : train {last[f'train_loss_{k}']:.3e} | val {last[f'val_loss_{k}']:.3e}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()