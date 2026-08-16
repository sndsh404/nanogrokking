"""
plot.py: every figure in the repo, generated from a run's log and checkpoint.

    python plot.py --run runs/paper        # figures from a completed run
    python plot.py --run runs/paper --out figures

The headline figure (accuracy versus step, log-scaled x axis) is the point of
the whole repo: train accuracy saturates almost immediately, validation
accuracy sits at chance for thousands of steps, then jumps. The Fourier plot
is the explanation: the embedding matrix ends up concentrated on a handful of
frequencies, meaning the model learned to do addition as rotation on a circle.
"""

import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")  # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt
import numpy as np
import torch


def read_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    return cols


def first_above(x, threshold):
    """First index where x crosses threshold, or None if it never does."""
    hits = np.nonzero(x >= threshold)[0]
    return int(hits[0]) if len(hits) else None


def plot_accuracy(log, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log["step"], log["train_acc"], label="train")
    ax.plot(log["step"], log["val_acc"], label="validation")
    ax.set_xscale("log")
    ax.set_xlabel("step (full-batch)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)

    # the two events that define the shape: memorization, then generalization
    memo = first_above(log["train_acc"], 0.99)
    grok = first_above(log["val_acc"], 0.95)
    if memo is not None:
        ax.annotate("memorized", xy=(log["step"][memo], 1.0), xytext=(0.35, 0.55),
                    textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="gray"))
    if grok is not None:
        ax.annotate("grokking", xy=(log["step"][grok], 0.95), xytext=(0.75, 0.3),
                    textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="gray"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss(log, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log["step"], log["train_loss"], label="train")
    ax.plot(log["step"], log["val_loss"], label="validation")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step (full-batch)")
    ax.set_ylabel("cross entropy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fourier_basis(p):
    """Real DFT basis over Z_p as unit-norm rows: const, then cos/sin pairs.

    Multiplying the embedding matrix by this basis measures how much of each
    token's embedding lives at each frequency. A model that does addition as
    rotation concentrates on a few frequencies; a memorizing model does not.
    """
    n = torch.arange(p, dtype=torch.float64)
    rows = [torch.ones(p, dtype=torch.float64) / math.sqrt(p)]
    for k in range(1, (p - 1) // 2 + 1):
        w = 2 * math.pi * k / p
        rows.append(torch.cos(w * n) * math.sqrt(2.0 / p))
        rows.append(torch.sin(w * n) * math.sqrt(2.0 / p))
    return torch.stack(rows)  # (p, p)


def plot_fourier(ckpt_path, out_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    p = ckpt["config"]["p"]
    w_e = ckpt["state_dict"]["embed.weight"].double()[:p]  # drop the "=" token row
    # squared norm of each frequency component, summed over the model dimension
    power = (fourier_basis(p) @ w_e).pow(2).sum(dim=1).numpy()
    freqs = np.array([0] + [k for k in range(1, (p - 1) // 2 + 1) for _ in (0, 0)])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freqs[1:], power[1:], marker="o", markersize=3, linewidth=1)
    ax.set_xlabel("frequency k")
    ax.set_ylabel("embedding power at frequency")
    ax.set_title("the embedding is sparse in the fourier basis")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run directory containing log.csv and model.pt")
    ap.add_argument("--out", default="figures", help="directory for the generated pngs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log = read_log(os.path.join(args.run, "log.csv"))
    plot_accuracy(log, os.path.join(args.out, "headline.png"))
    plot_loss(log, os.path.join(args.out, "loss.png"))
    plot_fourier(os.path.join(args.run, "model.pt"), os.path.join(args.out, "fourier.png"))
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
