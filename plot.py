"""
plot.py: every figure in the repo, generated from a run's log and checkpoint.

    python plot.py --run runs/paper        # figures from a completed run
    python plot.py --run runs/paper --out figures
    python plot.py --run runs/fast --wd0 runs/fast-wd0   # plus the wd=0 comparison
    python plot.py --run runs/fast --sweep runs/sweep    # plus fraction.png

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


def plot_sweep(points, steps, out_path):
    """Grokking step versus train fraction, from a set of sweep runs.

    points: (frac, grok_step or None) pairs, grok_step being the first step
    with validation accuracy >= 0.95. Runs that never grok are crosses above
    the plot area: at a fixed step budget "never" means "not within the
    budget", and the figure says so.
    """
    grokked = [(f, g) for f, g in points if g is not None]
    failed = [f for f, g in points if g is None]
    fig, ax = plt.subplots(figsize=(8, 5))
    if grokked:
        ax.plot([f for f, _ in grokked], [g for _, g in grokked],
                marker="o", label="grokking step")
    if failed:
        ax.scatter(failed, [steps * 1.08] * len(failed), marker="x", s=60,
                   color="red", label=f"no grokking within {steps} steps")
        ax.set_ylim(bottom=0, top=steps * 1.25)
    ax.set_xlabel("train fraction")
    ax.set_ylabel("first step with validation accuracy >= 0.95")
    ax.set_title("grokking needs enough training data")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def sweep_points(dirs):
    """(frac, grok_step or None) pairs and the step budget, from run dirs."""
    points = []
    steps = 0
    for d in dirs:
        cfg = torch.load(os.path.join(d, "model.pt"),
                         map_location="cpu", weights_only=False)["config"]
        log = read_log(os.path.join(d, "log.csv"))
        i = first_above(log["val_acc"], 0.95)
        points.append((cfg["frac_train"], log["step"][i] if i is not None else None))
        steps = max(steps, cfg["steps"])
    return sorted(points), steps


def plot_wd0(log, log_wd0, out_path):
    """Accuracy with and without weight decay, from two committed runs.

    Both runs memorize: both train curves saturate. Only the run with weight
    decay generalizes. The flat orange line is the causal evidence that weight
    decay, not time or step count, is what produces grokking.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log["step"], log["train_acc"], color="C0", label="train, wd=1.0")
    ax.plot(log["step"], log["val_acc"], color="C1", label="validation, wd=1.0")
    ax.plot(log_wd0["step"], log_wd0["train_acc"], color="C0", linestyle="--",
            alpha=0.5, label="train, wd=0")
    ax.plot(log_wd0["step"], log_wd0["val_acc"], color="C1", linestyle="--",
            alpha=0.5, label="validation, wd=0")
    ax.set_xscale("log")
    ax.set_xlabel("step (full-batch)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("no weight decay: memorization without grokking")
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


def per_frequency_power(embed, p):
    """Cosine and sine rows combined into one power value per frequency k."""
    power = (fourier_basis(p) @ embed.double()[:p]).pow(2).sum(dim=1).numpy()
    return np.array([power[1 + 2 * i] + power[2 + 2 * i] for i in range((p - 1) // 2)])


def plot_spectrum_time(snap_path, log, out_path):
    """Embedding spectrum at several steps, from a run's snapshots.pt.

    The claim this figure tests: the winning frequencies are already
    concentrated during the flat stretch, while validation accuracy is still
    low. Same y axis on every panel so the growth is visible directly.
    """
    snap = torch.load(snap_path, map_location="cpu", weights_only=False)
    p = snap["config"]["p"]
    ks = np.arange(1, (p - 1) // 2 + 1)
    n = len(snap["steps"])
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.2 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for ax, step, embed in zip(axes.flat, snap["steps"], snap["embed"]):
        ax.plot(ks, per_frequency_power(embed, p), marker="o", markersize=3, linewidth=1)
        i = int(np.nonzero(log["step"] == step)[0][0])
        ax.set_title(f"step {step}  (validation accuracy {log['val_acc'][i]:.2f})",
                     fontsize=10)
        ax.set_xlabel("frequency k")
        ax.set_ylabel("embedding power")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("the spectrum concentrates before the accuracy jumps")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run directory containing log.csv and model.pt")
    ap.add_argument("--wd0", default=None,
                    help="optional second run directory trained with --wd 0; "
                         "adds wd0.png comparing the two runs")
    ap.add_argument("--sweep", default=None,
                    help="optional directory of sweep run directories; adds "
                         "fraction.png of grokking step versus train fraction, "
                         "including the --run directory as one point")
    ap.add_argument("--out", default="figures", help="directory for the generated pngs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log = read_log(os.path.join(args.run, "log.csv"))
    plot_accuracy(log, os.path.join(args.out, "headline.png"))
    plot_loss(log, os.path.join(args.out, "loss.png"))
    plot_fourier(os.path.join(args.run, "model.pt"), os.path.join(args.out, "fourier.png"))
    snap_path = os.path.join(args.run, "snapshots.pt")
    if os.path.exists(snap_path):
        plot_spectrum_time(snap_path, log, os.path.join(args.out, "evolution.png"))
    if args.wd0 is not None:
        log_wd0 = read_log(os.path.join(args.wd0, "log.csv"))
        plot_wd0(log, log_wd0, os.path.join(args.out, "wd0.png"))
    if args.sweep is not None:
        dirs = [os.path.join(args.sweep, d) for d in sorted(os.listdir(args.sweep))]
        points, steps = sweep_points(dirs + [args.run])
        plot_sweep(points, steps, os.path.join(args.out, "fraction.png"))
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
