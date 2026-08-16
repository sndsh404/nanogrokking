"""
nanogrokking: grokking on modular arithmetic, small enough to study on a laptop.

A one-layer transformer is trained to compute (a + b) mod p from a fraction of
all possible pairs. It memorizes the training pairs early, sits at chance
accuracy on held-out pairs for thousands of steps, and then suddenly
generalizes. That delayed jump is grokking. This file is the whole project:
data, model, training. Figures live in plot.py.

Usage:
    python grok.py --preset tiny     # smoke test, runs in seconds
    python grok.py --preset fast     # groks in a few minutes on a cpu
    python grok.py --preset paper    # the Nanda et al. 2023 setup, hours on cpu

The task and architecture follow Nanda, Chan, Lieberum, Smith and Steinhardt,
"Progress Measures for Grokking via Mechanistic Interpretability" (ICLR 2023,
arXiv:2301.05217), implemented from the paper's description. See REFERENCES.md.
"""

import argparse
import csv
import math
import os
import random
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    p: int = 113             # the prime modulus; dataset is all p*p pairs
    frac_train: float = 0.3  # fraction of pairs used for training, rest are held out
    d_model: int = 128
    n_head: int = 4          # head width is d_model // n_head = 32
    d_mlp: int = 512
    steps: int = 40000       # full-batch gradient steps (epochs, since batch = all train pairs)
    lr: float = 1e-3
    weight_decay: float = 1.0  # unusually high, and essential: without it the model never groks
    warmup: int = 10         # linear lr warmup over the first few steps
    seed: int = 0
    log_every: int = 100     # eval and record metrics at this cadence


PRESETS = {
    # the canonical configuration from section 3 of the paper
    "paper": Config(),
    # small modulus and budget, tuned so grokking still appears on a weak cpu
    "fast": Config(p=53, d_model=96, n_head=3, d_mlp=384, steps=8000, log_every=50),
    # ci smoke test: proves the code runs and the loss moves, nothing more
    "tiny": Config(p=7, d_model=32, n_head=2, d_mlp=64, steps=30, log_every=10),
}


def make_data(cfg):
    """All p*p equations "a b =" as token ids, split once by a seeded shuffle.

    Tokens 0..p-1 are the integers, token p is "=". The answer (a+b) mod p is
    never an input; the model must produce it at the "=" position.
    """
    eq = cfg.p
    pairs = [(a, b) for a in range(cfg.p) for b in range(cfg.p)]
    rng = random.Random(cfg.seed)
    rng.shuffle(pairs)
    n_train = int(cfg.frac_train * len(pairs))
    train, val = pairs[:n_train], pairs[n_train:]

    def to_tensors(split):
        x = torch.tensor([[a, b, eq] for a, b in split], dtype=torch.long)
        y = torch.tensor([(a + b) % cfg.p for a, b in split], dtype=torch.long)
        return x, y

    return to_tensors(train), to_tensors(val)


class Attention(nn.Module):
    """Single-layer causal self-attention, written out operation by operation.

    Three tokens ("a", "b", "=") see each other under a causal mask, so the "="
    position attends to both operands. That is where the answer is read from.
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        # no biases in attention, matching the paper's model; fewer moving
        # parts means the learned circuit is easier to reverse engineer
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(3, 3)))

    def forward(self, x):
        b, t, d = x.shape
        q = self.wq(x).view(b, t, self.n_head, self.d_head).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_head, self.d_head).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_head, self.d_head).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        scores = scores.masked_fill(self.mask[:t, :t] == 0, float("-inf"))
        out = F.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.wo(out)


class Model(nn.Module):
    """The paper's mainline architecture: one attention layer, one ReLU MLP.

    Two deliberate omissions, both from the paper, both there to keep the
    learned mechanism legible: no LayerNorm (it would fold scale into every
    activation and smear the periodic structure the analysis looks for), and
    untied embedding/unembedding (the unembedding has its own job, rotating by
    -c to score each candidate answer).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.p + 1, cfg.d_model)
        self.pos = nn.Embedding(3, cfg.d_model)  # learned positions, seq len is always 3
        self.attn = Attention(cfg)
        self.w_in = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.w_out = nn.Linear(cfg.d_mlp, cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.p + 1, bias=False)
        # every weight starts as noise scaled by 1/sqrt(d_model), the init the
        # paper's model uses; the unembedding starts smaller still, since it
        # is the readout rather than the compute
        self.apply(self._init)
        nn.init.normal_(self.unembed.weight, std=1.0 / math.sqrt(cfg.p + 1))

    def _init(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=1.0 / math.sqrt(self.cfg.d_model))

    def forward(self, tokens):
        x = self.embed(tokens) + self.pos.weight[None, :, :]
        x = x + self.attn(x)
        x = x + self.w_out(F.relu(self.w_in(x)))
        return self.unembed(x[:, -1, :])  # logits at the "=" position only


def loss_at_answer(logits, targets):
    """Cross entropy over the p answer logits, computed in float64.

    Once the model is confident, float32 log_softmax underflows and the loss
    trace fills with spikes. The spike is an artifact of the measurement, not
    the training, so the measurement gets the extra precision.
    """
    return F.cross_entropy(logits.double(), targets)


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = loss_at_answer(logits, y).item()
        acc = (logits.argmax(-1) == y).float().mean().item()
    model.train()
    return loss, acc


def train(cfg, out_dir):
    # seed everything: same seed, same run. the paper's own code seeds the data
    # split but not the model init; here both are fixed so curves are comparable
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    (train_x, train_y), (val_x, val_y) = make_data(cfg)
    model = Model(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    # weight decay applies to every parameter, as in the paper. it is the
    # regularizer that eventually erases the memorized solution and exposes
    # the generalizing one, so nothing is exempted from it
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
    warmup = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / cfg.warmup, 1.0))

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "log.csv")
    t0 = time.time()
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "train_acc", "val_loss", "val_acc"])
        for step in range(cfg.steps + 1):
            if step % cfg.log_every == 0 or step == cfg.steps:
                train_loss, train_acc = evaluate(model, train_x, train_y)
                val_loss, val_acc = evaluate(model, val_x, val_y)
                writer.writerow([step, train_loss, train_acc, val_loss, val_acc])
                if step % (cfg.log_every * 10) == 0:
                    print(f"step {step:6d}  train loss {train_loss:.4f}  acc {train_acc:.3f}"
                          f"  |  val loss {val_loss:.4f}  acc {val_acc:.3f}"
                          f"  |  {time.time() - t0:.0f}s")
            if step == cfg.steps:
                break
            opt.zero_grad()
            loss = loss_at_answer(model(train_x), train_y)
            loss.backward()
            opt.step()
            warmup.step()

    torch.save({"config": asdict(cfg), "state_dict": model.state_dict()},
               os.path.join(out_dir, "model.pt"))
    print(f"done in {time.time() - t0:.0f}s, {n_params} params, log at {log_path}")
    return log_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", choices=PRESETS, default="fast")
    ap.add_argument("--steps", type=int, default=None, help="override the preset's step count")
    ap.add_argument("--seed", type=int, default=None, help="override the preset's seed")
    ap.add_argument("--out", default=None, help="output directory, default runs/<preset>")
    args = ap.parse_args()

    cfg = PRESETS[args.preset]
    if args.steps is not None:
        cfg.steps = args.steps
    if args.seed is not None:
        cfg.seed = args.seed
    train(cfg, args.out or os.path.join("runs", args.preset))


if __name__ == "__main__":
    main()
