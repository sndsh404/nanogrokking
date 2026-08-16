"""Tests for grok.py. Run with: pytest

The value assertions (exactly p*p pairs, correct labels, disjoint splits,
identical seeds giving identical runs) are what make the training curves
trustworthy: if the data or the seeding is wrong, the famous plot is a lie.
The committed-run assertions are what make the explanations trustworthy: the
fast run must grok with a long delay, the wd=0 run must never grok, and the
grokked embedding must be sparse in the Fourier basis. If the phenomenon or
the mechanism regresses, the README and WHY.md become lies, and these tests
fail.
"""

import torch

from grok import PRESETS, Config, Model, make_data, train


def test_dataset_has_all_pairs_and_disjoint_split():
    cfg = Config(p=13, seed=0)
    (train_x, _), (val_x, _) = make_data(cfg)
    assert len(train_x) + len(val_x) == 13 * 13
    train_pairs = {tuple(t[:2].tolist()) for t in train_x}
    val_pairs = {tuple(t[:2].tolist()) for t in val_x}
    assert train_pairs.isdisjoint(val_pairs)
    assert len(train_x) == int(cfg.frac_train * 13 * 13)


def test_labels_are_modular_sums():
    cfg = Config(p=13, seed=0)
    (train_x, train_y), (val_x, val_y) = make_data(cfg)
    for x, y in list(zip(train_x, train_y)) + list(zip(val_x, val_y)):
        a, b, eq = x.tolist()
        assert eq == cfg.p  # the third token is always "="
        assert y.item() == (a + b) % cfg.p


def test_forward_shape():
    cfg = Config(p=11, d_model=32, n_head=2, d_mlp=64)
    model = Model(cfg)
    tokens = torch.randint(0, cfg.p + 1, (5, 3))
    logits = model(tokens)
    assert logits.shape == (5, cfg.p + 1)


def test_tiny_preset_trains_and_loss_decreases(tmp_path):
    log_path = train(PRESETS["tiny"], str(tmp_path))
    import csv
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    first, last = float(rows[0]["train_loss"]), float(rows[-1]["train_loss"])
    assert last < first


def test_same_seed_same_trajectory(tmp_path):
    cfg = Config(p=7, d_model=32, n_head=2, d_mlp=64, steps=20, log_every=10, seed=42)
    log1 = train(cfg, str(tmp_path / "a"))
    log2 = train(cfg, str(tmp_path / "b"))
    with open(log1) as f1, open(log2) as f2:
        assert f1.read() == f2.read()


def test_no_weight_decay_never_groks():
    """The committed wd=0 run memorizes on schedule and never generalizes.

    reads runs/fast-wd0/log.csv, produced by:
        python grok.py --preset fast --wd 0 --out runs/fast-wd0

    Thresholds, and why each is where it is:
    - train acc 0.99 by step 500: the run must actually memorize, or "it never
      groks" is vacuous. The wd=1.0 run memorizes by step 150; 500 leaves
      room for hardware-level numeric drift without letting a slow run pass.
    - val acc below 0.5 at every logged step: chance is 1/53 (about 0.02), the
      run's actual plateau is 0.343, and grokking means crossing 0.95. 0.5 sits
      halfway between the plateau and grokking, so the assertion fails loudly
      if the counterfactual ever starts to generalize.
    - final param norm above the norm at memorization: with decay off nothing
      removes weight, so the norm only grows (49.6 at step 150, 55.8 at step
      6000 in the committed log). WHY.md derives why this growth is the point.
    """
    import csv
    with open("runs/fast-wd0/log.csv") as f:
        rows = list(csv.DictReader(f))
    step = [int(r["step"]) for r in rows]
    train_acc = [float(r["train_acc"]) for r in rows]
    val_acc = [float(r["val_acc"]) for r in rows]
    norm = [float(r["param_norm"]) for r in rows]

    memo = next((s for s, a in zip(step, train_acc) if a >= 0.99), None)
    assert memo is not None and memo <= 500
    assert max(val_acc) < 0.5
    assert norm[-1] > norm[step.index(memo)]


def test_committed_fast_run_groks():
    """The committed fast run memorizes early, waits, then generalizes.

    reads runs/fast/log.csv. Thresholds, and why each is where it is:
    - train acc 0.99 by step 500: the run must memorize first, or there is
      no grokking to speak of. Actual: step 150; 500 leaves room for
      hardware-level numeric drift.
    - val acc below 0.5 for at least 1000 steps after memorization: this is
      the flat stretch, the signature of grokking. Chance is 1/53 (0.02).
      The curve creeps above chance because a table covering 45 percent of
      pairs beats chance on the rest (WHY.md explains); the creep sits at
      0.46 around step 1500, so 0.5 stays above the creep and far below the
      0.95 of generalization. Actual crossing of 0.5: step 1900.
    - val acc later reaches 0.95: generalization. Actual: step 3150.
    - grok step minus memorization step of at least 1500: grokking is the
      delay, so the delay is the assertion. Actual gap: 3000 steps, twice
      the threshold.
    """
    import csv
    with open("runs/fast/log.csv") as f:
        rows = list(csv.DictReader(f))
    step = [int(r["step"]) for r in rows]
    train_acc = [float(r["train_acc"]) for r in rows]
    val_acc = [float(r["val_acc"]) for r in rows]

    memo = next((s for s, a in zip(step, train_acc) if a >= 0.99), None)
    assert memo is not None and memo <= 500
    first_half = next((s for s, a in zip(step, val_acc) if a >= 0.5), None)
    assert first_half is not None and first_half - memo >= 1000
    grok = next((s for s, a in zip(step, val_acc) if a >= 0.95), None)
    assert grok is not None
    assert grok - memo >= 1500


def test_committed_fast_run_embedding_is_fourier_sparse():
    """After grokking, a few frequencies carry almost all embedding weight.

    Computes the spectrum of runs/fast/model.pt with plot.fourier_basis, the
    same code that draws figures/fourier.png, so the figure and the test can
    never disagree. Each frequency k appears as a cosine row and a sine row;
    their power is summed before ranking.

    Threshold: the 4 largest frequencies must carry at least 90 percent of
    the total embedding power. There are 26 nonzero frequencies at p=53, so
    a diffuse embedding would put only 4/26 = 15 percent on its top 4. The
    committed run puts 94.1 percent on k = 1, 3, 9, 25. 90 percent leaves
    margin for seed and hardware drift while still failing any embedding
    that has not concentrated. Which frequencies get picked depends on the
    seed and is deliberately not asserted.
    """
    from plot import fourier_basis
    ckpt = torch.load("runs/fast/model.pt", map_location="cpu", weights_only=False)
    p = ckpt["config"]["p"]
    w_e = ckpt["state_dict"]["embed.weight"].double()[:p]
    power = (fourier_basis(p) @ w_e).pow(2).sum(dim=1).numpy()
    # rows are: constant, then cos/sin pairs for k = 1..(p-1)//2
    by_freq = {0: power[0]}
    for i, k in enumerate(k for k in range(1, (p - 1) // 2 + 1) for _ in (0, 0)):
        by_freq[k] = by_freq.get(k, 0.0) + power[1 + i]
    top4 = sorted(by_freq.values(), reverse=True)[:4]
    assert sum(top4) >= 0.90 * power.sum()
