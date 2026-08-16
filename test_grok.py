"""Tests for grok.py. Run with: pytest

The value assertions (exactly p*p pairs, correct labels, disjoint splits,
identical seeds giving identical runs) are what make the training curves
trustworthy: if the data or the seeding is wrong, the famous plot is a lie.
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
