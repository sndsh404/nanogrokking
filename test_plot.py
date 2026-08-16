"""Tests for plot.py. Run with: pytest

The Fourier code is the dangerous part: a wrong axis or normalization still
produces a plausible-looking spectrum, so the figure would be a lie that
looks right. The basis test pins the math (a real DFT basis is orthonormal),
and the synthetic-signal test pins the plumbing (a pure cosine at frequency 7
must land entirely on k = 7 and nowhere else). The smoke tests keep every
figure function runnable against the committed runs.
"""

import math

import numpy as np
import torch

from plot import (first_above, fourier_basis, per_frequency_power,
                  plot_accuracy, plot_fourier, plot_loss, plot_spectrum_time,
                  plot_wd0, read_log)


def test_fourier_basis_is_orthonormal():
    # an orthonormal basis satisfies B @ B.T = I; this fails if a row is
    # mis-scaled (wrong normalization) or built over the wrong axis
    b = fourier_basis(53)
    ident = b @ b.T
    assert torch.allclose(ident, torch.eye(53, dtype=torch.float64), atol=1e-9)


def test_per_frequency_power_finds_known_frequency():
    # an embedding whose rows are exactly cos(2 pi 7 n / 53) must put all of
    # its power on frequency 7; any leakage means the transform is wrong
    p, d = 53, 8
    n = torch.arange(p, dtype=torch.float64)
    row = torch.cos(2 * math.pi * 7 * n / p)
    embed = torch.cat([row[:, None].expand(p, d), torch.zeros(1, d)])  # "=" row
    power = per_frequency_power(embed, p)
    assert power[6] >= 0.99 * power.sum()
    assert power.sum() > 0


def test_read_log_roundtrip(tmp_path):
    path = tmp_path / "log.csv"
    path.write_text("step,train_loss,train_acc,val_loss,val_acc,param_norm\n"
                    "0,4.1,0.02,4.1,0.02,36.1\n"
                    "50,3.6,0.12,4.5,0.01,37.0\n")
    log = read_log(str(path))
    assert list(log["step"]) == [0, 50]
    assert log["param_norm"][1] == 37.0


def test_first_above():
    x = np.array([0.1, 0.4, 0.6, 0.99])
    assert first_above(x, 0.5) == 2
    assert first_above(x, 0.99) == 3
    assert first_above(x, 1.0) is None


def test_figure_functions_produce_files(tmp_path):
    log = read_log("runs/fast/log.csv")
    log_wd0 = read_log("runs/fast-wd0/log.csv")
    out = str(tmp_path)
    plot_accuracy(log, f"{out}/headline.png")
    plot_loss(log, f"{out}/loss.png")
    plot_fourier("runs/fast/model.pt", f"{out}/fourier.png")
    plot_wd0(log, log_wd0, f"{out}/wd0.png")
    plot_spectrum_time("runs/fast/snapshots.pt", log, f"{out}/evolution.png")
    for name in ["headline.png", "loss.png", "fourier.png", "wd0.png", "evolution.png"]:
        size = (tmp_path / name).stat().st_size
        assert size > 10000  # a real 8x5 figure is tens of KB; a broken plot is not
