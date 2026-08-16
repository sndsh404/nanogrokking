"""walkthrough.py: one trained model, one addition, every number shown.

    python walkthrough.py --run runs/fast --a 7 --b 40

Prints three things with real values from the committed checkpoint: the
Fourier spectrum of the embedding with the dominant frequencies named, the
angle each operand sits at on the learned circles, and a trace of
(a + b) mod p through the model: embeddings, attention weights, hidden
layer, logits. WALKTHROUGH.md is the prose companion to this output.
"""

import argparse
import math

import torch
import torch.nn.functional as F

from grok import Config, Model
from plot import fourier_basis, per_frequency_power


def load(run):
    ckpt = torch.load(f"{run}/model.pt", map_location="cpu", weights_only=False)
    cfg = Config(**ckpt["config"])
    model = Model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg, ckpt["state_dict"]


def spectrum(state, p, top=5):
    power = per_frequency_power(state["embed.weight"], p)
    total = power.sum()
    order = sorted(range(len(power)), key=lambda k: -power[k])
    print(f"embedding Fourier spectrum, total power {total:.1f}:")
    cum = 0.0
    for k in order[:top]:
        cum += power[k]
        print(f"  k = {k + 1:2d}   power {power[k]:6.2f}   "
              f"share {power[k] / total:.3f}   cumulative {cum / total:.3f}")
    return [k + 1 for k in order[:top]]


def circle_angles(state, p, ks, tokens):
    """Each token's angle at frequency k, from its embedding's projection.

    The embedding of token n is projected onto the cosine and sine components
    of frequency k (two vectors in the 96-dimensional model space), giving a
    point (x, y) per token. If the model places numbers on a circle, the
    angle of that point advances by 360 * k / p degrees per unit of n.
    """
    w_e = state["embed.weight"].double()[:p]
    n = torch.arange(p, dtype=torch.float64)
    print(f"tokens {tokens} as points on the learned circles "
          f"(angles relative to token 0):")
    for k in ks:
        c = torch.cos(2 * math.pi * k * n / p) * math.sqrt(2.0 / p)
        s = torch.sin(2 * math.pi * k * n / p) * math.sqrt(2.0 / p)
        alpha, beta = w_e.T @ c, w_e.T @ s  # per-dimension cos/sin components
        pts = [(float(w_e[t] @ alpha), float(w_e[t] @ beta)) for t in tokens]
        ang = [math.degrees(math.atan2(y, x)) for x, y in pts]
        rel = [(a - ang[0]) % 360 for a in ang]
        exp = [(360.0 * k * t / p) % 360 for t in tokens]
        print(f"  k = {k:2d}:  measured {[round(v, 1) for v in rel]}"
              f"   expected {[round(v, 1) for v in exp]}")
        print(f"        point for token {tokens[1]}: "
              f"({pts[1][0]:.3f}, {pts[1][1]:.3f})")
    print("  (measured angles are off by a few degrees because the embedding "
          "also\n   carries the other three frequencies; the circle is the "
          "dominant structure)")


def trace(model, cfg, a, b):
    p = cfg.p
    c = (a + b) % p
    x = torch.tensor([[a, b, p]])
    print(f"tracing ({a} + {b}) mod {p} = {c}:")

    with torch.no_grad():
        h = model.embed(x) + model.pos.weight[None]
        print(f"  tokens {[a, b, '=']} -> three 96-dim vectors, "
              f"norms {[round(float(v), 2) for v in h[0].norm(dim=1)]}")

        nh, dh = model.attn.n_head, model.attn.d_head
        q = model.attn.wq(h).view(1, 3, nh, dh).transpose(1, 2)
        k = model.attn.wk(h).view(1, 3, nh, dh).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(dh)
        scores = scores.masked_fill(model.attn.mask == 0, float("-inf"))
        att = F.softmax(scores, dim=-1)[0, :, 2, :]
        print('  attention at the "=" position, per head:')
        for hd in range(nh):
            print(f"    head {hd}: to {a} {float(att[hd, 0]):.3f}, "
                  f"to {b} {float(att[hd, 1]):.3f}, "
                  f'to "=" {float(att[hd, 2]):.3f}')

        h = h + model.attn(h)
        hidden = F.relu(model.w_in(h))
        top = hidden[0, -1].topk(5)
        print(f"  hidden layer: {int((hidden[0, -1] > 0).sum())} of "
              f"{cfg.d_mlp} neurons active at the answer position, "
              f"strongest {[round(float(v), 2) for v in top.values]}")

        logits = model(x)[0]
        probs = F.softmax(logits, dim=-1)
        vals, idx = logits.topk(5)
        print(f"  top-5 logits at the answer position "
              f"(the answer is {c}):")
        for v, i in zip(vals.tolist(), idx.tolist()):
            mark = "  <== answer" if i == c else ""
            print(f"    {i:2d}   logit {v:7.3f}   "
                  f"probability {float(probs[i]):.4f}{mark}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="runs/fast")
    ap.add_argument("--a", type=int, default=7)
    ap.add_argument("--b", type=int, default=40)
    args = ap.parse_args()

    model, cfg, state = load(args.run)
    ks = spectrum(state, cfg.p)
    print()
    circle_angles(state, cfg.p, ks[:4], [0, args.a, args.b, (args.a + args.b) % cfg.p])
    print()
    trace(model, cfg, args.a, args.b)


if __name__ == "__main__":
    main()
