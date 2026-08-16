# nanogrokking

A one-layer transformer learns modular addition, memorizes the training set, sits at chance on held-out pairs for thousands of steps, and then suddenly generalizes. That delayed jump is called grokking. This repo reproduces it from the original paper, small enough to run on a weak laptop with no GPU.

![train and validation accuracy: memorization first, generalization thousands of steps later](figures/headline.png)

## quickstart

```
pip install -r requirements.txt
python grok.py --preset fast
python plot.py --run runs/fast
```

That is the whole thing. About five minutes later you have the figure above, produced on your own machine. If you do not want to train anything, `runs/fast/log.csv` is committed, so `python plot.py --run runs/fast` works on a fresh clone in seconds.

## what it costs

Every number in this table was measured by actually running the code, not estimated. The reference machine is deliberately modest, because that is the point of the repo:

- **cpu:** Intel Core 5 120U (a low-power laptop chip, 12 threads)
- **ram:** 16 GB
- **gpu:** none. everything runs on cpu
- **os / stack:** Windows 11, Python 3.14, torch 2.10.0+cpu
- **peak memory:** the largest preset uses about 460 MB, so any machine that can open a browser can run this

If your laptop is newer than this one, every preset will be faster than the numbers below. See `budget.yaml` for the machine-readable version.

| preset | p | params | steps | runtime | what it is for |
|---|---|---|---|---|---|
| `tiny` | 7 | 8,896 | 30 | ~1 s | ci smoke test |
| `fast` | 53 | 121,728 | 6,000 | 4 min 44 s | the default run, full grokking curve |
| `paper` | 113 | 226,816 | 40,000 | see budget.yaml | the exact setup from the paper |

## what you are looking at

The task is addition modulo a prime p. There are exactly p squared possible questions, so the dataset is finite and small. The model sees 45 percent of the questions and is tested on the rest.

The blue curve is accuracy on questions the model trains on. It hits 100 percent almost immediately, by step 200 or so. The orange curve is accuracy on questions the model has never seen. It stays near chance for thousands of steps. Then, around step 4000, it climbs to near 100 percent in a few hundred steps.

For most of training, the model is a lookup table. It has memorized every training answer and knows nothing about addition. Then, late and suddenly, it stops being a lookup table. Something inside it changes while both curves look flat. That is the interesting part: generalization was not absent during the flat stretch, it was assembling.

The weight decay setting is what forces this. With weight decay 1.0, every parameter is constantly pulled toward zero. A memorized lookup table needs large, specific weights to survive that pull. A general mechanism needs less total weight to do the same job, so once the pieces of it exist, the pull toward zero erases the memorized solution first and the mechanism wins. Set `--wd 0` and the model memorizes forever and never groks.

## the fourier plot, or what the model actually learned

![the learned embedding concentrates on a handful of frequencies](figures/fourier.png)

The paper's central finding is that the transformer does not learn addition the way a person does. It learns to convert addition into rotation.

Take the embedding matrix and apply a discrete Fourier transform over the token index. After grokking, almost all the weight lives on a handful of frequencies, four of them in the run above. The model embeds each number as a point on a circle, rotates by the second number, and reads off where it landed, using sines and cosines and the trigonometric identities for a sum of angles. Nobody programmed this. Gradient descent found trigonometry because trigonometry is the low-weight solution to modular addition.

This is why grokking matters beyond a party trick. It is the cleanest example we have of a network holding two different algorithms at once, and of interpretability tools being able to watch one replace the other.

## things to try

- `python grok.py --preset fast --wd 0` and watch grokking disappear. The val curve rises slightly from memorization leakage and never jumps.
- `python grok.py --preset fast --frac-train 0.3` makes the transition later and harder. Below some fraction it never happens at all.
- `python grok.py --preset fast --seed 1` to see how much of the curve's shape is luck.
- `python grok.py --preset paper` if you can leave the machine alone for a few hours. The committed log in `runs/paper/` is from exactly this command.

## layout

- `grok.py` is the model, the data, and the training loop. Start here.
- `plot.py` turns a run's log and checkpoint into the three figures.
- `test_grok.py` checks the data, the shapes, the determinism, and that the tiny preset trains.
- `budget.yaml` is what each preset costs on modest hardware, measured rather than guessed.
- `REFERENCES.md` lists everything consulted and states what was and was not taken from each source.

## reference and originality

The task, architecture, and training setup follow Nanda, Chan, Lieberum, Smith and Steinhardt, "Progress Measures for Grokking via Mechanistic Interpretability", ICLR 2023, https://arxiv.org/abs/2301.05217. Grokking was discovered by Power et al. 2022, https://arxiv.org/abs/2201.02177.

All code here is an original implementation written from the papers' descriptions. No code was copied from any repository, including the paper authors' own code. `REFERENCES.md` has the full list of what was studied.
