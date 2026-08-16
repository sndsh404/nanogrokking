# Exercises

Four predictions to make before you run anything. For each one: write your guess down, run the command, then open the answer and check. The answers explain what happened and why, with numbers from the committed runs. Every command is a few minutes on a laptop cpu at the `fast` preset. Pin the thread count first so your logs match the committed ones bitwise: `export OMP_NUM_THREADS=10` (on windows cmd: `set OMP_NUM_THREADS=10`).

## 1. no weight decay

`--wd 0` turns weight decay off. Training runs for the same 6000 steps. What does the validation accuracy curve do?

```
python grok.py --preset fast --wd 0 --out runs/local/wd0
python plot.py --run runs/local/wd0 --out figures/v1
```

<details>
<summary>answer</summary>

Training accuracy hits 100 percent by step 150, exactly as with decay on. Validation accuracy climbs to 0.343 and stays there for the rest of the run. It never generalizes. Meanwhile the validation loss grows to 28.4, so the model becomes more and more confident while staying wrong, and the total parameter norm climbs from 49.6 to 55.8 without ever falling.

This run is committed in `runs/fast-wd0/`, so you can check your guess against `figures/wd0.png` without training anything. The explanation is in WHY.md: weight decay charges every weight rent each step, and that rent is what erases the memorized table. With no rent the table keeps growing, and the rotation circuit, though present, never outvotes it.

</details>

## 2. less training data

The committed run trains on 45 percent of all pairs and groks at step 3150. You shrink the train fraction to 0.3. What happens to the grokking step?

```
python grok.py --preset fast --frac-train 0.3 --out runs/local/frac03
python plot.py --run runs/local/frac03 --out figures/v1
```

<details>
<summary>answer</summary>

It never happens. At fraction 0.30, validation accuracy ends at 0.238 after all 6000 steps. The full sweep is committed in `runs/sweep/` and drawn in `figures/fraction.png`:

| train fraction | grokking step |
|---|---|
| 0.20 | none within 6000 steps |
| 0.25 | none |
| 0.30 | none |
| 0.35 | none |
| 0.40 | 5050 |
| 0.45 | 3150 |
| 0.50 | 2150 |

Two patterns. More training data makes the transition earlier, because the circuit learns from more pairs at once. And at this step budget there is a hard floor: somewhere between 0.35 and 0.40, grokking stops happening in 6000 steps at all. "None" means "not within the budget", not "never": with a larger step count a small fraction can still get there, but the run gets expensive fast, which is why the preset uses 0.45.

</details>

## 3. a different seed

Same preset, same everything, one change: `--seed 1`. How much of the curve's shape is luck?

```
python grok.py --preset fast --seed 1 --out runs/local/seed1
python plot.py --run runs/local/seed1 --out figures/v1
```

<details>
<summary>answer</summary>

The shape survives, the timing does not. Seed 1 memorizes at the same step 150 and groks at step 1800, against 3150 for seed 0, a difference of 1350 steps from a one-bit change in the randomness. Both runs end at validation accuracy 1.0 with the same long flat stretch in between.

Treat exact step counts as seed-dependent decoration. The structure (memorize early, wait, jump, end at 1.0) is what reproduces, and it has held for every seed tried here.

</details>

## 4. which frequencies

The committed run puts 94 percent of its embedding power on k = 1, 3, 9, 25. With `--seed 1`, does the model pick the same four frequencies?

```
python grok.py --preset fast --seed 1 --out runs/local/seed1
python walkthrough.py --run runs/local/seed1 --a 7 --b 40
```

(The first section of the walkthrough output prints the spectrum with the dominant frequencies named.)

<details>
<summary>answer</summary>

No. Seed 1 picks k = 3, 16, 18, 20, 25, with five frequencies carrying 95 percent of the power. Only k = 3 and k = 25 overlap with seed 0's set. The `paper` preset at p = 113 picks yet another set, dominated by k = 25, 37, 41, 43.

What is stable across seeds is the mechanism and the count: every grokked run concentrates almost all of its embedding weight on about five frequencies, because addition mod p only needs a handful of circles. Which frequencies get picked is decided by the random init and is different each time. This is why the Fourier test in `test_grok.py` asserts the concentration and deliberately not the identities.

</details>
