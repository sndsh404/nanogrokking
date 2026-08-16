# References

Everything consulted while building this repo, and what was taken from where.

## Primary source

- Nanda, Chan, Lieberum, Smith and Steinhardt, "Progress Measures for
  Grokking via Mechanistic Interpretability", ICLR 2023.
  https://arxiv.org/abs/2301.05217
  The task, architecture (one layer, d_model 128, 4 heads, ReLU MLP of width
  512, no LayerNorm, learned positional embeddings, untied unembedding), and
  training setup (30 percent train split, full-batch AdamW, learning rate
  1e-3, weight decay 1.0) all come from section 3 of this paper. The Fourier
  analysis in plot.py implements the embedding analysis of section 4.1.

- Power, Burda, Edwards, Babuschkin and Misra, "Grokking: Generalization
  Beyond Overfitting on Small Algorithmic Datasets", 2022.
  https://arxiv.org/abs/2201.02177
  The original discovery of grokking. Their code repository
  (github.com/openai/grokking) no longer exists, so only the paper was
  consulted.

## Code studied but not copied

- Neel Nanda and Tom Lieberum's grokking notebook and saved runs,
  github.com/neelnanda-io/Grokking. The repository carries no license, so no
  code was taken from it. Protocol details confirmed there and adopted as
  practices: computing the loss in float64 to avoid float32 log_softmax
  underflow once the model is confident, the 10-step linear learning rate
  warmup, AdamW betas (0.9, 0.98), and weight decay applied to every
  parameter.

- ARENA 3.0, github.com/callummcdougall/ARENA_3.0, chapter 1 part 52
  (grokking and modular arithmetic). Unlicensed educational material, studied
  for how to present this topic pedagogically. Their attribution phrasing
  ("adapted from the original notebook by Neel Nanda and Tom Lieberum") is
  the model for the credit above.

## Style references (MIT licensed)

- github.com/karpathy/nanoGPT: file layout, flat config exposure, eval
  cadence decoupled from logging, README structure.
- github.com/karpathy/micrograd: minimalism as a design principle.
- github.com/karpathy/minbpe: per-file usage examples, testing against a
  trusted reference.
- github.com/karpathy/build-nanogpt: every operation written out, nothing
  hidden behind framework conveniences.
- github.com/karpathy/makemore: naming and comment tone.

## Statement of originality

All code in this repository was written for this repository, implementing the
architecture and training setup from the mathematical description in the
papers above. No file is derived from any of the repositories listed here.
Where a protocol detail (a constant, a precision choice, an optimizer
setting) was adopted from studying those repositories, it is noted above and
in a comment at the point of use.
