# preference-optimization-from-scratch

[![CI](https://github.com/kishorhange111/preference-optimization-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/preference-optimization-from-scratch/actions/workflows/ci.yml)

Offline preference optimisation: **DPO**, **conservative DPO**, **IPO** and **SimPO**, each
written from its paper. They are compared on the task where the true reward is known: the
DPO paper's controlled-sentiment generation on IMDb. That makes it possible to measure what
each objective actually trades. The trade is reward against KL from the reference, and
preference accuracy against reward.

The companion repos do on-policy RL on the same kind of question:
[grpo-from-scratch](https://github.com/kishorhange111/grpo-from-scratch) and
[rlvr-grpo-from-scratch](https://github.com/kishorhange111/rlvr-grpo-from-scratch).

## Objectives ([`losses.py`](src/popt/losses.py))

With `h = log π(w)/π_ref(w) − log π(l)/π_ref(l)`, the policy's log-ratio margin between the
chosen and rejected completion:

| method | loss | what it fixes |
|---|---|---|
| DPO | `−log σ(β h)` | RLHF's KL-regularised optimum, without a reward model or sampling |
| cDPO | `−(1−ε) log σ(β h) − ε log σ(−β h)` | assumes labels flip with prob. ε: finite optimum `h* = log((1−ε)/ε)/β` |
| IPO | `(h − 1/(2τ))²` | with deterministic preferences DPO pushes `h → ∞` whatever β is; IPO regresses h onto a fixed target |
| SimPO | `−log σ(β (log π(w)/\|w\| − log π(l)/\|l\|) − γ)` | no reference model; length-normalised reward with margin γ |

## Experiment ([`train.py`](src/popt/train.py))

- **Policy and reference:** `lvwerra/gpt2-imdb`, a GPT-2 (124M) fine-tuned on IMDb reviews.
- **Ground-truth reward:** the positive-minus-negative logit of `lvwerra/distilbert-imdb`,
  a sentiment classifier.
- **Prompts:** the first 8 tokens of IMDb reviews.
- **Preference data:** for each of 8,192 training prompts, two completions (up to 48
  tokens) are sampled from the reference. The classifier picks the winner. Another 512
  pairs are held out. These are on-policy pairs with deterministic labels, the regime that
  IPO says DPO overfits in.
- **Training:** full fine-tuning, AdamW at lr 5e-6, batch 64, 2 epochs, fp16 autocast.
- **Evaluation:** one completion per 512 IMDb *test* prompts, sampled at temperature 1. The
  run reports:
  - the mean reward and the fraction classified positive;
  - **KL(π‖π_ref)**, estimated without bias from those same samples as
    `E[log π(y) − log π_ref(y)]`;
  - **held-out preference accuracy**: how often the method's own implicit reward ranks the
    held-out chosen completion higher.

## Tests

There are 12, on CPU, with a tiny random GPT-2.

Losses:
- DPO equals log 2 at the reference, and its gradient matches `−β σ(−βh)`.
- cDPO with ε = 0 is DPO, and its numerical minimiser is `log((1−ε)/ε)/β`.
- IPO is zero at its target margin.
- SimPO ignores the reference and is invariant to length scaling.
- With deterministic labels, gradient descent on a free margin keeps growing under DPO,
  while IPO converges to `1/(2τ)`.

LM helpers:
- `seq_logprobs` matches a token-by-token loop.
- Only the first EOS is kept in the mask.
- KV-cached sampling matches the model's distribution.
- The KL estimate is 0 for identical models.

End to end:
- On a toy reward, DPO and SimPO training raise held-out preference accuracy and sampled
  reward, and move KL off zero.

```
pytest -q                                                  # 12 tests, CPU
python -m popt.train --stage pairs --out results --device cuda
python -m popt.train --stage sweep --out results --device cuda   # resumable
```

## Results (Tesla T4, Colab)

There are 8,192 training pairs, 512 held-out pairs and 512 test prompts. The mean reward gap
between chosen and rejected completions is 2.88. Every config gets 2 epochs at lr 5e-6,
which takes about 3.5 minutes. Rewards and KLs are means ± standard error over the 512 test
samples. The full table is in [`results/summary.md`](results/summary.md) and the raw records,
with loss curves, in [`results/runs.jsonl`](results/runs.jsonl).

| config | reward | P(positive) | KL(π‖π_ref) | held-out pref. acc. |
|---|---|---|---|---|
| reference (sft) | 0.47 ± 0.15 | 0.559 | 0 | – |
| dpo β=0.03 | **4.96** ± 0.06 | 0.980 | 17.21 | 0.686 |
| dpo β=0.1 | 4.35 ± 0.08 | 0.953 | 6.90 | 0.732 |
| dpo β=0.3 | 3.49 ± 0.11 | 0.893 | 3.28 | 0.727 |
| dpo β=1.0 | 3.07 ± 0.11 | 0.877 | 2.34 | 0.725 |
| cdpo β=0.1 ε=0.2 | 3.72 ± 0.10 | 0.924 | 3.88 | 0.730 |
| cdpo β=0.3 ε=0.2 | 2.65 ± 0.12 | 0.840 | 1.78 | 0.709 |
| ipo τ=0.03 | 4.00 ± 0.09 | 0.938 | 4.59 | **0.740** |
| ipo τ=0.1 | 2.35 ± 0.13 | 0.793 | 1.53 | 0.701 |
| ipo τ=0.3 | 1.25 ± 0.14 | 0.674 | 0.86 | 0.584 |
| ipo τ=1.0 | 0.65 ± 0.15 | 0.592 | 0.65 | 0.525 |
| simpo β=1.0 γ=0.5 | 4.80 ± 0.07 | 0.969 | 23.39 | 0.590 |
| simpo β=2.5 γ=1.25 | 4.43 ± 0.09 | 0.939 | 22.34 | 0.578 |
| simpo β=5.0 γ=2.5 | 4.31 ± 0.08 | 0.943 | 22.51 | 0.576 |

What the numbers show:
- **β and τ trade reward for KL, as in the DPO paper's frontier.** Lowering DPO's β from 1.0
  to 0.03 raises reward from 3.07 to 4.96 and KL from 2.3 to 17.2. IPO's τ does the same
  over a lower range.
- **At matched KL the objectives sit close together.** IPO τ=0.03 reaches 4.00 at KL 4.59.
  On the straight line between DPO β=0.3 and β=0.1, DPO would reach about 3.80 at that KL.
  cDPO β=0.1 reaches 3.72 at KL 3.88, against about 3.63 on the same line. These gaps are about
  2 and 1 standard errors, from one seed each, so this is not evidence that one objective
  wins.
- **SimPO is far less KL-efficient here.** It has no reference model in its loss, and all
  three configs drift to KL ≈ 22–23. They get less reward (4.3–4.8) than DPO β=0.03 gets at
  KL 17.
- **Held-out preference accuracy does not track reward.** DPO β=0.03 has the most reward but
  only 0.686 accuracy, below β=0.1–1.0 (0.725–0.732). The accuracy here is measured with the
  reference-based implicit reward `(log π − log π_ref)(chosen) − (…)(rejected)`. SimPO never
  optimises that quantity, so its 0.58 is not a fair score for it.
- **IPO stays far from its target margin in 2 epochs.** The loss pulls the margin towards
  `1/(2τ)`, but the held-out margins are 3.6, 1.2, 0.40 and 0.13 against targets of 16.7,
  5, 1.7 and 0.5. With strong regularisation (τ=0.3 and 1.0), IPO barely moves from the
  reference.
