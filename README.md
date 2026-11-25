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

**Results are pending** from the Colab run.
