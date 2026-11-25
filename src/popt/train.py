"""Controlled-sentiment generation (the DPO paper's IMDb experiment), with four objectives.

    python -m popt.train --stage pairs --out results --device cuda   # build the preference data
    python -m popt.train --stage sweep --out results --device cuda   # train + eval every config
    python -m popt.train --stage report --out results

Setup:
- **Policy and reference:** ``lvwerra/gpt2-imdb``, a GPT-2 fine-tuned on IMDb reviews.
- **Ground-truth reward:** ``lvwerra/distilbert-imdb``, a sentiment classifier. The reward is
  its positive-minus-negative logit on prompt + completion.
- **Prompts:** the first 8 tokens of IMDb reviews.
- **Pairs:** two completions sampled from the reference for each training prompt, labelled
  by the classifier (the higher reward is "chosen"). The labels are therefore deterministic.
- **Evaluation:** one completion per held-out test prompt, sampled at temperature 1. The run
  records the mean reward, the fraction classified positive, and KL(π‖π_ref) estimated from
  the same samples. It also records preference accuracy on held-out pairs: the fraction
  where the model's implicit reward ranks the chosen completion higher.

Every finished config is appended to ``<out>/runs.jsonl`` and skipped on restart, so a sweep
can be spread over several short GPU sessions.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import torch

from . import losses
from .lm import completion_mask, sample, sampled_kl, seq_logprobs

POLICY = "lvwerra/gpt2-imdb"
CLASSIFIER = "lvwerra/distilbert-imdb"
PROMPT_TOKENS = 8
MAX_NEW = 48

# (name, method, hyper-parameters)
GRID = (
    [("sft", None, {})]
    + [(f"dpo β={b}", "dpo", {"beta": b}) for b in (0.03, 0.1, 0.3, 1.0)]
    + [(f"cdpo β={b} ε=0.2", "cdpo", {"beta": b, "eps": 0.2}) for b in (0.1, 0.3)]
    + [(f"ipo τ={t}", "ipo", {"tau": t}) for t in (0.03, 0.1, 0.3, 1.0)]
    + [(f"simpo β={b} γ={b / 2:g}", "simpo", {"beta": b, "gamma": b / 2}) for b in (1.0, 2.5, 5.0)]
)


# ---------------------------------------------------------------- data


def load_prompts(tok, split: str, n: int, seed: int) -> torch.Tensor:
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb", split=split).shuffle(seed=seed)
    out = []
    for text in ds["text"]:
        ids = tok(text.replace("<br />", " ").strip())["input_ids"]
        if len(ids) >= PROMPT_TOKENS:
            out.append(ids[:PROMPT_TOKENS])
        if len(out) == n:
            break
    return torch.tensor(out)


class Classifier:
    """Ground-truth reward: the sentiment classifier's positive − negative logit."""

    def __init__(self, gen_tok, device):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.gen_tok, self.device = gen_tok, device
        self.tok = AutoTokenizer.from_pretrained(CLASSIFIER)
        self.model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER).to(device)
        self.model.eval()
        labels = {v.upper(): int(k) for k, v in self.model.config.id2label.items()}
        self.pos, self.neg = labels.get("POSITIVE", 1), labels.get("NEGATIVE", 0)

    def texts(self, prompts, completions, mask):
        n = mask.sum(1).long().tolist()
        return [
            self.gen_tok.decode(torch.cat([p, c[:k]]).tolist(), skip_special_tokens=True)
            for p, c, k in zip(prompts.cpu(), completions.cpu(), n, strict=True)
        ]

    @torch.no_grad()
    def __call__(self, prompts, completions, mask, bs: int = 128) -> torch.Tensor:
        texts = self.texts(prompts, completions, mask)
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(
                texts[i : i + bs], return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            logits = self.model(**enc).logits.float()
            out.append(logits[:, self.pos] - logits[:, self.neg])
        return torch.cat(out).cpu()


@torch.no_grad()
def batched_logprobs(model, prompts, comps, masks, bs=64) -> torch.Tensor:
    return torch.cat(
        [
            seq_logprobs(model, prompts[i : i + bs], comps[i : i + bs], masks[i : i + bs]).cpu()
            for i in range(0, len(prompts), bs)
        ]
    )


def build_pairs(ref, prompts, reward_fn, eos, device, bs=128, seed=0) -> dict:
    """Sample two completions per prompt from the reference and label them with the reward."""
    g = torch.Generator(device=device).manual_seed(seed)
    ca, cb, ra, rb = [], [], [], []
    for i in range(0, len(prompts), bs):
        p = prompts[i : i + bs].to(device)
        for comps, rewards in ((ca, ra), (cb, rb)):
            c, m = sample(ref, p, MAX_NEW, eos, generator=g)
            comps.append(c.cpu())
            rewards.append(reward_fn(p, c, m))
    ca, cb, ra, rb = (torch.cat(x) for x in (ca, cb, ra, rb))
    a_wins = (ra >= rb)[:, None]
    cw, cl = torch.where(a_wins, ca, cb), torch.where(a_wins, cb, ca)
    rw, rl = torch.maximum(ra, rb), torch.minimum(ra, rb)
    mw, ml = completion_mask(cw, eos), completion_mask(cl, eos)
    pd = prompts.to(device)
    return {
        "prompts": prompts,
        "cw": cw,
        "cl": cl,
        "mw": mw,
        "ml": ml,
        "reward_w": rw,
        "reward_l": rl,
        "ref_w": batched_logprobs(ref, pd, cw.to(device), mw.to(device)),
        "ref_l": batched_logprobs(ref, pd, cl.to(device), ml.to(device)),
    }


def subset(pairs: dict, idx) -> dict:
    return {k: v[idx] for k, v in pairs.items()}


# ---------------------------------------------------------------- train / eval


def pair_batch(policy, d: dict, device) -> losses.Batch:
    p = d["prompts"].to(device)
    mw, ml = d["mw"].to(device), d["ml"].to(device)
    return losses.Batch(
        pw=seq_logprobs(policy, p, d["cw"].to(device), mw),
        pl=seq_logprobs(policy, p, d["cl"].to(device), ml),
        rw=d["ref_w"].to(device),
        rl=d["ref_l"].to(device),
        nw=mw.sum(1),
        nl=ml.sum(1),
    )


def train(policy, pairs: dict, method: str, hp: dict, device, epochs=2, bs=64, lr=5e-6, seed=0):
    """Full fine-tuning with AdamW, linear warm-up over 10% of steps, then constant."""
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    n = len(pairs["prompts"])
    steps = epochs * (n // bs)
    warm = max(1, steps // 10)
    g = torch.Generator().manual_seed(seed)
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    hist, step = [], 0
    policy.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n - bs + 1, bs):
            step += 1
            for grp in opt.param_groups:
                grp["lr"] = lr * min(1.0, step / warm)
            with torch.autocast("cuda", torch.float16, enabled=amp):
                b = pair_batch(policy, subset(pairs, perm[i : i + bs]), device)
                loss = losses.loss(method, b, **hp)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if step % 10 == 0 or step == steps:
                h = losses.margin(b).detach()
                hist.append({"step": step, "loss": loss.item(), "margin": h.mean().item()})
    policy.eval()
    return hist


@torch.no_grad()
def preference_accuracy(policy, pairs: dict, method: str | None, device, bs=64) -> dict:
    """Held-out pairs: how often the implicit reward prefers the chosen completion."""
    hs = []
    for i in range(0, len(pairs["prompts"]), bs):
        b = pair_batch(policy, subset(pairs, slice(i, i + bs)), device)
        if method == "simpo":
            hs.append((b.pw / b.nw - b.pl / b.nl).cpu())
        else:
            hs.append(losses.margin(b).cpu())
    h = torch.cat(hs)
    return {"pref_acc": float((h > 0).float().mean()), "val_margin": float(h.mean())}


@torch.no_grad()
def evaluate(policy, ref, prompts, reward_fn, eos, device, bs=128, seed=1234) -> dict:
    g = torch.Generator(device=device).manual_seed(seed)
    rewards, kls, lens = [], [], []
    for i in range(0, len(prompts), bs):
        p = prompts[i : i + bs].to(device)
        c, m = sample(policy, p, MAX_NEW, eos, generator=g)
        rewards.append(reward_fn(p, c, m))
        kls.append(sampled_kl(policy, ref, p, c, m).cpu())
        lens.append(m.sum(1).cpu())
    r, k, n = torch.cat(rewards), torch.cat(kls), torch.cat(lens)
    se = lambda x: float(x.std() / math.sqrt(len(x)))
    return {
        "reward": float(r.mean()),
        "reward_se": se(r),
        "frac_positive": float((r > 0).float().mean()),
        "kl": float(k.mean()),
        "kl_se": se(k),
        "length": float(n.mean()),
    }


# ---------------------------------------------------------------- report


def report(recs: list[dict], pairs_info: dict | None = None) -> str:
    lines = []
    if pairs_info:
        lines += [
            f"{pairs_info['n_train']} training pairs, {pairs_info['n_val']} held-out pairs, "
            f"{pairs_info['n_test']} test prompts. Mean reward gap chosen − rejected: "
            f"{pairs_info['gap']:.2f}.",
            "",
        ]
    lines += [
        "| config | reward | P(positive) | KL(π‖π_ref) | held-out pref. acc. | length |",
        "|---|---|---|---|---|---|",
    ]
    for r in recs:
        acc = f"{r['pref_acc']:.3f}" if r.get("pref_acc") is not None else "–"
        lines.append(
            f"| {r['name']} | {r['reward']:.2f} ± {r['reward_se']:.2f} | "
            f"{r['frac_positive']:.3f} | {r['kl']:.2f} ± {r['kl_se']:.2f} | {acc} | "
            f"{r['length']:.1f} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- CLI


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pairs", "sweep", "report", "all"], required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-train", type=int, default=8192)
    ap.add_argument("--n-val", type=int, default=512)
    ap.add_argument("--n-test", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--only", nargs="*", default=None, help="config names to run")
    ap.add_argument("--budget-min", type=float, default=1e9, help="start no config after this")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    pairs_path, runs_path = os.path.join(a.out, "pairs.pt"), os.path.join(a.out, "runs.jsonl")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(POLICY)
    eos = tok.eos_token_id

    def load_policy():
        return AutoModelForCausalLM.from_pretrained(POLICY).to(a.device).eval()

    if a.stage in ("pairs", "all") and not os.path.exists(pairs_path):
        ref = load_policy()
        clf = Classifier(tok, a.device)
        train_prompts = load_prompts(tok, "train", a.n_train + a.n_val, seed=0)
        test_prompts = load_prompts(tok, "test", a.n_test, seed=0)
        pairs = build_pairs(ref, train_prompts, clf, eos, a.device)
        torch.save({"pairs": pairs, "test_prompts": test_prompts, "n_val": a.n_val}, pairs_path)
        gap = float((pairs["reward_w"] - pairs["reward_l"]).mean())
        print(f"[pairs] {len(train_prompts)} pairs, mean reward gap {gap:.2f}", flush=True)

    if a.stage in ("sweep", "all"):
        blob = torch.load(pairs_path)
        pairs, test_prompts, n_val = blob["pairs"], blob["test_prompts"], blob["n_val"]
        n = len(pairs["prompts"])
        tr, va = subset(pairs, slice(0, n - n_val)), subset(pairs, slice(n - n_val, n))
        done = (
            {json.loads(x)["name"] for x in open(runs_path)} if os.path.exists(runs_path) else set()
        )
        ref = load_policy()
        clf = Classifier(tok, a.device)
        for name, method, hp in GRID:
            if name in done or (a.only and name not in a.only):
                continue
            if (time.time() - t0) / 60 > a.budget_min:
                print("[sweep] budget reached", flush=True)
                break
            t1 = time.time()
            policy = copy.deepcopy(ref)
            hist = (
                train(policy, tr, method, hp, a.device, epochs=a.epochs, lr=a.lr) if method else []
            )
            rec = {"name": name, "method": method or "sft", **hp, "epochs": a.epochs, "lr": a.lr}
            rec.update(evaluate(policy, ref, test_prompts, clf, eos, a.device))
            rec.update(preference_accuracy(policy, va, method, a.device) if method else {})
            rec.update({"history": hist, "seconds": time.time() - t1})
            with open(runs_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(
                f"[{time.strftime('%H:%M:%S')}] {name}: reward {rec['reward']:.2f} "
                f"kl {rec['kl']:.2f} acc {rec.get('pref_acc')} ({rec['seconds']:.0f}s)",
                flush=True,
            )
            del policy
            if a.device == "cuda":
                torch.cuda.empty_cache()

    if os.path.exists(runs_path):
        recs = [json.loads(x) for x in open(runs_path)]
        info = None
        if os.path.exists(pairs_path):
            blob = torch.load(pairs_path)
            p = blob["pairs"]
            info = {
                "n_train": len(p["prompts"]) - blob["n_val"],
                "n_val": blob["n_val"],
                "n_test": len(blob["test_prompts"]),
                "gap": float((p["reward_w"] - p["reward_l"]).mean()),
            }
        text = report(recs, info)
        with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
            f.write(text)
        print(text)


if __name__ == "__main__":
    main()
