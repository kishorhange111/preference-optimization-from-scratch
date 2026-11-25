import json
import math

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from popt import losses
from popt.lm import completion_mask, sample, sampled_kl, seq_logprobs
from popt.train import build_pairs, evaluate, preference_accuracy, report, subset, train

EOS = 63


def tiny(seed=0):
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=64, n_positions=64, n_embd=32, n_layer=2, n_head=2)
    return GPT2LMHeadModel(cfg).eval()


def rand_batch(n=16, seed=0, **kw):
    g = torch.Generator().manual_seed(seed)
    f = lambda: torch.randn(n, generator=g) * 3
    return losses.Batch(f(), f(), f(), f(), torch.full((n,), 10.0), torch.full((n,), 5.0), **kw)


# ---------------------------------------------------------------- losses


def test_dpo_at_reference_is_log2_and_gradient_matches_closed_form():
    b = rand_batch()
    b0 = losses.Batch(b.rw, b.rl, b.rw, b.rl, b.nw, b.nl)  # policy == reference
    assert losses.dpo(b0, 0.1).item() == pytest.approx(math.log(2))
    pw = b.pw.clone().requires_grad_()
    beta = 0.3
    losses.dpo(losses.Batch(pw, b.pl, b.rw, b.rl, b.nw, b.nl), beta).backward()
    h = losses.margin(b)
    torch.testing.assert_close(pw.grad, -beta * torch.sigmoid(-beta * h) / len(h))


def test_cdpo_reduces_to_dpo_and_has_a_finite_optimum():
    b = rand_batch()
    assert losses.cdpo(b, 0.2, 0.0).item() == pytest.approx(losses.dpo(b, 0.2).item())
    beta, eps = 0.5, 0.2
    hs = torch.linspace(-20, 40, 60001)
    z = beta * hs
    per = -(1 - eps) * torch.nn.functional.logsigmoid(z) - eps * torch.nn.functional.logsigmoid(-z)
    assert hs[per.argmin()].item() == pytest.approx(math.log((1 - eps) / eps) / beta, abs=2e-3)


def test_ipo_is_minimised_at_target_margin():
    tau = 0.25
    n = torch.ones(1)
    b = losses.Batch(torch.tensor([2.0]), torch.zeros(1), torch.zeros(1), torch.zeros(1), n, n)
    assert losses.ipo(b, tau).item() == pytest.approx(0.0)  # h = 2 = 1/(2τ)


def test_simpo_ignores_reference_and_normalises_by_length():
    b = rand_batch()
    b2 = losses.Batch(b.pw, b.pl, b.rw + 5, b.rl - 7, b.nw, b.nl)
    assert losses.simpo(b, 2.0, 1.0).item() == pytest.approx(losses.simpo(b2, 2.0, 1.0).item())
    # doubling a completion's length and its log-prob leaves the reward unchanged
    b3 = losses.Batch(2 * b.pw, b.pl, b.rw, b.rl, 2 * b.nw, b.nl)
    assert losses.simpo(b, 2.0, 1.0).item() == pytest.approx(losses.simpo(b3, 2.0, 1.0).item())
    assert not losses.uses_reference("simpo") and losses.uses_reference("dpo")


def test_dpo_drives_margin_to_infinity_with_deterministic_labels_but_ipo_does_not():
    """IPO's motivation: with h as a free parameter, DPO keeps increasing it, IPO stops."""
    out = {}
    for method, hp in (("dpo", {"beta": 0.1}), ("ipo", {"tau": 0.1})):
        h = torch.zeros(1, requires_grad=True)
        opt = torch.optim.SGD([h], lr=0.2)
        z, n = torch.zeros(1), torch.ones(1)
        for _ in range(5000):
            opt.zero_grad()
            losses.loss(method, losses.Batch(h, z, z, z, n, n), **hp).backward()
            opt.step()
        out[method] = h.item()
    assert out["ipo"] == pytest.approx(1 / (2 * 0.1), rel=1e-3)
    assert out["dpo"] > 3 * out["ipo"]


# ---------------------------------------------------------------- lm helpers


def test_completion_mask_keeps_first_eos_only():
    c = torch.tensor([[1, 2, EOS, 4, EOS], [1, 2, 3, 4, 5]])
    assert completion_mask(c, EOS).tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]


def test_seq_logprobs_matches_token_by_token():
    m = tiny()
    p = torch.randint(0, 60, (3, 4))
    c = torch.randint(0, 60, (3, 5))
    mask = torch.tensor([[1, 1, 1, 0, 0], [1] * 5, [1, 0, 0, 0, 0]]).float()
    got = seq_logprobs(m, p, c, mask)
    for i in range(3):
        want = 0.0
        for t in range(int(mask[i].sum())):
            ids = torch.cat([p[i], c[i, :t]])[None]
            want += torch.log_softmax(m(ids).logits[0, -1], -1)[c[i, t]].item()
        assert got[i].item() == pytest.approx(want, abs=1e-4)


def test_sampling_with_cache_matches_model_distribution():
    """The first sampled token's empirical frequencies match softmax(logits)."""
    m = tiny()
    p = torch.randint(0, 60, (1, 4)).repeat(4000, 1)
    c, mask = sample(m, p, 2, EOS, generator=torch.Generator().manual_seed(0))
    probs = torch.softmax(m(p[:1]).logits[0, -1], -1)
    freq = torch.bincount(c[:, 0], minlength=64).float() / len(c)
    assert (freq - probs).abs().max() < 0.02
    assert mask.shape == c.shape


def test_kl_estimate_is_zero_for_identical_models():
    m = tiny()
    p = torch.randint(0, 60, (8, 4))
    c, mask = sample(m, p, 6, EOS)
    assert sampled_kl(m, m, p, c, mask).abs().max() < 1e-5


# ---------------------------------------------------------------- pipeline


def toy_reward(prompts, completions, mask):
    """Reward = mean token id of the completion: continuous, so pairs are rarely tied."""
    return (completions.float() * mask).sum(1) / mask.sum(1)


@pytest.mark.parametrize(
    "method,hp", [("dpo", {"beta": 0.5}), ("simpo", {"beta": 2.0, "gamma": 1.0})]
)
def test_training_on_toy_preferences_raises_reward_and_accuracy(method, hp, monkeypatch):
    import popt.train as T

    monkeypatch.setattr(T, "MAX_NEW", 8)
    ref = tiny()
    prompts = torch.randint(0, 60, (256, 4))
    pairs = build_pairs(ref, prompts, toy_reward, EOS, "cpu", bs=128)
    assert (pairs["reward_w"] >= pairs["reward_l"]).all()
    tr, va = subset(pairs, slice(0, 192)), subset(pairs, slice(192, 256))
    import copy

    pol = copy.deepcopy(ref)
    before = preference_accuracy(pol, va, method, "cpu")
    hist = train(pol, tr, method, hp, "cpu", epochs=6, bs=32, lr=3e-3)
    after = preference_accuracy(pol, va, method, "cpu")
    assert hist[-1]["loss"] < hist[0]["loss"]
    assert after["pref_acc"] > before["pref_acc"] + 0.1
    ev0 = evaluate(ref, ref, prompts[:128], toy_reward, EOS, "cpu", bs=64)
    ev1 = evaluate(pol, ref, prompts[:128], toy_reward, EOS, "cpu", bs=64)
    assert ev1["reward"] > ev0["reward"]
    assert abs(ev0["kl"]) < 1e-4 and ev1["kl"] > 0.05


def test_report_renders(tmp_path):
    rec = {
        "name": "dpo β=0.1",
        "reward": 1.0,
        "reward_se": 0.1,
        "frac_positive": 0.6,
        "kl": 2.0,
        "kl_se": 0.2,
        "pref_acc": 0.7,
        "length": 40.0,
    }
    text = report([rec, {**rec, "name": "sft", "pref_acc": None}])
    assert "dpo β=0.1" in text and "| – |" in text
    json.dumps(rec)
