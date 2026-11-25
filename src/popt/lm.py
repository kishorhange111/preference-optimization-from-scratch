"""Log-probabilities, sampling and the KL estimate for causal LMs with fixed-length prompts.

All prompts have the same token length P, so batches need no padding on the left. A
completion ends at the first EOS. The EOS itself counts as a completion token, and
everything after it is masked out.
"""

from __future__ import annotations

import torch


def completion_mask(completions: torch.Tensor, eos: int) -> torch.Tensor:
    """1 up to and including the first EOS, 0 after it. ``completions (b, T)``."""
    is_eos = (completions == eos).int()
    after = (is_eos.cumsum(1) - is_eos) > 0  # strictly after the first EOS
    return (~after).float()


def seq_logprobs(model, prompts, completions, mask) -> torch.Tensor:
    """Σ_t log π(c_t | prompt, c_<t) over masked completion tokens -> ``(b,)``, in fp32."""
    ids = torch.cat([prompts, completions], 1)
    logits = model(ids).logits[:, prompts.shape[1] - 1 : -1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, completions[..., None])[..., 0]
    return (lp * mask).sum(1)


@torch.no_grad()
def sample(model, prompts, max_new: int, eos: int, temperature: float = 1.0, generator=None):
    """Ancestral sampling with a KV cache. Returns ``(completions (b, max_new), mask)``."""
    out = model(prompts, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1]
    toks = []
    done = torch.zeros(prompts.shape[0], dtype=torch.bool, device=prompts.device)
    for _ in range(max_new):
        probs = torch.softmax(logits.float() / temperature, -1)
        t = torch.multinomial(probs, 1, generator=generator)[:, 0]
        t = torch.where(done, torch.full_like(t, eos), t)
        toks.append(t)
        done |= t == eos
        if done.all():
            break
        out = model(t[:, None], past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1]
    c = torch.stack(toks, 1)
    if c.shape[1] < max_new:
        pad = torch.full((c.shape[0], max_new - c.shape[1]), eos, device=c.device, dtype=c.dtype)
        c = torch.cat([c, pad], 1)
    return c, completion_mask(c, eos)


@torch.no_grad()
def sampled_kl(policy, ref, prompts, completions, mask) -> torch.Tensor:
    """Per-sample Monte-Carlo estimate of KL(π‖π_ref) = E_{y∼π}[log π(y) − log π_ref(y)].

    The completions must have been sampled from ``policy``. The mean over samples is
    unbiased; individual terms can be negative.
    """
    return seq_logprobs(policy, prompts, completions, mask) - seq_logprobs(
        ref, prompts, completions, mask
    )
