"""Offline preference objectives, each written from its paper.

Every loss takes, per pair, the policy's and the reference's log-probabilities of the chosen
(w) and rejected (l) completions, summed over completion tokens, plus the token counts.

    h = (log π(w) − log π_ref(w)) − (log π(l) − log π_ref(l))       (log-ratio margin)

DPO (Rafailov et al. 2023). Under the Bradley–Terry model with the KL-regularised RLHF
optimum, the implicit reward is r(y) = β log π(y)/π_ref(y). Maximum likelihood on the
preferences gives

    L_DPO = −log σ(β h).

The gradient is −β σ(−β h) ∇h: pairs the model already ranks correctly are down-weighted.

cDPO ("conservative" DPO, Mitchell 2023). This assumes each label is flipped with
probability ε:

    L = −(1 − ε) log σ(β h) − ε log σ(−β h).

Its minimiser is a finite margin, h* = log((1 − ε)/ε)/β, rather than h → ∞.

IPO (Azar et al. 2023). When preferences are (near-)deterministic, DPO's optimum pushes
h → ∞ whatever β is, so the KL regularisation is lost. IPO instead regresses the margin
onto a fixed target:

    L_IPO = (h − 1/(2τ))².

SimPO (Meng et al. 2024). This drops the reference model and uses the *length-normalised*
log-likelihood as the reward, with a target margin γ:

    L_SimPO = −log σ(β (log π(w)/|w| − log π(l)/|l|) − γ).

There is no KL anchor at all. The only thing keeping the policy near its start is the
optimiser (few steps, small lr).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Batch:
    pw: torch.Tensor  # (n,) Σ log π(chosen)
    pl: torch.Tensor  # (n,) Σ log π(rejected)
    rw: torch.Tensor  # (n,) Σ log π_ref(chosen)
    rl: torch.Tensor  # (n,) Σ log π_ref(rejected)
    nw: torch.Tensor  # (n,) chosen token count
    nl: torch.Tensor  # (n,) rejected token count


def margin(b: Batch) -> torch.Tensor:
    return (b.pw - b.rw) - (b.pl - b.rl)


def dpo(b: Batch, beta: float) -> torch.Tensor:
    return -F.logsigmoid(beta * margin(b)).mean()


def cdpo(b: Batch, beta: float, eps: float) -> torch.Tensor:
    z = beta * margin(b)
    return (-(1 - eps) * F.logsigmoid(z) - eps * F.logsigmoid(-z)).mean()


def ipo(b: Batch, tau: float) -> torch.Tensor:
    return (margin(b) - 1 / (2 * tau)).pow(2).mean()


def simpo(b: Batch, beta: float, gamma: float) -> torch.Tensor:
    z = beta * (b.pw / b.nw - b.pl / b.nl) - gamma
    return -F.logsigmoid(z).mean()


def loss(method: str, b: Batch, **hp) -> torch.Tensor:
    if method == "dpo":
        return dpo(b, hp["beta"])
    if method == "cdpo":
        return cdpo(b, hp["beta"], hp["eps"])
    if method == "ipo":
        return ipo(b, hp["tau"])
    if method == "simpo":
        return simpo(b, hp["beta"], hp["gamma"])
    raise ValueError(method)


def uses_reference(method: str) -> bool:
    return method != "simpo"
