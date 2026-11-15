"""Guardrails (Guyton-Klinger style): start at 4%, keep last year's withdrawal, adjust only at the guardrails.

If last year's withdrawal is above 120% of the initial 4% rate relative to current wealth, cut it 10%;
if below 80%, raise it 10%. Wealth is already in real terms, so no inflation step. Same 50/50 allocation as Bengen.
"""

import torch

from strategies.bengen import Strategy as Bengen, WITHDRAWAL_RATE

LOWER, UPPER, STEP = 0.8, 1.2, 0.10


class Strategy(Bengen):
    def act(self, obs):
        weights, _ = super().act(obs)
        if obs.year == 0:
            return weights, WITHDRAWAL_RATE
        prev = obs.withdrawal_history[-1]  # (N,)
        ratio = prev / obs.wealth.clamp(min=1e-9) / WITHDRAWAL_RATE
        rate = torch.where(ratio > UPPER, prev * (1 - STEP), torch.where(ratio < LOWER, prev * (1 + STEP), prev))
        return weights, rate
