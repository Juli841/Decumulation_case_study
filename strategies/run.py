"""Submission: wealth-dependent withdrawal rate + allocation tilted by hand-built market features.

    rate    = c * wealth / (years_left + 1)      # funded ratio, c learned (c = 1 is the plain rule)
    weights = softmax(allocation_logits @ [1, term_spread, log_vix, credit_spread])

The features use spreads and volatility, not yield levels, because validation starts at lower
rates. They are standardised with train statistics. Trained like bengen_optimised (gradient
ascent on a smoothed J) plus an L2 penalty on the feature coefficients and a safety margin
(MARGIN) in the ruin term. No other files changed; strategies/funded_ratio.py, guardrails.py
and my_frontier.py are extra benchmarks and are not used by this strategy.
"""

import torch

from utils import DTYPE, evaluate, simulate

STEPS = 300
LEARNING_RATE = 0.05
TAU_START, TAU_END = 0.1, 0.005
L2 = 0.01
MARGIN = 0.2  # train-time buffer: a path counts as ruined when its margin drops below this (fraction of initial capital)


class Strategy:
    def _raw(self, m, names):
        """m: (..., V) market values -> (..., 3) raw features: term spread, log VIX, credit spread."""
        col = lambda name: m[..., names.index(name)]
        return torch.stack([col("US_ZC_10Y") - col("US_ZC_2Y"),
                            torch.log(col("US_VIX")),
                            col("US_GZ_SPREAD")], dim=-1)

    def features(self, obs):
        if obs.year == 0:  # no history yet: 0 is the train average once standardised
            return torch.zeros(obs.wealth.shape[0], 3, dtype=DTYPE)
        raw = self._raw(obs.market_history[-1], obs.market_names)   # latest year, (N, 3)
        return (raw - self.mu) / self.sd

    def fit(self, train, cfg):
        self.cfg = cfg
        raw = self._raw(train.market, train.market_names)   # (T, N, 3), train only
        self.mu = raw.mean(dim=(0, 1))                       # (3,)
        self.sd = raw.std(dim=(0, 1))                        # (3,)

        rows = train.n_assets
        cols = 1 + raw.shape[-1]  # bias + features
        self.allocation_logits = torch.zeros(rows, cols, dtype=DTYPE, requires_grad=True)  # equal weights
        self.log_c = torch.zeros(1, dtype=DTYPE, requires_grad=True)  # c = exp(log_c), starts at the plain funded-ratio rule
        optimizer = torch.optim.Adam([self.allocation_logits, self.log_c], lr=LEARNING_RATE)

        for step in range(STEPS):
            tau = TAU_START * (TAU_END / TAU_START) ** (step / (STEPS - 1))
            out = simulate(self, train, cfg)
            lowest_margin = out["margins"].min(dim=0).values  # <= 0 means ruined
            smooth_ruin_prob = torch.sigmoid(-(lowest_margin - MARGIN) / tau).mean()
            smooth_j = cfg.alpha * out["norm_wd"].mean() - (1 - cfg.alpha) * smooth_ruin_prob

            loss = -smooth_j + L2 * (self.allocation_logits[:, 1:] ** 2).sum()  # penalise features, not the bias

            optimizer.zero_grad()
            loss.backward()  # minimising -J maximises J
            optimizer.step()

            if (step + 1) % 100 == 0:
                m = evaluate(self, train, cfg)
                print(f"  step {step + 1}: train J {m['J']:.4f}, ruin {m['ruin_prob']:.4f}, norm_wd {m['norm_wd']:.4f}")

    def act(self, obs):
        x = torch.cat([torch.ones(obs.wealth.shape[0], 1, dtype=DTYPE), self.features(obs)], dim=1)  # (N, 1 + k)
        weights = torch.softmax(x @ self.allocation_logits.T, dim=1)  # (N, A)
        rate = self.log_c.exp() * obs.wealth / (obs.n_years - obs.year + 1)  # (N,), the simulator clips to [wd_min, wd_max]
        return weights, rate
