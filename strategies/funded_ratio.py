"""Funded-ratio rule: every year withdraw wealth / (remaining years + 1) (an amortisation schedule).

Same 50/50 allocation as Bengen, so only the withdrawal rule differs. The simulator clips the rate to [3%, 12%].
"""

from strategies.bengen import Strategy as Bengen


class Strategy(Bengen):
    def act(self, obs):
        weights, _ = super().act(obs)
        return weights, obs.wealth / (obs.n_years - obs.year + 1)  # +1: the last year keeps a remainder, else margin = 0 counts as ruin
