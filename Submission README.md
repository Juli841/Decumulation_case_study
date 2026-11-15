# Decumulation case study: submission summary

Score: `J = alpha * norm_wd - (1 - alpha) * ruin_prob`, where `norm_wd` is discounted withdrawals relative to a never-ruined 4% rule. 30 years, 7 assets. Train = 100 scenarios, validation = 80.

## 1. The submission (`strategies/run.py`)

**Withdrawals.** A funded-ratio rule with one learned multiplier:

`rate = c * wealth / (years_left + 1)`, with `c = exp(log_c)`, starting at 1.

Spend what is left spread over the remaining years. The `+1` avoids spending all of wealth in the last year, which the simulator counts as ruin.

**Allocation.** `softmax(logits @ [1, term_spread, log_vix, credit_spread])`. Features are standardised with train statistics. They are spreads and volatility, not yield levels, because yield levels differ between regimes.

**Training.** Gradient ascent on a smoothed J with annealed temperature: 300 steps, lr 0.05, L2 = 0.01 on the feature coefficients (not the bias). The ruin term uses a **0.2 safety margin**: a path counts as ruined once wealth falls below 20% of initial capital. This is a fixed design choice, not a learned parameter.

**Why this design.** Two decisions are made each year: how much to withdraw and how to invest. The strategy keeps each simple and learns very little.

- *Withdrawals* follow a rule that adapts on its own. If markets do badly, wealth is lower, so the withdrawal shrinks; if they do well, it grows. Dividing by years left means spending is spread across the horizon. The only learned part is the scale `c`, which sets how aggressive the spending is. It comes out below 1 when ruin matters more (about 0.76 at alpha 0.25) and near 1 at alpha 0.5. One parameter is hard to overfit.
- *Allocation* tilts the portfolio using three market signals (term spread, VIX, credit spread). These describe the market's mood, not its level, so they should carry over to a different rate environment. In practice the weights move only slightly with these signals; most of the portfolio is a learned near-static mix, which is why concentration is a weakness.
- *The 0.2 margin* trains the model to avoid ending a year with less than 20% of capital, not only to avoid hitting zero. Train scenarios rarely hit zero, so without a buffer the model sees no ruin and takes too much risk. The buffer acts as insurance against a worse regime than the one it trained on.

**The main difficulty.** The train and validation scenarios come from different regimes: validation has lower returns and lower rates. Anything fitted on train looked safe (zero ruin) and then failed on validation (about 30% ruin). Most of what I tried was an attempt to fix this without simply tuning to validation.

## 2. What I tried

The ideas fall into four groups.

1. **Making the inputs regime-proof:** switching from yield levels to spreads and volatility.
2. **Making the model more cautious:** the safety margin, CVaR, and scenario augmentation.
3. **Making the model simpler:** dropping the horizon feature, penalising the portfolio bias, static portfolios and baskets.
4. **Making the withdrawal rule smarter:** the funded-ratio rate, which gave the largest late gain.

To avoid fooling myself I ran a role swap (train on validation, score on train). An idea only counts if it helps in both directions. Augmentation passed in one direction and failed in the other, so it was dropped.

| Idea | Outcome                                                                                                                                                  |
|---|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Yield-level features | Overfit train; validation is a lower-rate regime (30% ruin). Switched to spreads and volatility.                                                         |
| More L2 on feature coefficients | No effect; it never touched the bias, where the portfolio lives.                                                                                         |
| CVaR ruin term | Identical to base: train margins were large, so it was inactive.                                                                                         |
| Safety margin in the ruin term | Biggest early gain. Useful range about 0.15 to 0.5; collapse at 0.7 and above. Chose 0.2 as I did not want to choose solely on validation scores.        |
| Scenario augmentation (volatility scaling) | Cut validation ruin 0.225 to 0.15, but collapsed to minimum withdrawals in a role swap (train on validation, score on train). Regime-dependent, dropped. |
| Horizon (time) feature | Fit train, no test gain; cut down as I don't want to add more parameters that would potentially cause overfitting behaviour                              |
| Bias L2, static portfolios, baskets, tilts | Diversifies (3-4 assets to 7) but costs about 0.03 J. No variant wins in both split directions. Kept the feature-only penalty.                           |
| Funded-ratio rate with learned `c` | Beat the old sigmoid-of-wealth rate at all alphas tested. **Adopted.**                                                                                   |

**What each idea means**

- **Yield-level features:** feeding the model the raw level of interest rates. Train rates are higher than validation rates, so the model learned "rates are high, so be aggressive" and that did not transfer. Spreads and volatility measure market stress, which means about the same thing in any rate environment.
- **L2 on feature coefficients:** a penalty that shrinks the weights on the market signals toward zero, to stop the model from over-reading them. It did nothing because the portfolio mix is mostly set by a separate constant term (the bias), which the penalty did not cover.
- **CVaR ruin term:** instead of the average ruin rate, optimise the average over the worst paths. It was identical to the base run because in train even the worst paths stayed far from ruin, so it never activated.
- **Safety margin:** count a path as ruined when wealth after withdrawing drops below a buffer (0.2 = 20% of starting capital), not only below zero. This gives the optimiser a ruin signal on train, where true ruin almost never happens.
- **Scenario augmentation:** create extra, more volatile training scenarios by scaling the risky assets' returns, so the model sees a harsher world. It cut validation ruin, but when trained on validation and scored on train it collapsed to minimum withdrawals. It buys protection in one regime at a large cost in another.
- **Horizon feature:** give the model the years remaining as an input, so it can change allocation and spending over time. It fit train but did not help out of sample, and it adds parameters. The funded-ratio rule already uses years left in a fixed way.
- **Bias L2, static portfolios, baskets, tilts:** attempts to fix the concentrated portfolio. Bias L2 penalises the constant term so the model spreads over all 7 assets. Static portfolios use fixed weights with no market signals. Baskets group assets into 4 buckets with learned weights. Tilts add a small number of risk-on/risk-off signals. All were diversified or simpler, but none beat the current model in both directions, and shrinking harder collapsed withdrawals.
- **Funded-ratio rate with learned `c`:** replaces a learned function of wealth with `c * wealth / (years_left + 1)`. It has fewer parameters and a sensible built-in logic, so it did better on train and on validation.

### Rule-based benchmarks: funded ratio and guardrails

I added two simple rules next to the given Bengen 4% baseline. Both use Bengen's fixed 50/50 US equity / 5Y Treasury mix, so they differ from Bengen only in how much they withdraw. Nothing in them is trained.

- **Funded ratio:** each year withdraw `wealth / (years_left + 1)`, i.e. the current wealth divided evenly over the remaining years (plus one, so the last year does not take everything). Withdrawals rise and fall with wealth, so the rule spends down the portfolio on purpose. My first version used `wealth / years_left`, which showed 71% ruin. That was a bug, not a weakness: in the final year it withdraws all wealth, so the margin is exactly zero and the simulator counts it as ruin. With the `+1` fix, validation ruin is 0.0875 and norm_wd 1.36.
- **Guardrails:** start at 4% of initial capital and adjust. If the current withdrawal is above 120% of the 4% rule measured against current wealth, cut it by 10%; if below 80%, raise it by 10%. This keeps withdrawals steady but reacts to large moves in wealth. On validation: ruin 0.0875, norm_wd 0.99, better than Bengen on both measures.

**Why I included them.**

1. **Bengen alone is a weak yardstick.** A fixed 4% rule ignores wealth, so any adaptive strategy should beat it. Beating it does not show that training adds anything. A trained model should also beat a simple adaptive rule.
2. **They check the training.** If a hand-written rule matched my model, the learning would not be earning its complexity. The comparison showed that a plain funded ratio is very strong, and tied with my model at alpha 0.25. That is a fair result to report.
3. **Funded ratio gave me the withdrawal rule.** It suggested replacing the learned sigmoid-of-wealth rate with `c * wealth / (years_left + 1)`, which fixed a rate that was hard to train and improved results.
4. **They guard against overfitting.** The rules have no fitted parameters, so they behave the same on every regime. They are a stable reference for judging my validation numbers, which I know are optimistic.

One caution on funded ratio: it ends with very little wealth (median about 0.08 of initial capital). The score gives no credit for leftover wealth, so part of its advantage comes from the metric.

## 3. Results

Validation frontier (ruin, norm_wd) for `run.py` trained per alpha: 0.0: 0.100 / 1.15; 0.25: 0.150 / 1.46; 0.5: 0.250 / 1.57; 1.0: 0.513 / 1.65. For reference: funded ratio 0.0875 / 1.36, guardrails 0.0875 / 0.99, Bengen 4% 0.225 / 0.96. See `frontier_mine.png`.

Both-direction check on the two splits (J, trained on train and scored on validation / trained on validation and scored on train):

| Strategy | alpha 0.25 | alpha 0.5 |
|---|---|---|
| Funded ratio (rule, no training) | 0.275 / 0.376 | 0.638 / 0.752 |
| Earlier `run.py` (sigmoid-of-wealth rate) | 0.203 / 0.260 | 0.586 / 0.910 |
| **Final `run.py`** | 0.253 / 0.344 | **0.658 / 0.934** |

The final `run.py` beats the earlier version in every column and no longer collapses to minimum withdrawals when trained on validation. Plain funded ratio is slightly ahead at alpha 0.25; `run.py` is ahead at alpha 0.5. Funded ratio and guardrails have zero ruin on train but 0.0875 on validation, which shows how much harder the validation regime is.

## 4. Caveats

- Validation was reused a couple of  times, so its numbers are optimistic. Most runs are single seed on 80 scenarios, so small J differences are noise.
- The portfolio is about 80% risky assets (mostly EM equity, REIT, US equity, 30Y Treasury; 5Y, TIPS and cash held at 0). Concentration is a known weakness, not fixed.
- Spend-down rules end with thin wealth (median about 0.08 to 0.4 of initial capital). J gives no credit for leftover wealth, so part of the gain is metric-specific.