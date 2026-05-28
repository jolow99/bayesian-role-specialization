# Aggregate model comparison — paper Section: How well does the model explain human team behavior?

Fitted on **204** clean human team-rounds from 5 exports (2026-05-25 pipeline scope). Bayesian models use the pipeline's `agg_ll`-objective fits; Random Walk's ε is grid-searched here on the same objective; other baselines are parameter-free. Stage-1 params: τ_prior = 4.6385, ε = 0.062416, memory_strategy = `drift_prior_0.500`.

Models sorted by `combo_r` (descending).

| Model | combo_r | marg_r | agg_ll | mean_ll |
|-------|--------:|-------:|-------:|--------:|
| Bayesian Walk | 0.4949 | 0.3588 | -2.4756 | -2.8269 |
| Bayesian-Belief | 0.4703 | 0.1961 | -2.5181 | -2.9042 |
| Bayesian Walk-PS | 0.4635 | 0.2258 | -2.5117 | -2.9064 |
| Mixture-PS | 0.4635 | 0.2258 | -2.5117 | -2.9064 |
| Bayesian-Value | 0.4081 | 0.1878 | -2.5655 | -3.1175 |
| Random Walk | 0.3925 | 0.2645 | -2.5871 | -3.0163 |
| Top-7 | 0.3507 | 0.5213 | -12.4214 | -20.7437 |
| Bayesian Thresh-PS | 0.3449 | 0.2841 | -7.4217 | -24.2088 |
| Contradict Others | 0.2927 | -0.2150 | -2.8611 | -3.9494 |
| Bayesian Threshold | 0.2836 | 0.2651 | -9.8620 | -26.9340 |
| Random | 0.2427 | -0.0000 | -2.8085 | -3.2958 |
| Optimal | 0.2395 | 0.4842 | -21.7164 | -36.4217 |
| Random-to-Optimal | 0.1801 | 0.3543 | -5.0678 | -12.6825 |
| Copy Others | -0.0038 | 0.2645 | -7.8026 | -31.7807 |

## Fitted parameters

| Model | Params |
|-------|--------|
| Bayesian Walk | tau_softmax=12.0943, epsilon_switch=0.5415 |
| Bayesian Walk-PS | epsilon_switch=0.8231 |
| Mixture-PS | w=1.0000 |
| Bayesian-Belief | (none) |
| Bayesian-Value | tau_softmax=11.6222 |
| Bayesian Threshold | tau_softmax=18.5785, delta=0.4286 |
| Bayesian Thresh-PS | epsilon_switch=0.3862, delta=0.0000 |
| Random Walk | eps=0.2500 |
| Top-7 | k=7 |
| Random-to-Optimal | (none) |
| Optimal | (none) |
| Copy Others | (none) |
| Contradict Others | (none) |
| Random | (none) |
