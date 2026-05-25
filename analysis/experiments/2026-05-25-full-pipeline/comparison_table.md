# Stage-2 model comparison (5 exports, human-only, clean teams)

Fitted on **204** clean human team-rounds using Stage-1 params (τ_prior = 4.6385, ε = 0.062416, memory_strategy = `drift_prior_0.500`). Each of the 7 models was fit under each of 3 objectives (`combo_r`, `agg_ll`, `mean_ll`); all four core metrics are then reported at the fitted point.

The **Random** row is a 1/27-uniform reference (no parameters, no fit).

## Eval metric: `combo_r` (Pearson, per-combo × stage × env)

| Model | Fit on `combo_r` | Fit on `agg_ll` | Fit on `mean_ll` |
|-------|--------:|--------:|--------:|
| Bayesian Walk | 0.5047 | 0.4949 | 0.4901 |
| Bayesian Walk-PS | 0.4703 | 0.4635 | 0.4687 |
| Mixture-PS | 0.4722 | 0.4635 | 0.4687 |
| Bayesian-Belief | 0.4703 | 0.4703 | 0.4703 |
| Bayesian-Value | 0.4096 | 0.4081 | 0.4081 |
| Bayesian Threshold | 0.3213 | 0.2836 | 0.2967 |
| Bayesian Thresh-PS | 0.3545 | 0.3449 | 0.3442 |
| _Random (1/27)_ | 0.2427 | 0.2427 | 0.2427 |

## Eval metric: `marg_r` (Pearson, role marginals)

| Model | Fit on `combo_r` | Fit on `agg_ll` | Fit on `mean_ll` |
|-------|--------:|--------:|--------:|
| Bayesian Walk | 0.3576 | 0.3588 | 0.3479 |
| Bayesian Walk-PS | 0.1961 | 0.2258 | 0.2114 |
| Mixture-PS | 0.2171 | 0.2258 | 0.2114 |
| Bayesian-Belief | 0.1961 | 0.1961 | 0.1961 |
| Bayesian-Value | 0.1852 | 0.1878 | 0.1831 |
| Bayesian Threshold | 0.3495 | 0.2651 | 0.2812 |
| Bayesian Thresh-PS | 0.2886 | 0.2841 | 0.2837 |
| _Random (1/27)_ | -0.0000 | -0.0000 | -0.0000 |

## Eval metric: `agg_ll` (aggregate cross-entropy (↑ better))

| Model | Fit on `combo_r` | Fit on `agg_ll` | Fit on `mean_ll` |
|-------|--------:|--------:|--------:|
| Bayesian Walk | -2.5073 | -2.4756 | -2.4846 |
| Bayesian Walk-PS | -2.5181 | -2.5117 | -2.5136 |
| Mixture-PS | -2.5118 | -2.5117 | -2.5136 |
| Bayesian-Belief | -2.5181 | -2.5181 | -2.5181 |
| Bayesian-Value | -2.5675 | -2.5655 | -2.5793 |
| Bayesian Threshold | -10.5305 | -9.8620 | -9.9329 |
| Bayesian Thresh-PS | -8.1085 | -7.4217 | -7.4219 |
| _Random (1/27)_ | -2.8085 | -2.8085 | -2.8085 |

## Eval metric: `mean_ll` (per-sample mean log-lik (↑ better))

| Model | Fit on `combo_r` | Fit on `agg_ll` | Fit on `mean_ll` |
|-------|--------:|--------:|--------:|
| Bayesian Walk | -2.9254 | -2.8269 | -2.8068 |
| Bayesian Walk-PS | -2.9042 | -2.9064 | -2.8981 |
| Mixture-PS | -2.9063 | -2.9064 | -2.8981 |
| Bayesian-Belief | -2.9042 | -2.9042 | -2.9042 |
| Bayesian-Value | -3.0992 | -3.1175 | -3.0934 |
| Bayesian Threshold | -27.1870 | -26.9340 | -26.8541 |
| Bayesian Thresh-PS | -24.9757 | -24.2088 | -24.2088 |
| _Random (1/27)_ | -3.2958 | -3.2958 | -3.2958 |

## Fitted parameters

| Model | Fit objective | Params |
|-------|---------------|--------|
| Bayesian Walk | combo_r | tau_softmax=7.2065, epsilon_switch=0.5590 |
| Bayesian Walk | agg_ll | tau_softmax=12.0943, epsilon_switch=0.5415 |
| Bayesian Walk | mean_ll | tau_softmax=15.7357, epsilon_switch=0.6135 |
| Bayesian Walk-PS | combo_r | epsilon_switch=1.0000 |
| Bayesian Walk-PS | agg_ll | epsilon_switch=0.8231 |
| Bayesian Walk-PS | mean_ll | epsilon_switch=0.9188 |
| Mixture-PS | combo_r | w=0.9042 |
| Mixture-PS | agg_ll | w=1.0000 |
| Mixture-PS | mean_ll | w=1.0000 |
| Bayesian-Belief | combo_r | (none) |
| Bayesian-Belief | agg_ll | (none) |
| Bayesian-Belief | mean_ll | (none) |
| Bayesian-Value | combo_r | tau_softmax=13.7160 |
| Bayesian-Value | agg_ll | tau_softmax=11.6222 |
| Bayesian-Value | mean_ll | tau_softmax=16.8575 |
| Bayesian Threshold | combo_r | tau_softmax=2.4709, delta=0.4643 |
| Bayesian Threshold | agg_ll | tau_softmax=18.5785, delta=0.4286 |
| Bayesian Threshold | mean_ll | tau_softmax=10.0496, delta=0.4643 |
| Bayesian Thresh-PS | combo_r | epsilon_switch=0.9992, delta=0.2500 |
| Bayesian Thresh-PS | agg_ll | epsilon_switch=0.3862, delta=0.0000 |
| Bayesian Thresh-PS | mean_ll | epsilon_switch=0.3618, delta=0.0000 |
