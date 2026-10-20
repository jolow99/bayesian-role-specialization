# Fitted parameters by variant (agg-LL objective unless noted)

Code names: `epsilon_switch` = weight on the switch/sampling distribution (paper's ε = 1 − epsilon_switch); Random Walk `eps_switch` = total switch prob.

| Model | none | lapse (λ=0.05) | floor (1e-12) |
|---|---|---|---|
| Bayesian-BR | tau_softmax = 11.622 | tau_softmax = 10.574 | tau_softmax = 11.622 |
| Bayesian-Walk-BR | tau_softmax = 12.094, epsilon_switch = 0.542 | tau_softmax = 10.050, epsilon_switch = 0.525 | tau_softmax = 12.094, epsilon_switch = 0.542 |
| Bayesian-Walk-PS | epsilon_switch = 0.823 | epsilon_switch = 0.782 | epsilon_switch = 0.823 |
| Bayesian-Threshold-BR | undefined (agg-LL = −∞ everywhere) | tau_softmax = 14.314, delta = 0.464 | tau_softmax = 18.578, delta = 0.429 |
| Bayesian-Threshold-PS [agg_ll] | undefined (−∞ at every δ) | δ = 0.44 (plateau 0.44–1.00) | δ = 0.44 (plateau 0.44–1.00) |
| Bayesian-Threshold-PS [mean_ll] | undefined (−∞ at every δ) | δ = 0.44 (plateau 0.44–1.00) | δ = 0.44 (plateau 0.44–1.00) |
| Random Walk | eps_switch = 0.270 | eps_switch = 0.260 | eps_switch = 0.270 |
