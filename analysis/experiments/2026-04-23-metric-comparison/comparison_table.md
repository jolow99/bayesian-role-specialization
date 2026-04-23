# Metric Comparison (04-23 export, human-only, clean teams)

Fitted on 42 clean human team-rounds from the 04-23 export (see `pipeline.load_human_team_records` — teams without a matching precomputed value matrix are skipped so that all seven models are compared on the same records). Each cell shows the metric value when the model was fit under that objective.

## Eval metric: `combo_r`

| Model | Fit: combo_r | Fit: agg_ll | Fit: mean_ll |
|-------|--------:|--------:|--------:|
| bayesian_walk_ps | 0.2390 | 0.2354 | 0.2372 |
| bayesian_walk | 0.3522 | 0.3083 | 0.3023 |
| mixture_ps | 0.2485 | 0.2354 | 0.2372 |
| bayesian_thresh_ps | 0.2446 | 0.2213 | 0.2220 |
| bayesian_thresh | 0.3333 | 0.2872 | 0.2901 |
| bayesian_belief | 0.2311 | 0.2311 | 0.2311 |
| bayesian_value | 0.3032 | 0.2934 | 0.2550 |

## Eval metric: `marg_r`

| Model | Fit: combo_r | Fit: agg_ll | Fit: mean_ll |
|-------|--------:|--------:|--------:|
| bayesian_walk_ps | 0.3568 | 0.2867 | 0.3102 |
| bayesian_walk | 0.6106 | 0.5232 | 0.5199 |
| mixture_ps | 0.3790 | 0.2867 | 0.3102 |
| bayesian_thresh_ps | 0.3802 | 0.4306 | 0.4290 |
| bayesian_thresh | 0.5423 | 0.3718 | 0.3787 |
| bayesian_belief | 0.2415 | 0.2415 | 0.2415 |
| bayesian_value | 0.3749 | 0.3523 | 0.3268 |

## Eval metric: `agg_ll`

| Model | Fit: combo_r | Fit: agg_ll | Fit: mean_ll |
|-------|--------:|--------:|--------:|
| bayesian_walk_ps | -2.7353 | -2.7198 | -2.7208 |
| bayesian_walk | -5.5648 | -2.5790 | -2.5817 |
| mixture_ps | -2.9997 | -2.7198 | -2.7208 |
| bayesian_thresh_ps | -10.9942 | -9.6958 | -9.6961 |
| bayesian_thresh | -12.8777 | -9.4854 | -9.4875 |
| bayesian_belief | -2.7219 | -2.7219 | -2.7219 |
| bayesian_value | -2.7046 | -2.6496 | -2.6813 |

## Eval metric: `mean_ll`

| Model | Fit: combo_r | Fit: agg_ll | Fit: mean_ll |
|-------|--------:|--------:|--------:|
| bayesian_walk_ps | -3.0156 | -2.9889 | -2.9839 |
| bayesian_walk | -7.6226 | -2.7133 | -2.7100 |
| mixture_ps | -3.5703 | -2.9889 | -2.9839 |
| bayesian_thresh_ps | -24.0537 | -21.6413 | -21.6411 |
| bayesian_thresh | -27.5381 | -24.2144 | -24.2127 |
| bayesian_belief | -3.0186 | -3.0186 | -3.0186 |
| bayesian_value | -3.3734 | -3.1897 | -3.1399 |

## Fitted Parameters

| Model | Objective | Params |
|-------|-----------|--------|
| bayesian_walk_ps | combo_r | epsilon_switch=0.6281 |
| bayesian_walk_ps | agg_ll | epsilon_switch=0.8827 |
| bayesian_walk_ps | mean_ll | epsilon_switch=0.8097 |
| bayesian_walk | combo_r | tau_softmax=0.6618, epsilon_switch=0.4290 |
| bayesian_walk | agg_ll | tau_softmax=10.6457, epsilon_switch=0.5579 |
| bayesian_walk | mean_ll | tau_softmax=12.8927, epsilon_switch=0.5300 |
| mixture_ps | combo_r | w=0.3504 |
| mixture_ps | agg_ll | w=1.0000 |
| mixture_ps | mean_ll | w=1.0000 |
| bayesian_thresh_ps | combo_r | epsilon_switch=1.0000, delta=0.0000 |
| bayesian_thresh_ps | agg_ll | epsilon_switch=0.2831, delta=0.0000 |
| bayesian_thresh_ps | mean_ll | epsilon_switch=0.3148, delta=0.0000 |
| bayesian_thresh | combo_r | tau_softmax=0.5461, delta=0.0357 |
| bayesian_thresh | agg_ll | tau_softmax=14.3142, delta=0.5000 |
| bayesian_thresh | mean_ll | tau_softmax=11.4715, delta=0.5000 |
| bayesian_belief | combo_r | (none) |
| bayesian_belief | agg_ll | (none) |
| bayesian_belief | mean_ll | (none) |
| bayesian_value | combo_r | tau_softmax=5.3367 |
| bayesian_value | agg_ll | tau_softmax=8.4800 |
| bayesian_value | mean_ll | tau_softmax=14.7632 |
