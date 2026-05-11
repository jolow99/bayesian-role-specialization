# Metric Comparison (current April exports, human-only, clean teams)

Fitted on 114 clean human team-rounds from the current April exports (see `pipeline.load_human_team_records` — every team-round must have a matching value matrix in `human-envs-big-pilot-matrices/<stat_profile>__<role_combo>`; missing matrices raise rather than silently dropping records). Each cell shows the metric value when the model was fit under that objective.

## Eval metric: `combo_r`

| Model | Fit: combo_r | Fit: agg_ll | Fit: mean_ll |
|-------|--------:|--------:|--------:|
| bayesian_walk_ps | 0.3977 | 0.3922 | 0.3926 |
| bayesian_walk | 0.4348 | 0.4210 | 0.4127 |
| mixture_ps | 0.3977 | 0.3921 | 0.3926 |
| bayesian_thresh_ps | 0.3055 | 0.2943 | 0.2890 |
| bayesian_thresh | 0.2623 | 0.2269 | 0.2361 |
| bayesian_belief | 0.3977 | 0.3977 | 0.3977 |
| bayesian_value | 0.3717 | 0.3692 | 0.3707 |

## Eval metric: `marg_r`

| Model | Fit: combo_r | Fit: agg_ll | Fit: mean_ll |
|-------|--------:|--------:|--------:|
| bayesian_walk_ps | 0.1246 | 0.1542 | 0.1527 |
| bayesian_walk | 0.3940 | 0.3885 | 0.3813 |
| mixture_ps | 0.1246 | 0.1545 | 0.1527 |
| bayesian_thresh_ps | 0.2508 | 0.2557 | 0.2493 |
| bayesian_thresh | 0.4218 | 0.3094 | 0.3256 |
| bayesian_belief | 0.1246 | 0.1246 | 0.1246 |
| bayesian_value | 0.2677 | 0.2719 | 0.2659 |

## Eval metric: `agg_ll`

| Model | Fit: combo_r | Fit: agg_ll | Fit: mean_ll |
|-------|--------:|--------:|--------:|
| bayesian_walk_ps | -2.5070 | -2.5041 | -2.5042 |
| bayesian_walk | -2.4818 | -2.4456 | -2.4516 |
| mixture_ps | -2.5070 | -2.5041 | -2.5042 |
| bayesian_thresh_ps | -9.3282 | -9.0631 | -9.1870 |
| bayesian_thresh | -12.0134 | -10.7845 | -10.8505 |
| bayesian_belief | -2.5070 | -2.5070 | -2.5070 |
| bayesian_value | -2.5342 | -2.5271 | -2.5437 |

## Eval metric: `mean_ll`

| Model | Fit: combo_r | Fit: agg_ll | Fit: mean_ll |
|-------|--------:|--------:|--------:|
| bayesian_walk_ps | -2.8826 | -2.8722 | -2.8722 |
| bayesian_walk | -2.8668 | -2.7613 | -2.7558 |
| mixture_ps | -2.8826 | -2.8723 | -2.8722 |
| bayesian_thresh_ps | -23.5717 | -23.3099 | -23.3074 |
| bayesian_thresh | -28.5679 | -27.5228 | -27.1756 |
| bayesian_belief | -2.8826 | -2.8826 | -2.8826 |
| bayesian_value | -3.0951 | -3.1164 | -3.0935 |

## Fitted Parameters

| Model | Objective | Params |
|-------|-----------|--------|
| bayesian_walk_ps | combo_r | epsilon_switch=1.0000 |
| bayesian_walk_ps | agg_ll | epsilon_switch=0.8871 |
| bayesian_walk_ps | mean_ll | epsilon_switch=0.8936 |
| bayesian_walk | combo_r | tau_softmax=7.2091, epsilon_switch=0.6208 |
| bayesian_walk | agg_ll | tau_softmax=12.8915, epsilon_switch=0.5794 |
| bayesian_walk | mean_ll | tau_softmax=16.0850, epsilon_switch=0.5772 |
| mixture_ps | combo_r | w=1.0000 |
| mixture_ps | agg_ll | w=0.9989 |
| mixture_ps | mean_ll | w=1.0000 |
| bayesian_thresh_ps | combo_r | epsilon_switch=0.8862, delta=0.2000 |
| bayesian_thresh_ps | agg_ll | epsilon_switch=0.4265, delta=0.0000 |
| bayesian_thresh_ps | mean_ll | epsilon_switch=0.2858, delta=0.0000 |
| bayesian_thresh | combo_r | tau_softmax=1.5210, delta=0.5000 |
| bayesian_thresh | agg_ll | tau_softmax=18.5785, delta=0.0357 |
| bayesian_thresh | mean_ll | tau_softmax=14.3141, delta=0.1429 |
| bayesian_belief | combo_r | (none) |
| bayesian_belief | agg_ll | (none) |
| bayesian_belief | mean_ll | (none) |
| bayesian_value | combo_r | tau_softmax=14.7631 |
| bayesian_value | agg_ll | tau_softmax=11.6215 |
| bayesian_value | mean_ll | tau_softmax=16.8573 |
