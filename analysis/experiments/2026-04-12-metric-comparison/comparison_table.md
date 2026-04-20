# Metric Comparison: 3 Objectives × 7 Models (Human-Only)

Fitted on human-only records. Each cell shows the metric value when the model was fit under that objective.

## Eval metric: `combo_r`

| Model | Fit: combo_r | Fit: agg_ll | Fit: mean_ll |
|-------|--------:|--------:|--------:|
| bayesian_walk_ps | 0.5168 | 0.5144 | 0.5157 |
| bayesian_walk | 0.5389 | 0.5295 | 0.5117 |
| mixture_ps | 0.5170 | 0.5141 | 0.5158 |
| bayesian_thresh_ps | 0.4582 | 0.4557 | 0.4581 |
| bayesian_thresh | 0.4294 | 0.3612 | 0.3978 |
| bayesian_belief | 0.5036 | 0.5036 | 0.5036 |
| bayesian_value | 0.4167 | 0.4077 | 0.3983 |

## Eval metric: `marg_r`

| Model | Fit: combo_r | Fit: agg_ll | Fit: mean_ll |
|-------|--------:|--------:|--------:|
| bayesian_walk_ps | 0.5123 | 0.5206 | 0.5040 |
| bayesian_walk | 0.6316 | 0.6227 | 0.6167 |
| mixture_ps | 0.5133 | 0.5203 | 0.5042 |
| bayesian_thresh_ps | 0.5174 | 0.5056 | 0.5188 |
| bayesian_thresh | 0.5648 | 0.4811 | 0.5187 |
| bayesian_belief | 0.4674 | 0.4674 | 0.4674 |
| bayesian_value | 0.4562 | 0.4295 | 0.4182 |

## Eval metric: `agg_ll`

| Model | Fit: combo_r | Fit: agg_ll | Fit: mean_ll |
|-------|--------:|--------:|--------:|
| bayesian_walk_ps | -2.5702 | -2.5660 | -2.5770 |
| bayesian_walk | -2.5671 | -2.5226 | -2.5475 |
| mixture_ps | -2.5679 | -2.5658 | -2.5767 |
| bayesian_thresh_ps | -6.7502 | -6.7473 | -6.7510 |
| bayesian_thresh | -12.2023 | -11.0614 | -11.4728 |
| bayesian_belief | -2.6043 | -2.6043 | -2.6043 |
| bayesian_value | -2.7836 | -2.6929 | -2.7013 |

## Eval metric: `mean_ll`

| Model | Fit: combo_r | Fit: agg_ll | Fit: mean_ll |
|-------|--------:|--------:|--------:|
| bayesian_walk_ps | -2.6712 | -2.7048 | -2.6650 |
| bayesian_walk | -2.9341 | -2.7176 | -2.6502 |
| mixture_ps | -2.6771 | -2.7121 | -2.6650 |
| bayesian_thresh_ps | -21.4700 | -21.4719 | -21.4700 |
| bayesian_thresh | -26.3819 | -26.0632 | -25.9980 |
| bayesian_belief | -2.7020 | -2.7020 | -2.7020 |
| bayesian_value | -3.2058 | -3.0237 | -3.0134 |

## Fitted Parameters

| Model | Objective | Params |
|-------|-----------|--------|
| bayesian_walk_ps | combo_r | epsilon_switch=0.7239 |
| bayesian_walk_ps | agg_ll | epsilon_switch=0.6069 |
| bayesian_walk_ps | mean_ll | epsilon_switch=0.8031 |
| bayesian_walk | combo_r | tau_softmax=6.2419, epsilon_switch=0.3468 |
| bayesian_walk | agg_ll | tau_softmax=10.0508, epsilon_switch=0.4240 |
| bayesian_walk | mean_ll | tau_softmax=13.3966, epsilon_switch=0.5771 |
| mixture_ps | combo_r | w=0.9510 |
| mixture_ps | agg_ll | w=0.9670 |
| mixture_ps | mean_ll | w=0.9963 |
| bayesian_thresh_ps | combo_r | epsilon_switch=0.3790, delta=0.0000 |
| bayesian_thresh_ps | agg_ll | epsilon_switch=0.5238, delta=0.0000 |
| bayesian_thresh_ps | mean_ll | epsilon_switch=0.3594, delta=0.0000 |
| bayesian_thresh | combo_r | tau_softmax=2.4887, delta=0.5000 |
| bayesian_thresh | agg_ll | tau_softmax=14.3146, delta=0.0000 |
| bayesian_thresh | mean_ll | tau_softmax=8.6296, delta=0.5000 |
| bayesian_belief | combo_r | (none) |
| bayesian_belief | agg_ll | (none) |
| bayesian_belief | mean_ll | (none) |
| bayesian_value | combo_r | tau_softmax=7.4312 |
| bayesian_value | agg_ll | tau_softmax=12.6690 |
| bayesian_value | mean_ll | tau_softmax=15.8101 |
