#import "@preview/polylux:0.4.0": *
#import "@preview/metropolis-polylux:0.1.0" as metropolis
#import metropolis: focus, new-section

#show: metropolis.setup

// Title Slide
#slide[
  #set page(header: none, footer: none, margin: 3em)

  #text(size: 1.3em)[
    *A Bayesian Model of Ad-Hoc Role Specialization*
  ]

  CogSci 2026

  #metropolis.divider

  #set text(size: .8em, weight: "light")
  Joseph Low#super[1], Bhavyesh Sajja#super[2], Tan Zhi-Xuan#super[2]

  #super[1]Metagov, #super[2]National University of Singapore
]

#slide[
  = Agenda

  #metropolis.outline
]

// ============================================
#new-section[Introduction]
// ============================================

#slide[
  = The Puzzle of Spontaneous Coordination

  Humans coordinate with unfamiliar teammates *without explicit communication*

  #v(1em)

  *Role specialization*: spontaneous differentiation into complementary roles
  - "The planner," "the executor," "the quality checker"
  - Emerges without central authority
  - Makes teams more effective

  #v(1em)

  *Key question*: How do people infer teammates' roles and decide their own?
]

#slide[
  = Our Approach

  *Hypothesis*: People solve this through *Bayesian inference*

  #v(1em)

  1. Maintain probabilistic beliefs about teammates' roles
  2. Update beliefs based on observed actions
  3. Select own role to maximize team utility

  #v(1em)

  This is a *best-response* strategy: fill gaps in team composition
]

// ============================================
#new-section[Computational Model]
// ============================================

#slide[
  = Roles as Latent Variables

  *Core insight*: Roles generate observable actions

  #v(1em)

  - *Roles* $bold(r) = (r_0, r_1, r_2)$ are latent (unobserved)
  - *Actions* $bold(a) = (a_0, a_1, a_2)$ are observed
  - Each role $r$ defines a policy $pi_r: cal(S) arrow Delta(cal(A))$

  #v(1em)

  Agent $i$ maintains belief over role assignments:
  $ b_i^t (bold(r)) = P(bold(r) | bold(o)_(1:t)) $
]

#slide[
  = Bayesian Belief Update

  Update beliefs via Bayes' rule:
  $ b_i^(t+1)(bold(r)) prop P(bold(a)_(t+1) | bold(r), s_(t+1)) dot b_i^t (bold(r)) $

  #v(1em)

  Observation likelihood (conditional independence):
  $ P(bold(a) | bold(r), s) = product_(j=0)^2 pi_(r_j) (a_j | s) $
]

#slide[
  = Role Selection

  Agent $i$ chooses a role that maximizes the expected reward

  $ r_i = op("softmax")_(r in "all role combos") ( EE_(r_(-i) ~b_i^t (r_i, dot, dot)) [V^(r_i, r_(-i)) (s)] ) $

  $ EE_(r ~b_i^t (dot)) [V^r (s)] = sum_(r_(-i)) P(r_(-i)) dot V^((r_i, r_(-i)))(s) $



  #v(1em)

  - $V^(bold(r))(s)$ = state-value function for role assignment $bold(r)$
  - Select role via softmax over expected values
  - *Best-response*: complement teammates' likely roles
]

// ============================================
#new-section[The Cooperative Battle Game]
// ============================================

#slide[
  = Game Setup

  *3 players* work together to defeat an enemy

  #v(0.5em)

  - Team and enemy each have health points
  - Win: reduce enemy health to 0
  - Lose: team health reaches 0
  - Maximum 20 rounds, simultaneous resolution

  #v(1em)

  *Player capabilities*: STR, DEF, SUP (sum to 6)
  - Asymmetries incentivize specialization
]

#slide[
  = Three Roles

  Roles commit for *2 consecutive rounds*

  #v(1em)

  #table(
    columns: (auto, 1fr),
    stroke: none,
    row-gutter: 0.8em,
    [*Fighter*], [Always attacks (uses STR)],
    [*Tank*], [Blocks if enemy attacks, else attacks (uses DEF)],
    [*Medic*], [Heals if team health low, else attacks (uses SUP)],
  )

  #v(1em)

  *Key*: Players observe *actions* but not *roles*
]

#slide[
  = Why Specialization is Beneficial

  #v(0.5em)

  1. *Complementarity*: One player per role is typically optimal

  2. *Block sub-additivity*: Multiple blockers provide no extra benefit

  3. *Role commitment*: 2-round commitment encourages stable adoption

  #v(1em)

  Creates clear incentives for differentiation
]

#slide[
  = Formal Definition: MDP

  $chevron.l cal(N), cal(S), cal(A), cal(O), cal(R), T, gamma chevron.r$

  #v(0.5em)

  - *Agents*: $cal(N) = {0, 1, 2}$
  - *State*: Health values, enemy intent
  - *Actions*: Attack, Block, Heal
  - *Observations*: State + all actions (but not roles)
  - *Reward*: $+100$ win, $-100$ loss (shared)
]

#slide[
  = Role-Based Policies

  Epsilon-greedy policies ($epsilon = 0.1$):

  #v(0.5em)

  *Fighter*:
  $
    pi_"Fighter"(a | s) = cases(
      1 - epsilon + epsilon/3 & "if" a = "Attack",
      epsilon/3 & "otherwise"
    )
  $

  #v(0.5em)

  *Tank* and *Medic*: Similar structure with state-dependent preferred actions
]

#slide[
  = Implementation

  Probabilistic programming with `memo` in JAX

  #v(1em)

  1. Precompute value functions for all 27 role assignments
  2. Initialize uniform prior over assignments
  3. Each round:
    - Generate enemy intent
    - Players sample roles (softmax over EV)
    - Convert roles to actions
    - Update beliefs via Bayesian inference
]

// ============================================
#new-section[Example Gameplay]
// ============================================

#slide[
  = Balanced Team (2-2-2)

  #align(center)[
    #image("222_222_222_FTM/env_4060_round4.png", height: 80%)
  ]
]

#slide[
  = Specialized Team (4-1-1)

  #align(center)[
    #image("411_141_114_FTM/env_4119_round4.png", height: 80%)
  ]
]

#slide[
  = Mixed Team (1-2-2)

  #align(center)[
    #image("114_222_222_MFF/env_959_round4.png", height: 80%)
  ]
]

// ============================================
#new-section[Human Experiment]
// ============================================

#slide[
  = Experimental Design

  *150 participants* via Prolific, teams of 3

  #v(1em)

  *8 rounds per team*:
  - 6 with human teammates
  - 2 with bots (attention checks)

  #v(1em)

  *15 environment configurations*:
  - Balanced: All equal (STR=2, DEF=2, SUP=2)
  - AllUnique: Each player specializes differently
  - OneUnique: One specialist, two balanced
]

#slide[
  = Procedure

  1. Instructions + 2 tutorial rounds with bots
  2. 8 game rounds (up to 5 stages each)
  3. Exit survey on strategy and role beliefs

  #v(1em)

  *Critical feature*: Players observe actions but not roles

  Matches the inference problem in our model
]

#slide[
  = Model Predictions

  #v(0.5em)

  *P1*: Specialist players adopt specialist roles

  *P2*: Rapid convergence with AllUnique stats

  *P3*: Slower convergence with Balanced stats

  *P4*: Learning effects across rounds

  *P5*: Sensitivity to environment configuration
]

// ============================================
#new-section[Results]
// ============================================

#slide[
  = P1: Specialists Adopt Specialist Roles

  #align(center)[
    #rect(width: 80%, height: 120pt, stroke: 1pt + gray)[
      #align(center + horizon)[
        _Role selection frequency by stat profile_
      ]
    ]
  ]

  #v(0.5em)

  Players with specialized stats predominantly adopted matching roles
]

#slide[
  = P4: Learning Effects

  #align(center)[
    #rect(width: 80%, height: 120pt, stroke: 1pt + gray)[
      #align(center + horizon)[
        _Specialist role adoption rate across rounds_
      ]
    ]
  ]

  #v(0.5em)

  Players who didn't adopt specialist role initially converged over time
]

#slide[
  = Interesting Deviations

  *"First-mover" dynamics*:

  #v(0.5em)

  - Balanced-stat player adopted and maintained Tank role
  - DEF-specialist *deferred* to Fighter rather than compete

  #v(1em)

  Suggests social dynamics beyond pure utility maximization
]

// ============================================
#new-section[Discussion]
// ============================================

#slide[
  = Key Findings

  #v(0.5em)

  1. *Specialists adopt matching roles* --- supports Bayesian account

  2. *Learning effects* --- coordination improves with experience

  3. *First-mover dynamics* --- not captured by model

  #v(1em)

  Human coordination involves social dynamics beyond utility maximization
]

#slide[
  = Limitations

  - Assumes common knowledge of game mechanics
  - Treats all players identically
  - Simplified 3-player structure

  #v(1em)

  *Future work*:
  - Social preferences
  - Individual differences
  - Learning dynamics
]

#slide[
  #show: focus

  Bayesian inference provides a useful framework for understanding how people coordinate behavior in service of shared goals
]

#slide[
  = Contributions

  #v(0.5em)

  1. *Theoretical*: Formal Bayesian account of emergent role differentiation

  2. *Empirical*: Controlled experimental paradigm

  3. *Computational*: Implementation via probabilistic programming

  #v(1em)

  *Thank you!*

  Questions?
]
