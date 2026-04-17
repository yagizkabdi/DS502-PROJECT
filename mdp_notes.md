# MDP notes for BDTP

Short summary of the MDP reformulation used in Deliverable 5.

## State
At the beginning of period t:
- period index t
- for each donor k: remaining cooldown (periods until donor can donate
  a non-NONE type again)
- for each donor k, product j: amount of j already collected from k in
  the horizon so far (needed for the donor cap E_j)
- for each product j: age-bucketed inventory vector of length shelf[j]

## Action
A tuple of donation types, one per donor. NONE is allowed.
Feasibility:
- if cooldown[k] > 0 then donor k must take NONE
- donor cap: used[k][j] + a[i][j] <= E[j] for all j

## Transition
Deterministic in the base model.
1. collect new units into age-0 bucket
2. serve demand FIFO (oldest first)
3. any units that age out beyond shelf[j] become disposal
4. update cooldown and used

## Reward / cost
Cost at period t:
    sum_k c[type_k] + sum_j h[j] * end_inv[j] + sum_j d_tilde[j] * disposed[j]

## Policy
A rule pi(s) -> action. Here we compute pi exactly via backward
induction for small instances.

## Horizon
Finite horizon L. Terminal salvage = 0.

## Properties
- deterministic (demand assumed known)
- fully observable
- undiscounted
