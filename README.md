# DS502 Semester Project

Blood Donation Tailoring Problem (BDTP), based on Ozener, Ekici, Coban
(2019). MIP, MDP reformulation with backward induction, two heuristic
baselines, and a genetic algorithm.

## Files
- `src/bdtp_data.py` — instance generator
- `src/bdtp_mip.py` — Gurobi MIP (reports objective, best bound, optimality gap, wall + CPU time)
- `src/mdp.py` — MDP state / action / transition / cost
- `src/dp.py` — backward induction
- `src/heuristics.py` — rule-of-thumb and myopic greedy
- `src/bdtp_ga.py` — genetic algorithm
- `src/experiments.py` — method-comparison grid + MIP scaling study + figures
- `src/main.py` — smoke test
- `mdp_notes.md`

## Run
```
pip install -r requirements.txt
cd src
python main.py
python experiments.py
```

## Revision

Added the three things from the review:

- Optimality gap: the MIP now returns Gurobi's gap (MIPGap) and best bound
  (ObjBound), so a time-limited run shows how far the solution is from optimal.
- CPU runtime: every method logs CPU time (process_time) next to wall time.
- Larger instances: rewrote the deferral constraints to the equivalent compact
  form x[k,i,t] <= x[k,NONE,t+dt], which uses fewer rows so the MIP solves
  faster and reaches larger instances. experiments.py also has a MIP-only
  scaling run that charts solve time and optimality gap as the size grows.

New files in results/: mip_scaling.csv, mip_optimality_gap.png,
mip_runtime_vs_size.png, cpu_runtime_vs_size.png. results.csv has extra columns
cpu_runtime, mip_gap, obj_bound, status.
