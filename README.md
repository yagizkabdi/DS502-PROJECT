# DS502 Semester Project

Blood Donation Tailoring Problem (BDTP), based on Ozener, Ekici, Coban
(2019). MIP, MDP reformulation with backward induction, two heuristic
baselines, and a genetic algorithm.

## Files
- `src/bdtp_data.py` — instance generator
- `src/bdtp_mip.py` — Gurobi MIP
- `src/mdp.py` — MDP state / action / transition / cost
- `src/dp.py` — backward induction
- `src/heuristics.py` — rule-of-thumb and myopic greedy
- `src/bdtp_ga.py` — genetic algorithm
- `src/experiments.py` — full grid + figures
- `src/main.py` — smoke test
- `mdp_notes.md`

## Run
```
pip install -r requirements.txt
cd src
python main.py
python experiments.py
```
