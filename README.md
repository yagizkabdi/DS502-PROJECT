# DS502 Semester Project

Blood Donation Tailoring Problem (BDTP), based on Ozener, Ekici, Coban
(2019). We implement the MIP and reformulate it as an MDP.

## Files
- `src/bdtp_data.py` — instance generator
- `src/bdtp_mip.py` — Gurobi MIP
- `src/mdp.py` — MDP (state / action / transition / cost)
- `src/dp.py` — backward induction
- `src/main.py` — runner
- `mdp_notes.md` — short MDP notes

## Run
```
pip install gurobipy
cd src
python main.py
```
