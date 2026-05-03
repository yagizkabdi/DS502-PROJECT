"""Smoke test: MIP vs MDP-DP on a small instance."""

import random

import numpy as np

from bdtp_data import make_instance, pretty_print_instance
from bdtp_mip import build_and_solve_mip, print_result as print_mip
from dp import solve as dp_solve, print_trajectory


SEED = 7


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_small() -> None:
    print("=" * 60)
    print("Small illustrative instance (smoke test)")
    print("=" * 60)
    inst = make_instance(N=2, L=4, demand_pattern="constant", seed=SEED)
    pretty_print_instance(inst)

    print("\n-- MIP --")
    mip_res = build_and_solve_mip(inst, time_limit=30, verbose=False)
    print_mip(inst, mip_res)

    print("\n-- MDP / backward induction --")
    dp_res = dp_solve(inst)
    print_trajectory(inst, dp_res)

    if mip_res["obj"] is not None:
        gap = abs(mip_res["obj"] - dp_res["opt_cost"])
        print(f"\n|MIP - DP| = {gap:.4f}")
        assert gap < 1e-6, "MIP and DP disagree on the small instance"
        print("MIP <-> DP agreement: OK")


def run_medium() -> None:
    print("=" * 60)
    print("Medium instance (MIP only)")
    print("=" * 60)
    inst = make_instance(N=5, L=10, demand_pattern="increasing", seed=SEED)
    pretty_print_instance(inst)
    mip_res = build_and_solve_mip(inst, time_limit=60, verbose=False)
    print_mip(inst, mip_res)


if __name__ == "__main__":
    set_seeds()
    run_small()
    print()
    run_medium()
