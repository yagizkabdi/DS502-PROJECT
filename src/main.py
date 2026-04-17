"""Runner: solves a small instance with the MIP and the MDP."""

from bdtp_data import make_instance, pretty_print_instance
from bdtp_mip import build_and_solve_mip, print_result as print_mip
from dp import solve as dp_solve, print_trajectory


def run_small():
    print("=" * 60)
    print("Small illustrative instance")
    print("=" * 60)
    inst = make_instance(N=2, L=4, demand_pattern="constant", seed=1)
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


def run_medium():
    print("=" * 60)
    print("Medium instance (MIP only)")
    print("=" * 60)
    inst = make_instance(N=5, L=10, demand_pattern="increasing", seed=3)
    pretty_print_instance(inst)
    mip_res = build_and_solve_mip(inst, time_limit=60, verbose=False)
    print_mip(inst, mip_res)


if __name__ == "__main__":
    run_small()
    print()
    run_medium()
