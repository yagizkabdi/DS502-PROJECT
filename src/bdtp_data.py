"""Synthetic BDTP instance generator.

Donation types: 0 WB, 1 SP, 2 DP, 3 RP, 4 RPP, 5 NONE.
Products: 0 RBC, 1 platelets, 2 plasma.
"""

import random


PRODUCT_NAMES = ["RBC", "platelets", "plasma"]
DONATION_NAMES = ["WB", "SP", "DP", "RP", "RPP", "NONE"]

COLLECTION = [
    [1.0, 0.1, 0.5],   # WB
    [0.0, 1.0, 0.0],   # SP
    [0.0, 2.0, 0.0],   # DP
    [1.0, 0.0, 1.0],   # RP
    [1.0, 1.0, 1.0],   # RPP
    [0.0, 0.0, 0.0],   # NONE
]

DEFERRAL = [
    # to: WB  SP  DP  RP  RPP NONE
    [    56,  2,  7,  7,  7,  0],   # from WB
    [     2,  2,  2,  2,  2,  0],   # from SP
    [     7,  2,  7,  7,  7,  0],   # from DP
    [     7,  2,  7,  7,  7,  0],   # from RP
    [     7,  2,  7,  7,  7,  0],   # from RPP
    [     0,  0,  0,  0,  0,  0],   # from NONE
]

DONATION_COST = [138, 158, 260, 312, 364, 0]
SHELF_LIFE_FULL = [42, 5, 365]
HOLDING_COST = [0.5, 1.0, 0.2]
DISPOSAL_COST = [5.0, 8.0, 2.0]
MAX_UNITS_PER_DONOR = [6, 24, 12]


def make_instance(N=3, L=10, demand_pattern="constant", seed=7, scale_shelf=True):
    rng = random.Random(seed)

    I = list(range(len(DONATION_NAMES)))
    J = list(range(len(PRODUCT_NAMES)))
    K = list(range(N))
    T = list(range(L))

    if scale_shelf:
        shelf = [min(sh, L) for sh in SHELF_LIFE_FULL]
    else:
        shelf = list(SHELF_LIFE_FULL)

    demand = {}
    if demand_pattern == "constant":
        base = [1.0, 1.0, 0.5]
        for t in T:
            for j in J:
                demand[(j, t)] = base[j]
    elif demand_pattern == "increasing":
        for t in T:
            for j in J:
                demand[(j, t)] = 0.5 + 0.1 * t
    elif demand_pattern == "random":
        for t in T:
            for j in J:
                demand[(j, t)] = round(rng.uniform(0.5, 1.5), 2)
    else:
        raise ValueError(demand_pattern)

    return {
        "I": I, "J": J, "K": K, "T": T,
        "donation_names": DONATION_NAMES,
        "product_names": PRODUCT_NAMES,
        "c": DONATION_COST,
        "h": HOLDING_COST,
        "d_tilde": DISPOSAL_COST,
        "s": DEFERRAL,
        "shelf": shelf,
        "a": COLLECTION,
        "demand": demand,
        "E": MAX_UNITS_PER_DONOR,
        "N": N, "L": L,
    }


def pretty_print_instance(inst):
    print(f"donors  : {inst['N']}")
    print(f"periods : {inst['L']}")
    print(f"types   : {inst['donation_names']}")
    print(f"products: {inst['product_names']}")
    print("demand per period (RBC, platelets, plasma):")
    for t in inst["T"]:
        row = [inst["demand"][(j, t)] for j in inst["J"]]
        print(f"  t={t:2d}  {row}")


if __name__ == "__main__":
    inst = make_instance(N=3, L=8)
    pretty_print_instance(inst)
