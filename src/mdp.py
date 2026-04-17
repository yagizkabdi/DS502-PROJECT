"""MDP reformulation of the BDTP."""

from itertools import product as iproduct


def initial_state(inst):
    N = inst["N"]
    J = inst["J"]
    shelf = inst["shelf"]
    cooldown = tuple([0] * N)
    used = tuple(tuple([0.0] * len(J)) for _ in range(N))
    inv = tuple(tuple([0.0] * shelf[j]) for j in J)
    return (0, cooldown, used, inv)


def feasible_actions(inst, state):
    _, cooldown, used, _ = state
    I_types = inst["I"]
    J = inst["J"]
    N = inst["N"]
    a_coll = inst["a"]
    E = inst["E"]
    none_idx = len(I_types) - 1

    per_donor_options = []
    for k in range(N):
        opts = []
        if cooldown[k] > 0:
            opts.append(none_idx)
        else:
            for i in I_types:
                ok = True
                for j in J:
                    if used[k][j] + a_coll[i][j] > E[j] + 1e-9:
                        ok = False
                        break
                if ok:
                    opts.append(i)
        per_donor_options.append(opts)

    for combo in iproduct(*per_donor_options):
        yield combo


def step(inst, state, action):
    t, cooldown, used, inv = state
    I_types = inst["I"]
    J = inst["J"]
    N = inst["N"]
    a_coll = inst["a"]
    c = inst["c"]
    h = inst["h"]
    d_tilde = inst["d_tilde"]
    s = inst["s"]
    shelf = inst["shelf"]
    demand = inst["demand"]
    none_idx = len(I_types) - 1

    donation_cost = 0.0
    collected = [0.0 for _ in J]
    for k in range(N):
        i = action[k]
        donation_cost += c[i]
        for j in J:
            collected[j] += a_coll[i][j]

    new_inv_list = []
    disposed = [0.0 for _ in J]
    for j in J:
        sh = shelf[j]
        shifted = [0.0] + list(inv[j][: sh - 1])
        aging_out = inv[j][sh - 1] if sh >= 1 else 0.0
        shifted[0] = collected[j]

        need = demand[(j, t)]
        use = min(aging_out, need)
        aging_out -= use
        need -= use
        for age in range(sh - 1, -1, -1):
            if need <= 0:
                break
            take = min(shifted[age], need)
            shifted[age] -= take
            need -= take

        if need > 1e-9:
            return None, float("inf"), {"infeasible": True}

        disposed[j] = aging_out
        new_inv_list.append(tuple(shifted))

    hold_cost = sum(h[j] * sum(new_inv_list[j]) for j in J)
    disp_cost = sum(d_tilde[j] * disposed[j] for j in J)
    stage_cost = donation_cost + hold_cost + disp_cost

    new_cooldown = []
    new_used = []
    for k in range(N):
        i = action[k]
        if i == none_idx:
            cd = max(0, cooldown[k] - 1)
        else:
            cd = min(s[i][ip] for ip in I_types if ip != none_idx) - 1
            cd = max(cd, 0)
        new_cooldown.append(cd)

        row = list(used[k])
        for j in J:
            row[j] += a_coll[i][j]
        new_used.append(tuple(row))

    next_state = (t + 1, tuple(new_cooldown), tuple(new_used),
                  tuple(new_inv_list))

    return next_state, stage_cost, {"collected": collected,
                                    "disposed": disposed,
                                    "donation_cost": donation_cost,
                                    "hold_cost": hold_cost,
                                    "disp_cost": disp_cost}


def is_terminal(inst, state):
    return state[0] >= inst["L"]


def terminal_cost(inst, state):
    return 0.0
