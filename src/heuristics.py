"""Heuristic baselines: rule-of-thumb (uniform type) and myopic greedy."""

import time

from mdp import initial_state, feasible_actions, step, is_terminal


def simulate(inst, policy_fn):
    state = initial_state(inst)
    total_cost = 0.0
    trajectory = []
    while not is_terminal(inst, state):
        action = policy_fn(state, inst)
        if action is None:
            return float("inf"), trajectory
        next_state, stage_cost, info = step(inst, state, action)
        if stage_cost == float("inf") or info.get("infeasible"):
            return float("inf"), trajectory
        trajectory.append({"t": state[0], "action": action,
                           "stage_cost": stage_cost})
        total_cost += stage_cost
        state = next_state
    return total_cost, trajectory


def _fixed_type_policy(types_per_donor):
    def policy(state, inst):
        none_idx = len(inst["I"]) - 1
        _, cooldown, used, _ = state
        action = []
        for k in range(inst["N"]):
            i = types_per_donor[k]
            if cooldown[k] > 0:
                action.append(none_idx)
                continue
            ok = True
            for j in inst["J"]:
                if used[k][j] + inst["a"][i][j] > inst["E"][j] + 1e-9:
                    ok = False
                    break
            action.append(i if ok else none_idx)
        return tuple(action)
    return policy


def rule_of_thumb(inst):
    none_idx = len(inst["I"]) - 1
    t0 = time.time()
    best = {"cost": float("inf"), "type": None, "trajectory": None}
    for i in inst["I"]:
        if i == none_idx:
            continue
        types = [i] * inst["N"]
        cost, traj = simulate(inst, _fixed_type_policy(types))
        if cost < best["cost"]:
            best.update({"cost": cost, "type": i, "trajectory": traj})
    return {
        "obj": best["cost"],
        "type": best["type"],
        "trajectory": best["trajectory"],
        "runtime": time.time() - t0,
    }


def _greedy_action(state, inst):
    t = state[0]
    demand = [inst["demand"][(j, t)] for j in inst["J"]]
    inv = state[3]
    available = [sum(inv[j]) for j in inst["J"]]
    short = [max(0.0, demand[j] - available[j]) for j in inst["J"]]

    best_action = None
    best_score = float("inf")
    for action in feasible_actions(inst, state):
        collected = [0.0 for _ in inst["J"]]
        donation_cost = 0.0
        for k, i in enumerate(action):
            donation_cost += inst["c"][i]
            for j in inst["J"]:
                collected[j] += inst["a"][i][j]
        unmet = sum(max(0.0, short[j] - collected[j]) for j in inst["J"])
        score = donation_cost + 1e6 * unmet
        if score < best_score:
            best_score = score
            best_action = action
            if unmet == 0 and donation_cost == 0:
                break
    return best_action


def myopic_greedy(inst):
    t0 = time.time()
    cost, traj = simulate(inst, _greedy_action)
    return {"obj": cost, "trajectory": traj, "runtime": time.time() - t0}


if __name__ == "__main__":
    from bdtp_data import make_instance

    inst = make_instance(N=3, L=8, demand_pattern="constant", seed=7)
    rot = rule_of_thumb(inst)
    grd = myopic_greedy(inst)
    names = inst["donation_names"]
    print(f"rule-of-thumb: cost={rot['obj']:.2f}  type={names[rot['type']]}  "
          f"runtime={rot['runtime']:.3f}s")
    print(f"myopic greedy: cost={grd['obj']:.2f}  runtime={grd['runtime']:.3f}s")
