"""Backward induction for the deterministic BDTP MDP."""

import time

from mdp import feasible_actions, step, is_terminal, terminal_cost, initial_state


def solve(inst, verbose=False):
    memo = {}

    def V(state):
        if is_terminal(inst, state):
            return terminal_cost(inst, state), None
        if state in memo:
            return memo[state]

        best_cost = float("inf")
        best_action = None

        for action in feasible_actions(inst, state):
            next_state, stage_cost, _ = step(inst, state, action)
            if stage_cost == float("inf"):
                continue
            future_cost, _ = V(next_state)
            total = stage_cost + future_cost
            if total < best_cost:
                best_cost = total
                best_action = action

        memo[state] = (best_cost, best_action)
        if verbose and len(memo) % 1000 == 0:
            print(f"  ... explored {len(memo)} states")
        return best_cost, best_action

    s0 = initial_state(inst)
    t0 = time.time()
    opt_cost, _ = V(s0)
    runtime = time.time() - t0

    trajectory = []
    s = s0
    while not is_terminal(inst, s):
        _, a = memo[s]
        next_state, stage_cost, info = step(inst, s, a)
        trajectory.append({
            "t": s[0],
            "state_cooldown": s[1],
            "action": a,
            "stage_cost": stage_cost,
            "collected": info.get("collected"),
            "disposed": info.get("disposed"),
        })
        s = next_state

    return {
        "opt_cost": opt_cost,
        "runtime": runtime,
        "states_explored": len(memo),
        "trajectory": trajectory,
    }


def print_trajectory(inst, result):
    names = inst["donation_names"]
    print(f"DP optimal cost   : {result['opt_cost']:.2f}")
    print(f"DP runtime        : {result['runtime']:.3f}s")
    print(f"states explored   : {result['states_explored']}")
    print("trajectory:")
    print(f"  {'t':>2s}  {'action':<40s}  {'cost':>8s}")
    for step_info in result["trajectory"]:
        act = " ".join(names[i] for i in step_info["action"])
        print(f"  {step_info['t']:>2d}  {act:<40s}  {step_info['stage_cost']:>8.2f}")


if __name__ == "__main__":
    from bdtp_data import make_instance
    inst = make_instance(N=2, L=5)
    res = solve(inst, verbose=True)
    print_trajectory(inst, res)
