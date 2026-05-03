"""GA for the BDTP."""

import random
import time

from mdp import initial_state, step, is_terminal
from heuristics import _greedy_action, _fixed_type_policy


PENALTY = 1e6


def _repair_and_simulate(inst, chromosome):
    N = inst["N"]
    L = inst["L"]
    none_idx = len(inst["I"]) - 1

    repaired = list(chromosome)
    state = initial_state(inst)
    total_cost = 0.0

    while not is_terminal(inst, state):
        t, cooldown, used, _ = state
        action = []
        for k in range(N):
            i = repaired[k * L + t]
            if cooldown[k] > 0:
                i = none_idx
            else:
                ok = True
                for j in inst["J"]:
                    if used[k][j] + inst["a"][i][j] > inst["E"][j] + 1e-9:
                        ok = False
                        break
                if not ok:
                    i = none_idx
            repaired[k * L + t] = i
            action.append(i)
        next_state, stage_cost, info = step(inst, state, tuple(action))
        if stage_cost == float("inf") or info.get("infeasible"):
            return total_cost + PENALTY * (L - t), repaired
        total_cost += stage_cost
        state = next_state

    return total_cost, repaired


def _random_individual(inst, rng):
    n = inst["N"] * inst["L"]
    n_types = len(inst["I"])
    return [rng.randrange(n_types) for _ in range(n)]


def _individual_from_policy(inst, policy_fn):
    L = inst["L"]
    N = inst["N"]
    none_idx = len(inst["I"]) - 1
    chromosome = [none_idx] * (N * L)
    state = initial_state(inst)
    while not is_terminal(inst, state):
        t = state[0]
        action = policy_fn(state, inst)
        if action is None:
            break
        for k in range(N):
            chromosome[k * L + t] = action[k]
        next_state, stage_cost, info = step(inst, state, action)
        if stage_cost == float("inf") or info.get("infeasible"):
            break
        state = next_state
    return chromosome


def _seed_population(inst, pop_size, rng):
    seeds = [_individual_from_policy(inst, _greedy_action)]
    none_idx = len(inst["I"]) - 1
    for i in inst["I"]:
        if i == none_idx:
            continue
        types = [i] * inst["N"]
        seeds.append(_individual_from_policy(inst, _fixed_type_policy(types)))
    seeds = seeds[:pop_size]
    while len(seeds) < pop_size:
        seeds.append(_random_individual(inst, rng))
    return seeds


def _tournament(pop, fits, k, rng):
    contenders = rng.sample(range(len(pop)), k)
    winner = min(contenders, key=lambda idx: fits[idx])
    return list(pop[winner])


def _two_point_crossover(p1, p2, rng):
    n = len(p1)
    if n < 2:
        return list(p1), list(p2)
    a, b = sorted(rng.sample(range(n), 2))
    return p1[:a] + p2[a:b] + p1[b:], p2[:a] + p1[a:b] + p2[b:]


def _mutate(individual, n_types, rate, rng):
    for idx in range(len(individual)):
        if rng.random() < rate:
            individual[idx] = rng.randrange(n_types)
    return individual


def run_ga(inst, pop_size=80, generations=150, cx_rate=0.8,
           mut_rate=None, tourn_k=3, seed=7, verbose=False):
    rng = random.Random(seed)
    n_types = len(inst["I"])
    n_genes = inst["N"] * inst["L"]
    if mut_rate is None:
        mut_rate = 1.0 / max(1, n_genes)

    t0 = time.time()
    population = _seed_population(inst, pop_size, rng)
    evaluated = [_repair_and_simulate(inst, ind) for ind in population]
    fits = [c for c, _ in evaluated]
    population = [r for _, r in evaluated]

    best_idx = min(range(pop_size), key=lambda i: fits[i])
    best_fit = fits[best_idx]
    best_ind = list(population[best_idx])
    history = [best_fit]

    for gen in range(generations):
        new_pop = [list(best_ind)]
        while len(new_pop) < pop_size:
            p1 = _tournament(population, fits, tourn_k, rng)
            p2 = _tournament(population, fits, tourn_k, rng)
            if rng.random() < cx_rate:
                c1, c2 = _two_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1, p2
            _mutate(c1, n_types, mut_rate, rng)
            _mutate(c2, n_types, mut_rate, rng)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        evaluated = [_repair_and_simulate(inst, ind) for ind in new_pop]
        fits = [c for c, _ in evaluated]
        population = [r for _, r in evaluated]

        gen_best_idx = min(range(pop_size), key=lambda i: fits[i])
        if fits[gen_best_idx] < best_fit:
            best_fit = fits[gen_best_idx]
            best_ind = list(population[gen_best_idx])
        history.append(best_fit)

        if verbose and (gen + 1) % 25 == 0:
            print(f"  gen {gen+1:4d}  best={best_fit:.2f}")

    runtime = time.time() - t0
    schedule = {k: [best_ind[k * inst["L"] + t] for t in inst["T"]]
                for k in inst["K"]}
    return {
        "obj": best_fit,
        "schedule": schedule,
        "history": history,
        "runtime": runtime,
        "params": {
            "pop_size": pop_size,
            "generations": generations,
            "cx_rate": cx_rate,
            "mut_rate": mut_rate,
            "tourn_k": tourn_k,
            "seed": seed,
        },
    }


if __name__ == "__main__":
    from bdtp_data import make_instance

    inst = make_instance(N=3, L=8, demand_pattern="constant", seed=7)
    res = run_ga(inst, pop_size=60, generations=80, seed=1, verbose=True)
    names = inst["donation_names"]
    print(f"\nGA best objective : {res['obj']:.2f}")
    print(f"GA runtime        : {res['runtime']:.3f}s")
    print("schedules:")
    for k, sch in res["schedule"].items():
        print(f"  donor {k}: " + " ".join(names[i] for i in sch))
