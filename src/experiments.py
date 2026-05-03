"""Run the (N, L, demand) grid and dump CSV + PNG into ../results/."""

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bdtp_data import make_instance
from bdtp_mip import build_and_solve_mip
from bdtp_ga import run_ga
from heuristics import rule_of_thumb, myopic_greedy
from dp import solve as dp_solve


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
GA_SEEDS = [1, 2, 3]
GA_POP = 80
GA_GEN = 150

GRID = {
    "N": [3, 5, 8],
    "L": [8, 14, 21],
    "demand": ["constant", "increasing", "random"],
    "instance_seed": [7],
}


def _run_methods(inst, run_dp=False):
    rows = []

    t = time.time()
    try:
        mip = build_and_solve_mip(inst, time_limit=120, verbose=False)
        mip_obj = mip["obj"]
        mip_rt = mip.get("runtime", time.time() - t)
    except Exception as exc:
        print(f"   MIP skipped: {type(exc).__name__}: {exc}")
        mip_obj, mip_rt = None, time.time() - t
    rows.append({
        "method": "MIP",
        "obj": mip_obj,
        "runtime": mip_rt,
        "ga_seed": None,
    })

    rot = rule_of_thumb(inst)
    rows.append({"method": "RuleOfThumb", "obj": rot["obj"],
                 "runtime": rot["runtime"], "ga_seed": None})

    grd = myopic_greedy(inst)
    rows.append({"method": "MyopicGreedy", "obj": grd["obj"],
                 "runtime": grd["runtime"], "ga_seed": None})

    if run_dp:
        dp = dp_solve(inst)
        rows.append({"method": "MDP-DP", "obj": dp["opt_cost"],
                     "runtime": dp["runtime"], "ga_seed": None})

    for seed in GA_SEEDS:
        ga = run_ga(inst, pop_size=GA_POP, generations=GA_GEN, seed=seed)
        rows.append({"method": "GA", "obj": ga["obj"],
                     "runtime": ga["runtime"], "ga_seed": seed})

    return rows


def _is_tractable_for_dp(inst):
    return inst["N"] <= 2 and inst["L"] <= 5


def run_grid():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []
    for N in GRID["N"]:
        for L in GRID["L"]:
            for pattern in GRID["demand"]:
                for iseed in GRID["instance_seed"]:
                    inst = make_instance(N=N, L=L,
                                         demand_pattern=pattern,
                                         seed=iseed)
                    print(f"-- N={N} L={L} demand={pattern} iseed={iseed} --")
                    rows = _run_methods(inst,
                                        run_dp=_is_tractable_for_dp(inst))
                    for r in rows:
                        r.update({
                            "N": N, "L": L, "demand": pattern,
                            "instance_seed": iseed,
                        })
                        all_rows.append(r)
                        suffix = (f" (seed={r['ga_seed']})"
                                  if r["ga_seed"] is not None else "")
                        obj_str = (f"{r['obj']:.2f}"
                                   if r["obj"] is not None else "n/a")
                        print(f"   {r['method']:12s}{suffix:>10s}  "
                              f"obj={obj_str:>10s}  rt={r['runtime']:.2f}s")
                    pd.DataFrame(all_rows).to_csv(
                        os.path.join(RESULTS_DIR, "results.csv"),
                        index=False,
                    )
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")
    return all_rows


def small_dp_sanity_table():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    inst = make_instance(N=2, L=4, demand_pattern="constant", seed=7)
    print("-- sanity table on N=2 L=4 --")
    mip = build_and_solve_mip(inst, time_limit=30)
    dp = dp_solve(inst)
    rot = rule_of_thumb(inst)
    grd = myopic_greedy(inst)
    ga = run_ga(inst, pop_size=80, generations=100, seed=1)
    table = pd.DataFrame([
        {"method": "MIP", "obj": mip["obj"], "runtime": mip["runtime"]},
        {"method": "MDP-DP", "obj": dp["opt_cost"], "runtime": dp["runtime"]},
        {"method": "RuleOfThumb", "obj": rot["obj"], "runtime": rot["runtime"]},
        {"method": "MyopicGreedy", "obj": grd["obj"], "runtime": grd["runtime"]},
        {"method": "GA", "obj": ga["obj"], "runtime": ga["runtime"]},
    ])
    print(table.to_string(index=False))
    out = os.path.join(RESULTS_DIR, "sanity_table.csv")
    table.to_csv(out, index=False)
    print(f"wrote {out}")


def _aggregate(df):
    df = df.copy()
    df["obj"] = df["obj"].replace([np.inf, -np.inf], np.nan)
    base = df[df["method"] != "GA"].copy()
    base["obj_mean"] = base["obj"]
    base["obj_min"] = base["obj"]
    base["obj_max"] = base["obj"]
    base["runtime_mean"] = base["runtime"]

    ga = (df[df["method"] == "GA"]
          .groupby(["N", "L", "demand"], as_index=False)
          .agg(obj_mean=("obj", "mean"),
               obj_min=("obj", "min"),
               obj_max=("obj", "max"),
               runtime_mean=("runtime", "mean")))
    ga["method"] = "GA"
    ga["instance_seed"] = df["instance_seed"].iloc[0]
    out = pd.concat([base[["N", "L", "demand", "method",
                           "obj_mean", "obj_min", "obj_max", "runtime_mean"]],
                     ga[["N", "L", "demand", "method",
                         "obj_mean", "obj_min", "obj_max", "runtime_mean"]]],
                    ignore_index=True)
    return out


def plot_objective_vs_method(df, demand="constant"):
    sub = df[df["demand"] == demand].copy()
    if sub.empty:
        return
    pivot = sub.pivot_table(index=["N", "L"], columns="method",
                            values="obj_mean")
    pivot = pivot[[c for c in ["MIP", "GA", "MyopicGreedy", "RuleOfThumb"]
                   if c in pivot.columns]]
    ax = pivot.plot.bar(figsize=(8, 4))
    ax.set_ylabel("total cost")
    ax.set_title(f"Objective by method  ({demand} demand)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"obj_by_method_{demand}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def plot_runtime_vs_size(df):
    sub = df[df["demand"] == "constant"].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for method, g in sub.groupby("method"):
        g = g.sort_values(["N", "L"])
        labels = [f"N{n}/L{l}" for n, l in zip(g["N"], g["L"])]
        ax.plot(labels, g["runtime_mean"], marker="o", label=method)
    ax.set_yscale("log")
    ax.set_ylabel("runtime [s, log]")
    ax.set_title("Runtime vs. instance size  (constant demand)")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "runtime_vs_size.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def plot_gap_to_mip(df):
    sub = df[df["demand"] == "constant"].copy()
    if sub.empty:
        return
    pivot = sub.pivot_table(index=["N", "L"], columns="method",
                            values="obj_mean")
    if "MIP" not in pivot.columns:
        return
    gap = pivot.subtract(pivot["MIP"], axis=0).divide(pivot["MIP"], axis=0) * 100
    gap = gap.drop(columns=["MIP"], errors="ignore")
    ax = gap.plot.bar(figsize=(8, 4))
    ax.set_ylabel("gap to MIP optimum [%]")
    ax.set_title("Optimality gap by method  (constant demand)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "gap_to_mip.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def make_figures():
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    if not os.path.exists(csv_path):
        print("no results.csv yet")
        return
    df = pd.read_csv(csv_path)
    agg = _aggregate(df)
    agg.to_csv(os.path.join(RESULTS_DIR, "results_aggregated.csv"), index=False)
    for d in agg["demand"].unique():
        plot_objective_vs_method(agg, demand=d)
    plot_runtime_vs_size(agg)
    plot_gap_to_mip(agg)


if __name__ == "__main__":
    np.random.seed(7)
    small_dp_sanity_table()
    print()
    run_grid()
    print()
    make_figures()
