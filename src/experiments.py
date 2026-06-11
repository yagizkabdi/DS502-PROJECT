"""Method comparison grid + MIP scaling study, dumps CSV + PNG into ../results/.

Records wall and CPU runtime for every method, plus the MIP optimality gap and
best bound from Gurobi.
"""

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

COMPARE_SIZES = [(2, 8), (3, 10), (4, 14), (5, 18), (6, 21), (8, 26)]
COMPARE_DEMANDS = ["constant", "increasing", "random"]
COMPARE_SEED = 7
COMPARE_MIP_TL = 60

MIP_SCALING_SIZES = [(3, 14), (5, 21), (8, 30), (10, 40), (12, 50), (15, 60), (20, 80)]
MIP_SCALING_MIP_TL = 120

GREEDY_MAX_N = 8


def _timed(fn):
    w0, c0 = time.time(), time.process_time()
    out = fn()
    return out, time.time() - w0, time.process_time() - c0


def _blank_row(method, status=None):
    return {
        "method": method, "obj": None, "runtime": None, "cpu_runtime": None,
        "mip_gap": None, "obj_bound": None, "status": status, "ga_seed": None,
    }


def _run_mip(inst, time_limit):
    try:
        mip, wall, cpu = _timed(
            lambda: build_and_solve_mip(inst, time_limit=time_limit, verbose=False)
        )
    except Exception as exc:
        print(f"   MIP skipped: {type(exc).__name__}: {exc}")
        return _blank_row("MIP", status="ERROR")

    return {
        "method": "MIP",
        "obj": mip["obj"],
        "runtime": mip.get("runtime", wall),
        "cpu_runtime": mip.get("cpu_runtime", cpu),
        "mip_gap": mip.get("mip_gap"),
        "obj_bound": mip.get("obj_bound"),
        "status": mip.get("status"),
        "ga_seed": None,
    }


def _run_methods(inst, run_dp=False, mip_time_limit=COMPARE_MIP_TL):
    rows = [_run_mip(inst, mip_time_limit)]

    rot, wall, cpu = _timed(lambda: rule_of_thumb(inst))
    rows.append({**_blank_row("RuleOfThumb"), "obj": rot["obj"],
                 "runtime": wall, "cpu_runtime": cpu})

    if inst["N"] <= GREEDY_MAX_N:
        grd, wall, cpu = _timed(lambda: myopic_greedy(inst))
        rows.append({**_blank_row("MyopicGreedy"), "obj": grd["obj"],
                     "runtime": wall, "cpu_runtime": cpu})
    else:
        rows.append(_blank_row("MyopicGreedy", status=f"skipped(N>{GREEDY_MAX_N})"))

    if run_dp:
        dp, wall, cpu = _timed(lambda: dp_solve(inst))
        rows.append({**_blank_row("MDP-DP"), "obj": dp["opt_cost"],
                     "runtime": wall, "cpu_runtime": cpu})

    if inst["N"] <= GREEDY_MAX_N:
        for seed in GA_SEEDS:
            ga, wall, cpu = _timed(
                lambda s=seed: run_ga(inst, pop_size=GA_POP, generations=GA_GEN, seed=s)
            )
            rows.append({**_blank_row("GA"), "obj": ga["obj"],
                         "runtime": wall, "cpu_runtime": cpu, "ga_seed": seed})
    else:
        rows.append(_blank_row("GA", status=f"skipped(N>{GREEDY_MAX_N})"))

    return rows


def _is_tractable_for_dp(inst):
    return inst["N"] <= 2 and inst["L"] <= 5


def _fmt(x, nd=2):
    return f"{x:.{nd}f}" if x is not None and np.isfinite(x) else "n/a"


def run_grid():
    """Run every method over COMPARE_SIZES, write results.csv."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []
    for N, L in COMPARE_SIZES:
        for pattern in COMPARE_DEMANDS:
            inst = make_instance(N=N, L=L, demand_pattern=pattern, seed=COMPARE_SEED)
            print(f"-- N={N} L={L} demand={pattern} (N*L={N * L}) --")
            rows = _run_methods(inst, run_dp=_is_tractable_for_dp(inst))
            for r in rows:
                r.update({"N": N, "L": L, "demand": pattern,
                          "instance_seed": COMPARE_SEED})
                all_rows.append(r)
                suffix = (f" (seed={r['ga_seed']})"
                          if r["ga_seed"] is not None else "")
                gap_str = (f"  gap={r['mip_gap'] * 100:.2f}%"
                           if r.get("mip_gap") is not None else "")
                print(f"   {r['method']:12s}{suffix:>10s}  "
                      f"obj={_fmt(r['obj']):>10s}  "
                      f"wall={_fmt(r['runtime'], 3):>8s}s  "
                      f"cpu={_fmt(r['cpu_runtime'], 3):>8s}s{gap_str}")
            pd.DataFrame(all_rows).to_csv(
                os.path.join(RESULTS_DIR, "results.csv"), index=False)
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")
    return all_rows


def run_mip_scaling():
    """Run the MIP over MIP_SCALING_SIZES, write mip_scaling.csv."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    print("\n== MIP scaling study (constant demand) ==")
    for N, L in MIP_SCALING_SIZES:
        inst = make_instance(N=N, L=L, demand_pattern="constant", seed=COMPARE_SEED)
        n_vars = N * len(inst["I"]) * L + 2 * len(inst["J"]) * L
        r = _run_mip(inst, MIP_SCALING_MIP_TL)
        r.update({"N": N, "L": L, "size": N * L, "approx_vars": n_vars})
        rows.append(r)
        gap_str = (f"  gap={r['mip_gap'] * 100:.2f}%"
                   if r.get("mip_gap") is not None else "")
        bound_str = (f"  bound={_fmt(r['obj_bound'])}"
                     if r.get("obj_bound") is not None else "")
        print(f"-- N={N:2d} L={L:2d}  N*L={N * L:3d}  ~vars={n_vars:4d}  "
              f"status={str(r['status']):>10s}  obj={_fmt(r['obj']):>10s}  "
              f"wall={_fmt(r['runtime'], 3):>8s}s  cpu={_fmt(r['cpu_runtime'], 3):>8s}s"
              f"{bound_str}{gap_str}")
    out = os.path.join(RESULTS_DIR, "mip_scaling.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")
    return rows


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
        {"method": "MIP", "obj": mip["obj"], "runtime": mip["runtime"],
         "cpu_runtime": mip.get("cpu_runtime"), "mip_gap": mip.get("mip_gap")},
        {"method": "MDP-DP", "obj": dp["opt_cost"], "runtime": dp["runtime"],
         "cpu_runtime": None, "mip_gap": None},
        {"method": "RuleOfThumb", "obj": rot["obj"], "runtime": rot["runtime"],
         "cpu_runtime": None, "mip_gap": None},
        {"method": "MyopicGreedy", "obj": grd["obj"], "runtime": grd["runtime"],
         "cpu_runtime": None, "mip_gap": None},
        {"method": "GA", "obj": ga["obj"], "runtime": ga["runtime"],
         "cpu_runtime": None, "mip_gap": None},
    ])
    print(table.to_string(index=False))
    out = os.path.join(RESULTS_DIR, "sanity_table.csv")
    table.to_csv(out, index=False)
    print(f"wrote {out}")


def _aggregate(df):
    df = df.copy()
    df["obj"] = df["obj"].replace([np.inf, -np.inf], np.nan)
    keep = ["N", "L", "demand", "method", "obj_mean", "obj_min", "obj_max",
            "runtime_mean", "cpu_runtime_mean", "mip_gap", "obj_bound"]

    base = df[df["method"] != "GA"].copy()
    base["obj_mean"] = base["obj"]
    base["obj_min"] = base["obj"]
    base["obj_max"] = base["obj"]
    base["runtime_mean"] = base["runtime"]
    base["cpu_runtime_mean"] = base["cpu_runtime"]

    ga = (df[df["method"] == "GA"]
          .groupby(["N", "L", "demand"], as_index=False)
          .agg(obj_mean=("obj", "mean"),
               obj_min=("obj", "min"),
               obj_max=("obj", "max"),
               runtime_mean=("runtime", "mean"),
               cpu_runtime_mean=("cpu_runtime", "mean")))
    ga["method"] = "GA"
    ga["mip_gap"] = np.nan
    ga["obj_bound"] = np.nan

    out = pd.concat([base[keep], ga[keep]], ignore_index=True)
    return out


def plot_objective_vs_method(df, demand="constant"):
    sub = df[df["demand"] == demand].copy()
    if sub.empty:
        return
    pivot = sub.pivot_table(index=["N", "L"], columns="method", values="obj_mean")
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


def _runtime_plot(df, value, ylabel, title, fname):
    sub = df[df["demand"] == "constant"].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for method, g in sub.groupby("method"):
        g = g.sort_values(["N", "L"])
        labels = [f"N{n}/L{l}" for n, l in zip(g["N"], g["L"])]
        ax.plot(labels, g[value], marker="o", label=method)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, fname)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def plot_runtime_vs_size(df):
    _runtime_plot(df, "runtime_mean", "wall runtime [s, log]",
                  "Wall-clock runtime vs. instance size  (constant demand)",
                  "runtime_vs_size.png")


def plot_cpu_runtime_vs_size(df):
    _runtime_plot(df, "cpu_runtime_mean", "CPU runtime [s, log]",
                  "CPU runtime vs. instance size  (constant demand)",
                  "cpu_runtime_vs_size.png")


def plot_gap_to_mip(df):
    sub = df[df["demand"] == "constant"].copy()
    if sub.empty:
        return
    pivot = sub.pivot_table(index=["N", "L"], columns="method", values="obj_mean")
    if "MIP" not in pivot.columns:
        return
    gap = pivot.subtract(pivot["MIP"], axis=0).divide(pivot["MIP"], axis=0) * 100
    gap = gap.drop(columns=["MIP"], errors="ignore")
    ax = gap.plot.bar(figsize=(8, 4))
    ax.set_ylabel("gap to MIP optimum [%]")
    ax.set_title("Heuristic gap to MIP optimum  (constant demand)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "gap_to_mip.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def plot_mip_optimality_gap(df):
    sub = df[(df["method"] == "MIP") & df["mip_gap"].notna()].copy()
    if sub.empty:
        return
    sub = sub.sort_values(["N", "L"])
    pivot = sub.pivot_table(index=["N", "L"], columns="demand", values="mip_gap") * 100
    ax = pivot.plot.bar(figsize=(8, 4))
    ax.set_ylabel("MIP optimality gap [%]")
    ax.set_title("MIP optimality gap by instance size\n(0% = proven optimal within time limit)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "mip_optimality_gap.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def plot_mip_scaling(scaling_csv):
    if not os.path.exists(scaling_csv):
        return
    df = pd.read_csv(scaling_csv).sort_values("size")
    solved = df[df["runtime"].notna()]
    if solved.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"N{n}/L{l}" for n, l in zip(solved["N"], solved["L"])]
    ax.plot(labels, solved["runtime"], marker="o", label="wall runtime")
    ax.plot(labels, solved["cpu_runtime"], marker="s", label="CPU runtime")
    ax.set_ylabel("runtime [s]")
    ax.set_title("MIP solve time vs. instance size  (constant demand)")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "mip_runtime_vs_size.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def make_figures():
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        agg = _aggregate(df)
        agg.to_csv(os.path.join(RESULTS_DIR, "results_aggregated.csv"), index=False)
        for d in agg["demand"].unique():
            plot_objective_vs_method(agg, demand=d)
        plot_runtime_vs_size(agg)
        plot_cpu_runtime_vs_size(agg)
        plot_gap_to_mip(agg)
        plot_mip_optimality_gap(df)
    else:
        print("no results.csv yet")
    plot_mip_scaling(os.path.join(RESULTS_DIR, "mip_scaling.csv"))


if __name__ == "__main__":
    np.random.seed(7)
    small_dp_sanity_table()
    print()
    run_grid()
    run_mip_scaling()
    print()
    make_figures()
