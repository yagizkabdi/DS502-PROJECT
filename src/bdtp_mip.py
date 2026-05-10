import gurobipy as gp
from gurobipy import GRB


def build_and_solve_mip(inst, time_limit=60, verbose=False):
    I_types = inst["I"]
    J = inst["J"]
    K = inst["K"]
    T = inst["T"]
    L = inst["L"]
    a = inst["a"]
    c = inst["c"]
    h = inst["h"]
    d_tilde = inst["d_tilde"]
    s = inst["s"]
    shelf = inst["shelf"]
    demand = inst["demand"]
    E = inst["E"]

    none_idx = len(I_types) - 1

    m = gp.Model("BDTP")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = time_limit

    x = m.addVars(K, I_types, T, vtype=GRB.BINARY, name="x")
    Iv = m.addVars(J, T, lb=0, name="I")
    Pv = m.addVars(J, T, lb=0, name="P")

    m.setObjective(
        gp.quicksum(c[i] * x[k, i, t] for k in K for i in I_types for t in T)
        + gp.quicksum(h[j] * Iv[j, t] for j in J for t in T)
        + gp.quicksum(d_tilde[j] * Pv[j, t] for j in J for t in T),
        GRB.MINIMIZE,
    )

    for k in K:
        for t in T:
            m.addConstr(gp.quicksum(x[k, i, t] for i in I_types) == 1,
                        name=f"assign_{k}_{t}")

    for k in K:
        for t in T:
            for i in I_types:
                if i == none_idx:
                    continue
                wait = min(s[i][ip] for ip in I_types if ip != none_idx)
                for dt in range(1, wait):
                    tp = t + dt
                    if tp >= L:
                        break
                    for ip in I_types:
                        if ip == none_idx:
                            continue
                        m.addConstr(
                            x[k, i, t] + x[k, ip, tp] <= 1,
                            name=f"defer_{k}_{t}_{i}_{tp}_{ip}",
                        )

    for j in J:
        for t in T:
            prev = Iv[j, t - 1] if t > 0 else 0
            collected = gp.quicksum(a[i][j] * x[k, i, t]
                                    for k in K for i in I_types)
            m.addConstr(prev - Iv[j, t] - Pv[j, t] + collected == demand[(j, t)],
                        name=f"balance_{j}_{t}")

    for k in K:
        for j in J:
            m.addConstr(
                gp.quicksum(a[i][j] * x[k, i, t] for i in I_types for t in T)
                <= E[j],
                name=f"cap_{k}_{j}",
            )

    for j in J:
        for t in T:
            if t < shelf[j]:
                continue
            m.addConstr(
                Iv[j, t]
                <= gp.quicksum(
                    a[i][j] * x[k, i, tp]
                    for k in K for i in I_types
                    for tp in range(t - shelf[j] + 1, t + 1)
                ),
                name=f"shelf_{j}_{t}",
            )

    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        return {"status": m.Status, "obj": None, "schedule": None}

    schedule = {}
    for k in K:
        schedule[k] = []
        for t in T:
            picked = None
            for i in I_types:
                if x[k, i, t].X > 0.5:
                    picked = i
                    break
            schedule[k].append(picked)

    inv = {(j, t): Iv[j, t].X for j in J for t in T}
    disp = {(j, t): Pv[j, t].X for j in J for t in T}

    return {
        "status": m.Status,
        "obj": m.ObjVal,
        "schedule": schedule,
        "inventory": inv,
        "disposal": disp,
        "runtime": m.Runtime,
    }


def print_result(inst, res):
    if res["obj"] is None:
        print("MIP did not return a solution. Status:", res["status"])
        return
    print(f"MIP status : {res['status']}")
    print(f"objective  : {res['obj']:.2f}")
    print(f"runtime    : {res['runtime']:.3f}s")
    names = inst["donation_names"]
    print("donor schedules:")
    for k, sch in res["schedule"].items():
        print(f"  donor {k}: " + " ".join(names[i] for i in sch))
    print("end-of-period inventory:")
    for j in inst["J"]:
        row = [f"{res['inventory'][(j, t)]:.1f}" for t in inst["T"]]
        print(f"  {inst['product_names'][j]:>10s}: {row}")
    print("disposal:")
    for j in inst["J"]:
        row = [f"{res['disposal'][(j, t)]:.1f}" for t in inst["T"]]
        print(f"  {inst['product_names'][j]:>10s}: {row}")


if __name__ == "__main__":
    from bdtp_data import make_instance
    inst = make_instance(N=3, L=8)
    res = build_and_solve_mip(inst, verbose=False)
    print_result(inst, res)
