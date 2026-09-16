"""混合方法：SA 预热 → CP-SAT（初始提示 + 上界 + 解析下界割）。

与 benchmark.py 同一问题定义、同一算例，总时间与单一方法相同。
解析下界：块向右、向上膨胀 δ 后两两不相交，且都在 (W+δ)×(H+δ) 内，故
    Σ(w_i+δ)(h_i+δ) ≤ (W+δ)(H+δ)。
在 H ≤ W ≤ κH（或对称情形）下，W·H 的最小值取在 W = κH 的边界上。

运行：.venv/Scripts/python hybrid.py [总秒数，默认 60]
"""
import json
import math
import sys
import time

from ortools.sat.python import cp_model

import benchmark as B

TOTAL = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0


def analytic_lb(inst):
    k, d = B.KAPPA, B.DELTA
    S = sum((w + d) * (h + d) for w, h in zip(inst["w"], inst["h"]))
    H = (-d * (k + 1) + math.sqrt((d * (k + 1)) ** 2 - 4 * k * (d * d - S))) / (2 * k)
    return math.ceil(k * H * H - 1e-9)


def run_hybrid(inst, total):
    t0 = time.time()
    sa = B.run_sa(inst, total / 2)
    W0, H0 = B.area_of(inst, sa["x"], sa["y"])
    ub = W0 * H0
    lb_cut = analytic_lb(inst)

    w, h, n = inst["w"], inst["h"], len(inst["w"])
    Wmax = sum(w) + B.DELTA * n
    Hmax = sum(h) + B.DELTA * n
    m = cp_model.CpModel()
    X = [m.NewIntVar(0, Wmax - w[i], f"x{i}") for i in range(n)]
    Y = [m.NewIntVar(0, Hmax - h[i], f"y{i}") for i in range(n)]
    XI = [m.NewIntervalVar(X[i], w[i] + B.DELTA, X[i] + w[i] + B.DELTA, f"xi{i}") for i in range(n)]
    YI = [m.NewIntervalVar(Y[i], h[i] + B.DELTA, Y[i] + h[i] + B.DELTA, f"yi{i}") for i in range(n)]
    m.AddNoOverlap2D(XI, YI)
    W = m.NewIntVar(max(w), Wmax, "W")
    H = m.NewIntVar(max(h), Hmax, "H")
    for i in range(n):
        m.Add(X[i] + w[i] <= W)
        m.Add(Y[i] + h[i] <= H)
    m.Add(W <= B.KAPPA * H)
    m.Add(H <= B.KAPPA * W)
    # 对称性破除与 SA 提示可能冲突，这里不加，保证提示可直接使用
    A = m.NewIntVar(lb_cut, ub, "A")                     # 解析下界割 + SA 上界
    m.AddMultiplicationEquality(A, [W, H])
    m.Minimize(A)
    for i in range(n):
        m.AddHint(X[i], sa["x"][i])
        m.AddHint(Y[i], sa["y"][i])
    m.AddHint(W, W0)
    m.AddHint(H, H0)

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = total - (time.time() - t0)
    s.parameters.num_workers = 8
    st = s.Solve(m)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        x = [s.Value(v) for v in X]
        y = [s.Value(v) for v in Y]
        Wf, Hf = B.area_of(inst, x, y)
        best = min(ub, Wf * Hf)
    else:
        x, y, best = sa["x"], sa["y"], ub
    lb = max(lb_cut, s.BestObjectiveBound()) if st != cp_model.MODEL_INVALID else lb_cut
    return {"sa_area_m2": ub / 100, "area_m2": best / 100, "lower_bound_m2": round(lb / 100, 1),
            "analytic_lb_m2": lb_cut / 100, "proven": st == cp_model.OPTIMAL,
            "gap_pct": round(100 * (best - lb) / best, 2), "time_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    out = []
    for inst in B.INSTANCES:
        r = run_hybrid(inst, TOTAL)
        r["instance"] = inst["name"]
        out.append(r)
        print(inst["name"], r, flush=True)
    (B.HERE / "results_hybrid.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
