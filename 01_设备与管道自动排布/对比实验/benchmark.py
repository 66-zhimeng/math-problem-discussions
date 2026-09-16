"""摆放子问题求解方法对比实验。

问题：矩形设备块两两不重叠（净距 δ），0° 朝向，最小化外接矩形面积 W·H，
      并满足长宽比约束 W ≤ κH、H ≤ κW（主文档 12.5）。
      W、H 为“占地矩形”变量：包住全部设备，且满足长宽比。给定设备坐标时，
      最小占地矩形 = 实际外包尺寸按长宽比补足后的矩形（见 room_of）。
方法：
  SA      序列对 + 最长路压紧 + 模拟退火（启发式，无下界）
  MILP    析取大 M 模型，HiGHS；ε 约束法（逐步收紧 W 上限、最小化 H）枚举 Pareto 前沿
  Z3      SMT 优化（整数线性算术 + 析取），同样的 ε 约束法
  CPSAT   OR-Tools CP-SAT，NoOverlap2D + 面积乘积约束，直接最小化 W·H
所有尺寸为 100 mm 的整数倍，坐标以 100 mm 为单位建模。最长路压紧的解必在该网格上，
因此以 100 mm 为单位不损失最优性。

运行：.venv/Scripts/python benchmark.py [每个方法每个算例的秒数，默认 60]
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
TIME_LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
DELTA = 8  # 800 mm
KAPPA = 2  # 长宽比上限 κ，实验取值


# ============================================================ 算例
def make_instance(name, spec):
    """spec: [(类型名, 宽, 深, 数量)]，尺寸单位 100 mm。同类型的块完全相同，可互换。"""
    names, w, h, group = [], [], [], []
    for gid, (t, bw, bh, cnt) in enumerate(spec):
        for k in range(cnt):
            names.append(f"{t}{k + 1}")
            w.append(bw); h.append(bh); group.append(gid)
    return {"name": name, "names": names, "w": w, "h": h, "group": group}


INSTANCES = [
    make_instance("8块", [("冷却塔组", 120, 30, 1), ("制冷链", 60, 40, 4), ("分水器", 25, 8, 1),
                         ("集水器", 25, 8, 1), ("定压补水", 15, 12, 1)]),
    make_instance("20块", [("冷却塔组", 120, 30, 2), ("制冷链", 60, 40, 8), ("分水器", 25, 8, 2),
                          ("集水器", 25, 8, 2), ("定压补水", 15, 12, 2), ("板换", 30, 20, 2),
                          ("水处理", 20, 15, 2)]),
    make_instance("40块", [("冷却塔组", 120, 30, 4), ("制冷链", 60, 40, 16), ("分水器", 25, 8, 4),
                          ("集水器", 25, 8, 4), ("定压补水", 15, 12, 2), ("板换", 30, 20, 4),
                          ("水处理", 20, 15, 4), ("配电柜", 40, 10, 2)]),
]


def identical_pairs(inst):
    """同组相邻编号的块对，用于对称性破除 x_a ≤ x_b。"""
    g = inst["group"]
    return [(i, i + 1) for i in range(len(g) - 1) if g[i] == g[i + 1]]


def verify(inst, x, y):
    w, h, n = inst["w"], inst["h"], len(inst["w"])
    for i in range(n):
        for j in range(i + 1, n):
            gx = max(x[j] - (x[i] + w[i]), x[i] - (x[j] + w[j]))
            gy = max(y[j] - (y[i] + h[i]), y[i] - (y[j] + h[j]))
            if max(gx, gy) < DELTA:
                return False
    return min(x) >= 0 and min(y) >= 0


def room_of(W, H):
    """实际外包 W×H → 满足长宽比的最小占地矩形。"""
    return max(W, -(-H // KAPPA)), max(H, -(-W // KAPPA))


def area_of(inst, x, y):
    W = max(xi + wi for xi, wi in zip(x, inst["w"]))
    H = max(yi + hi for yi, hi in zip(y, inst["h"]))
    return room_of(W, H)


# ============================================================ SA：序列对
def sp_pack(order_p, order_n, w, h):
    """序列对 (Γ+, Γ−) → 最小坐标。i 在 j 左：两序列中 i 都在 j 前；i 在 j 下：Γ+ 中 i 在后、Γ− 中 i 在前。"""
    n = len(w)
    pos_p = [0] * n
    for k, b in enumerate(order_p):
        pos_p[b] = k
    x = [0] * n
    y = [0] * n
    placed = []
    for b in order_n:                    # 按 Γ− 顺序，前面的块在 Γ− 中都在 b 之前
        xb = yb = 0
        pb = pos_p[b]
        for a in placed:
            if pos_p[a] < pb:            # a 在 b 左
                v = x[a] + w[a] + DELTA
                if v > xb:
                    xb = v
            else:                        # a 在 b 下
                v = y[a] + h[a] + DELTA
                if v > yb:
                    yb = v
        x[b], y[b] = xb, yb
        placed.append(b)
    W, H = room_of(max(x[i] + w[i] for i in range(n)), max(y[i] + h[i] for i in range(n)))
    return W * H, x, y


def run_sa(inst, limit, seed=0):
    rng = random.Random(seed)
    w, h, n = inst["w"], inst["h"], len(inst["w"])
    gp, gn = list(range(n)), list(range(n))
    rng.shuffle(gp); rng.shuffle(gn)
    cur, x, y = sp_pack(gp, gn, w, h)
    best = (cur, x, y)
    t0 = time.time()
    T0, T1 = cur * 0.05, cur * 0.0005
    evals = 0
    trace = [(0.0, cur)]
    while True:
        el = time.time() - t0
        if el > limit:
            break
        T = T0 * (T1 / T0) ** (el / limit)
        np_, nn_ = gp[:], gn[:]
        i, j = rng.sample(range(n), 2)
        mv = rng.random()
        if mv < 0.4:
            np_[i], np_[j] = np_[j], np_[i]
        elif mv < 0.8:
            nn_[i], nn_[j] = nn_[j], nn_[i]
        else:
            a, b = np_[i], np_[j]
            np_[i], np_[j] = b, a
            ia, ib = nn_.index(a), nn_.index(b)
            nn_[ia], nn_[ib] = b, a
        val, xx, yy = sp_pack(np_, nn_, w, h)
        evals += 1
        if val <= cur or rng.random() < math.exp((cur - val) / T):
            gp, gn, cur = np_, nn_, val
            if val < best[0]:
                best = (val, xx, yy)
                trace.append((round(el, 2), val))
    return {"x": best[1], "y": best[2], "lower_bound": None, "proven": False,
            "evals": evals, "trace": trace}


# ============================================================ ε 约束法外壳（MILP、Z3 共用）
def epsilon_constraint(inst, solve_min_H, limit):
    """从宽到窄枚举 Pareto 前沿：每次要求 W ≤ W̄ 最小化 H，再令 W̄ = W − 1。
    所有子问题都证明最优且扫描到底时，最优面积得到证明。"""
    w = inst["w"]
    Wbar = sum(w) + DELTA * (len(w) - 1)
    Wmin = max(w)
    t0 = time.time()
    best, all_proven, finished, trace = None, True, False, []
    while Wbar >= Wmin:
        remain = limit - (time.time() - t0)
        if remain <= 0.5:
            break
        r = solve_min_H(Wbar, remain)
        if r is None:                       # 超时且无解
            all_proven = False
            break
        x, y, proven, infeasible = r
        if infeasible:
            finished = True
            break
        if not proven:
            all_proven = False
        W, H = area_of(inst, x, y)
        if best is None or W * H < best[0]:
            best = (W * H, x, y)
            trace.append((round(time.time() - t0, 2), W * H))
        Wbar = W - 1
    else:
        finished = True
    if best is None:
        return {"x": None, "y": None, "lower_bound": None, "proven": False, "trace": trace}
    proven = finished and all_proven
    return {"x": best[1], "y": best[2], "lower_bound": best[0] if proven else None,
            "proven": proven, "trace": trace}


# ============================================================ MILP（HiGHS）
def run_milp(inst, limit):
    from scipy.optimize import Bounds, LinearConstraint, milp

    w, h, n = inst["w"], inst["h"], len(inst["w"])
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sym = identical_pairs(inst)
    Hmax = sum(h) + DELTA * n

    def solve(Wbar, remain):
        nv = 2 * n + 2 + 4 * len(pairs)
        HV, WV = 2 * n, 2 * n + 1
        M = Wbar + Hmax
        rows, lo, hi = [], [], []

        def add(coef, l, u):
            r = np.zeros(nv)
            for idx, val in coef:
                r[idx] += val
            rows.append(r); lo.append(l); hi.append(u)

        for p, (i, j) in enumerate(pairs):
            z = 2 * n + 2 + 4 * p
            add([(i, 1), (j, -1), (z, M)], -np.inf, M - w[i] - DELTA)
            add([(j, 1), (i, -1), (z + 1, M)], -np.inf, M - w[j] - DELTA)
            add([(n + i, 1), (n + j, -1), (z + 2, M)], -np.inf, M - h[i] - DELTA)
            add([(n + j, 1), (n + i, -1), (z + 3, M)], -np.inf, M - h[j] - DELTA)
            add([(z + k, 1) for k in range(4)], 1, np.inf)
        for i in range(n):
            add([(n + i, 1), (HV, -1)], -np.inf, -h[i])
            add([(i, 1), (WV, -1)], -np.inf, -w[i])
        add([(WV, 1), (HV, -KAPPA)], -np.inf, 0)
        add([(HV, 1), (WV, -KAPPA)], -np.inf, 0)
        for a, b in sym:
            add([(a, 1), (b, -1)], -np.inf, 0)
        c = np.zeros(nv); c[HV] = Wbar + 1; c[WV] = 1      # 先最小化 H，再最小化 W
        ub = np.full(nv, np.inf)
        ub[:n] = [Wbar - wi for wi in w]
        ub[n:2 * n] = [Hmax - hi_ for hi_ in h]
        ub[HV] = Hmax
        ub[WV] = Wbar
        ub[2 * n + 2:] = 1
        integ = np.ones(nv)
        res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi),
                   bounds=Bounds(np.zeros(nv), ub), integrality=integ,
                   options={"time_limit": max(remain, 1)})
        if res.status == 2:
            return [], [], True, True
        if res.x is None:
            return None
        x = [int(round(v)) for v in res.x[:n]]
        y = [int(round(v)) for v in res.x[n:2 * n]]
        return x, y, res.status == 0, False

    return epsilon_constraint(inst, solve, limit)


# ============================================================ Z3
def run_z3(inst, limit):
    import z3

    w, h, n = inst["w"], inst["h"], len(inst["w"])
    sym = identical_pairs(inst)

    def solve(Wbar, remain):
        opt = z3.Optimize()
        opt.set("timeout", int(remain * 1000))
        X = [z3.Int(f"x{i}") for i in range(n)]
        Y = [z3.Int(f"y{i}") for i in range(n)]
        H = z3.Int("H")
        W = z3.Int("W")
        opt.add(W <= Wbar, W <= KAPPA * H, H <= KAPPA * W)
        for i in range(n):
            opt.add(X[i] >= 0, X[i] + w[i] <= W, Y[i] >= 0, Y[i] + h[i] <= H)
        for i in range(n):
            for j in range(i + 1, n):
                opt.add(z3.Or(X[i] + w[i] + DELTA <= X[j], X[j] + w[j] + DELTA <= X[i],
                              Y[i] + h[i] + DELTA <= Y[j], Y[j] + h[j] + DELTA <= Y[i]))
        for a, b in sym:
            opt.add(X[a] <= X[b])
        opt.minimize(H)
        opt.minimize(W)                      # 字典序：先 H 后 W
        r = opt.check()
        if r == z3.unsat:
            return [], [], True, True
        try:
            m = opt.model()
            x = [m.eval(X[i], model_completion=True).as_long() for i in range(n)]
            y = [m.eval(Y[i], model_completion=True).as_long() for i in range(n)]
        except z3.Z3Exception:
            return None
        if not verify(inst, x, y):
            return None
        return x, y, r == z3.sat, False

    return epsilon_constraint(inst, solve, limit)


# ============================================================ CP-SAT
def run_cpsat(inst, limit):
    from ortools.sat.python import cp_model

    w, h, n = inst["w"], inst["h"], len(inst["w"])
    Wmax = sum(w) + DELTA * n
    Hmax = sum(h) + DELTA * n
    m = cp_model.CpModel()
    X = [m.NewIntVar(0, Wmax - w[i], f"x{i}") for i in range(n)]
    Y = [m.NewIntVar(0, Hmax - h[i], f"y{i}") for i in range(n)]
    # 每个块向右、向上膨胀 δ，膨胀后的区间两两不重叠 ⇔ 原块净距 ≥ δ
    XI = [m.NewIntervalVar(X[i], w[i] + DELTA, X[i] + w[i] + DELTA, f"xi{i}") for i in range(n)]
    YI = [m.NewIntervalVar(Y[i], h[i] + DELTA, Y[i] + h[i] + DELTA, f"yi{i}") for i in range(n)]
    m.AddNoOverlap2D(XI, YI)
    W = m.NewIntVar(max(w), Wmax, "W")
    H = m.NewIntVar(max(h), Hmax, "H")
    for i in range(n):
        m.Add(X[i] + w[i] <= W)
        m.Add(Y[i] + h[i] <= H)
    m.Add(W <= KAPPA * H)
    m.Add(H <= KAPPA * W)
    for a, b in identical_pairs(inst):
        m.Add(X[a] <= X[b])
    lb_area = sum(wi * hi for wi, hi in zip(w, h))
    A = m.NewIntVar(lb_area, Wmax * Hmax, "A")
    m.AddMultiplicationEquality(A, [W, H])
    m.Minimize(A)

    trace = []
    t0 = time.time()

    class CB(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            trace.append((round(time.time() - t0, 2), self.ObjectiveValue()))

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = limit
    s.parameters.num_workers = 8
    st = s.Solve(m, CB())
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"x": None, "y": None, "lower_bound": s.BestObjectiveBound(), "proven": False, "trace": trace}
    return {"x": [s.Value(v) for v in X], "y": [s.Value(v) for v in Y],
            "lower_bound": s.BestObjectiveBound(), "proven": st == cp_model.OPTIMAL, "trace": trace}


# ============================================================ 主程序
METHODS = [("SA", run_sa), ("MILP", run_milp), ("Z3", run_z3), ("CPSAT", run_cpsat)]


def main():
    results = []
    for inst in INSTANCES:
        area_lb_trivial = sum(wi * hi for wi, hi in zip(inst["w"], inst["h"]))
        for mname, fn in METHODS:
            t0 = time.time()
            r = fn(inst, TIME_LIMIT)
            el = time.time() - t0
            row = {"instance": inst["name"], "n": len(inst["w"]), "method": mname,
                   "time_s": round(el, 1), "proven": r["proven"], "trace": r.get("trace")}
            if r["x"] is not None and verify(inst, r["x"], r["y"]):
                W, H = area_of(inst, r["x"], r["y"])
                row.update(W_mm=W * 100, H_mm=H * 100, area_m2=round(W * H / 100, 1),
                           x=r["x"], y=r["y"])
            else:
                row.update(W_mm=None, H_mm=None, area_m2=None)
            lb = r.get("lower_bound")
            row["lower_bound_m2"] = round(lb / 100, 1) if lb else None
            row["trivial_lb_m2"] = round(area_lb_trivial / 100, 1)
            if "evals" in r:
                row["evals"] = r["evals"]
            results.append(row)
            print(f"{inst['name']:<4} {mname:<6} 面积={row['area_m2']} m²  下界={row['lower_bound_m2']}  "
                  f"已证最优={row['proven']}  用时={row['time_s']} s", flush=True)
            (HERE / "results.json").write_text(json.dumps(
                {"time_limit_s": TIME_LIMIT, "delta_mm": DELTA * 100, "kappa": KAPPA, "results": results},
                ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
