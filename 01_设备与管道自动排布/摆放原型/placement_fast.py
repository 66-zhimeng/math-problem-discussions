"""块级摆放原型（提速版）。问题定义、下界定义与 placement_cpsat.py 完全相同。

提速手段（均不改变可行域与最优值）：
  1. 可变尺寸区间：每块一个矩形，尺寸 = Σ 姿态布尔 × 该姿态尺寸；不重叠约束从“姿态数”降到“块数”
  2. 粗网格预求解：尺寸向上取整、检修区向外取整，粗解放大回细网格后必然可行，作为提示与上界
  3. 由上界推出的坐标范围：J ≤ J_ub ⇒ A ≤ J_ub·A₀/w_A ⇒ W, H ≤ √(κ·A_max)
  4. 冗余下界割：Σ(wᵢ+δ)(hᵢ+δ) ≤ (W+δ)(H+δ)

运行：.venv/Scripts/python placement_fast.py [总秒数，默认 60] [粗网格倍数，默认 5]
"""
import json
import math
import sys
import time

from ortools.sat.python import cp_model

import placement_cpsat as base

TOTAL = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
COARSE = int(sys.argv[2]) if len(sys.argv) > 2 else 5
WORKERS = 12


def coarsen(blocks, P, f):
    """细网格 → 粗网格（保守）：块尺寸、净距向上取整，检修区向外取整。仅用于生成提示解。"""
    cb = []
    for b in blocks:
        poses = []
        for p in b["poses"]:
            poses.append({**p,
                          "W": -(-p["W"] // f), "H": -(-p["H"] // f),
                          "ports": {k: (round(v[0] / f), round(v[1] / f), v[2], v[3], round(v[4] / f))
                                    for k, v in p["ports"].items()},
                          "zones": [(zx // f, zy // f, -(-(zx + zw) // f) - zx // f, -(-(zy + zh) // f) - zy // f)
                                    for zx, zy, zw, zh in p["zones"]]})
        fixed = None
        if b["fixed"]:
            assert b["fixed"][0] % f == 0 and b["fixed"][1] % f == 0, "固定块坐标须为粗网格整数倍"
            fixed = (b["fixed"][0] // f, b["fixed"][1] // f)
        cb.append({**b, "poses": poses, "fixed": fixed, "footprint": poses[0]["W"] * poses[0]["H"]})
    A0 = sum(b["footprint"] for b in cb)
    cP = {**P, "grid": P["grid"] * f, "delta": -(-P["delta"] // f), "lmin": -(-P["lmin"] // f),
          "A0": A0, "L0": P["B0"] * math.sqrt(A0)}
    return cb, cP


def solve(blocks, nets, P, weights, limit, hint=None, J_ub=None, trace=None, seed=0):
    m = cp_model.CpModel()
    n = len(blocks)
    d, lmin, k = P["delta"], P["lmin"], P["kappa"]
    wA, wL, wB = weights["area"], weights["length"], weights["bends"]

    # ---- 坐标范围：默认宽松；有上界时按 A_max 收紧
    loose = sum(max(max(p["W"], p["H"]) for p in b["poses"]) for b in blocks) + d * n
    Dmax = loose
    if J_ub is not None and wA > 0:
        A_max = J_ub * P["A0"] / wA
        Dmax = min(loose, int(math.isqrt(int(k * A_max))) + 1)

    X, Y, B, SX, SY = [], [], [], [], []
    for i, b in enumerate(blocks):
        if b["fixed"]:
            X.append(m.NewConstant(b["fixed"][0])); Y.append(m.NewConstant(b["fixed"][1]))
        else:
            X.append(m.NewIntVar(0, Dmax, "")); Y.append(m.NewIntVar(0, Dmax, ""))
        bs = [m.NewBoolVar("") for _ in b["poses"]]
        m.AddExactlyOne(bs)
        B.append(bs)
        ws = sorted({p["W"] for p in b["poses"]}); hs = sorted({p["H"] for p in b["poses"]})
        sx = m.NewIntVarFromDomain(cp_model.Domain.FromValues(ws), "")
        sy = m.NewIntVarFromDomain(cp_model.Domain.FromValues(hs), "")
        m.Add(sx == sum(bp * p["W"] for p, bp in zip(b["poses"], bs)))
        m.Add(sy == sum(bp * p["H"] for p, bp in zip(b["poses"], bs)))
        SX.append(sx); SY.append(sy)

    # 区间终点须为仿射表达式，故单独建终点变量 EX = X + SX
    EX = [m.NewIntVar(0, Dmax, "") for _ in range(n)]
    EY = [m.NewIntVar(0, Dmax, "") for _ in range(n)]
    raw_x = [m.NewIntervalVar(X[i], SX[i], EX[i], "") for i in range(n)]
    raw_y = [m.NewIntervalVar(Y[i], SY[i], EY[i], "") for i in range(n)]
    inf_x = [m.NewIntervalVar(X[i], SX[i] + d, EX[i] + d, "") for i in range(n)]
    inf_y = [m.NewIntervalVar(Y[i], SY[i] + d, EY[i] + d, "") for i in range(n)]
    m.AddNoOverlap2D(inf_x, inf_y)

    # 复制组
    groups = {}
    for i, b in enumerate(blocks):
        if b["copy_group"]:
            groups.setdefault(b["copy_group"], []).append(i)
    for members in groups.values():
        vnames = sorted({p["variant"] for p in blocks[members[0]]["poses"]})
        vb = {v: m.NewBoolVar("") for v in vnames}
        m.AddExactlyOne(vb.values())
        for i in members:
            for v in vnames:
                m.Add(sum(bp for p, bp in zip(blocks[i]["poses"], B[i]) if p["variant"] == v) == vb[v])

    W = m.NewIntVar(1, Dmax, "W"); H = m.NewIntVar(1, Dmax, "H")
    for i in range(n):
        m.Add(EX[i] <= W); m.Add(EY[i] <= H)

    # 检修区
    for i, b in enumerate(blocks):
        zx_, zy_ = [], []
        for p, bp in zip(b["poses"], B[i]):
            for zx, zy, zw, zh in p["zones"]:
                zx_.append(m.NewOptionalIntervalVar(X[i] + zx, zw, X[i] + zx + zw, bp, ""))
                zy_.append(m.NewOptionalIntervalVar(Y[i] + zy, zh, Y[i] + zy + zh, bp, ""))
                if P["zones_inside"]:
                    m.Add(X[i] + zx >= 0).OnlyEnforceIf(bp); m.Add(Y[i] + zy >= 0).OnlyEnforceIf(bp)
                    m.Add(X[i] + zx + zw <= W).OnlyEnforceIf(bp); m.Add(Y[i] + zy + zh <= H).OnlyEnforceIf(bp)
        if zx_:
            m.AddNoOverlap2D(zx_ + [raw_x[j] for j in range(n) if j != i],
                             zy_ + [raw_y[j] for j in range(n) if j != i])

    m.Add(W <= k * H); m.Add(H <= k * W)
    A = m.NewIntVar(1, Dmax * Dmax, "A")
    m.AddMultiplicationEquality(A, [W, H])
    # 冗余下界割
    S = sum(min((p["W"] + d) * (p["H"] + d) for p in b["poses"]) for b in blocks)
    A2 = m.NewIntVar(S, (Dmax + d) ** 2, "A2")
    m.AddMultiplicationEquality(A2, [W + d, H + d])

    # 管道下界（与 base 相同定义）
    L_terms, B_terms = [], []
    for e in nets:
        terms = e["terms"]
        stub = len(terms) >= 3
        zs = [blocks[i]["poses"][0]["ports"][pn][4] for i, pn in terms]
        const = (len(terms) * lmin if stub else 0) + (max(zs) - min(zs))
        rng = []
        for axis in (0, 1):
            vs = []
            for i, pn in terms:
                off = sum(bp * (p["ports"][pn][axis] + (lmin * p["ports"][pn][2 + axis] if stub else 0))
                          for p, bp in zip(blocks[i]["poses"], B[i]))
                v = m.NewIntVar(-lmin, Dmax + lmin, "")
                m.Add(v == (X[i] if axis == 0 else Y[i]) + off)
                vs.append(v)
            hi = m.NewIntVar(-lmin, Dmax + lmin, ""); lo = m.NewIntVar(-lmin, Dmax + lmin, "")
            m.AddMaxEquality(hi, vs); m.AddMinEquality(lo, vs)
            rng.append(hi - lo)
        L_terms.append(rng[0] + rng[1] + const)
        if len(terms) == 2:
            (i, pn), (j, qn) = terms
            be = m.NewIntVar(0, 2, "")
            for p, bp in zip(blocks[i]["poses"], B[i]):
                for q, bq in zip(blocks[j]["poses"], B[j]):
                    c = base.bend_lb(p["ports"][pn], q["ports"][qn])
                    if c:
                        m.Add(be >= c * (bp + bq - 1))
            B_terms.append(be)
    L = sum(L_terms); Bn = sum(B_terms)
    obj = wA / P["A0"] * A + wL / P["L0"] * L + wB / P["B0"] * Bn
    m.Minimize(obj)

    if hint:
        for i, h_ in enumerate(hint):
            if not blocks[i]["fixed"]:
                m.AddHint(X[i], h_["x"]); m.AddHint(Y[i], h_["y"])
            for kk, bp in enumerate(B[i]):
                m.AddHint(bp, kk == h_["pose"])

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = max(limit, 0.5)
    s.parameters.num_workers = WORKERS
    s.parameters.random_seed = seed
    t0 = time.time()

    class CB(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            if trace is not None:
                trace.append((time.time() - t0, self.ObjectiveValue(), self.BestObjectiveBound()))

    st = s.Solve(m, CB())
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    sol = [{"x": s.Value(X[i]), "y": s.Value(Y[i]),
            "pose": next(kk for kk, bp in enumerate(B[i]) if s.Value(bp))} for i in range(n)]
    return {"status": s.StatusName(st), "objective": s.ObjectiveValue(), "bound": s.BestObjectiveBound(),
            "solution": sol, "model_L": s.Value(L), "model_B": s.Value(Bn)}


def run(blocks, nets, P, weights, total, f, trace, seed=0):
    """粗网格预求解（总时间的 20%）→ 细网格求解（提示 + 上界）。trace 记录细网格求解的时间—目标曲线。"""
    t0 = time.time()
    hint = J_ub = None
    if f > 1:
        cb, cP = coarsen(blocks, P, f)
        rc = solve(cb, nets, cP, weights, total * 0.2, seed=seed)
        if rc:
            hint = [{"x": h["x"] * f, "y": h["y"] * f, "pose": h["pose"]} for h in rc["solution"]]
            v = base.validate(blocks, nets, P, hint, weights)       # 放大后在细网格上必然可行
            J_ub = v["J_full_weights"] + 1e-9
            trace.append((time.time() - t0, J_ub, None))
    offset = time.time() - t0
    fine_trace = []
    r = solve(blocks, nets, P, weights, total - offset, hint, J_ub, fine_trace, seed=seed)
    trace.extend((t + offset, o, b) for t, o, b in fine_trace)
    return r


def main():
    blocks, nets, P = base.load(base.HERE / "算例.json")
    trace = []
    t0 = time.time()
    r = run(blocks, nets, P, P["w"], TOTAL, COARSE, trace)
    el = time.time() - t0
    v = base.validate(blocks, nets, P, r["solution"], P["w"])
    assert v["_L"] == r["model_L"] and v["_B"] == r["model_B"]
    gap = (r["objective"] - r["bound"]) / r["objective"] * 100
    print(f"状态={r['status']}  用时={el:.1f} s  J={r['objective']:.4f}  下界={r['bound']:.4f}  差距={gap:.2f}%")
    print(f"占地 {v['area_m2']} m²  管长下界 {v['L_lb_m']} m  弯头下界 {v['bends_lb']}")
    print("时间—目标：", [(round(t, 1), round(o, 4)) for t, o, _ in trace])


if __name__ == "__main__":
    main()
