"""块级摆放原型：CP-SAT + 管道下界。

对应主文档第 12 节与《形式化求解-其他路线比较.md》路线 A/C。只做摆放，不布管；
管长、弯头为**可证明的下界**，真实值需布管后才知道。

模型内容
  - 每个块：若干内部排法（variant）× 允许的 90° 旋转，二者合称“姿态”，恰选一个
  - 同一复制组（copy_group）的块必须选同一种内部排法
  - 块间净距 δ_ee；检修区不得与其他块重叠（检修区之间可重叠）
  - 检修区是否必须在占地矩形内：必填参数 service_zones_inside_footprint，无默认值
      false：检修区可伸出占地矩形（例如占地矩形外是通道）；检修区不计入占地面积
      true ：检修区必须完全落在占地矩形 [0,W]×[0,H] 内（例如矩形边界就是墙）
  - 固定块位置锁定
  - 占地矩形 W×H 包住所有块，长宽比 ≤ κ，面积 A = W·H
  - 管网长度下界（每个管网）：
      2 端点：两端口的曼哈顿距离
      ≥3 端点：必含三通，每个端点先沿端口方向直行 ≥ ℓ_min；
               长度 ≥ k·ℓ_min + 各轴上“伸出点”坐标跨度之和（直角 Steiner 树下界）
  - 弯头下界（2 端点管网）：出发方向 d₀ = u_p，到达方向 d₁ = −u_q
      两端高度不同 → 2；否则 d₀ = d₁ → 0，垂直 → 1，相反 → 2
  - 目标：w_A·A/A₀ + w_L·L/L₀ + w_B·B/B₀（主文档 12.8）

运行：.venv/Scripts/python placement_cpsat.py [每个模型的秒数，默认 60]
"""
import json
import math
import sys
import time
from pathlib import Path

from ortools.sat.python import cp_model

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
TIME_LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0


# ============================================================ 读取算例、展开姿态
def g(v, grid):
    assert v % grid == 0, f"{v} 不是 {grid} 的整数倍"
    return v // grid


def rot_point(x, y, w, h, r):
    return {0: (x, y), 90: (h - y, x), 180: (w - x, h - y), 270: (y, w - x)}[r]


def rot_dir(ux, uy, r):
    return {0: (ux, uy), 90: (-uy, ux), 180: (-ux, -uy), 270: (uy, -ux)}[r]


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    P = data["params"]
    grid = P["grid_mm"]
    blocks = []
    for b in data["blocks"]:
        variants = data["templates"][b["template"]]["variants"] if "template" in b else b["variants"]
        poses = []
        for vname, v in variants.items():
            w, h = g(v["size"][0], grid), g(v["size"][1], grid)
            for r in b["rotations"]:
                W, H = (w, h) if r in (0, 180) else (h, w)
                ports = {}
                for pn, p in v["ports"].items():
                    px, py = rot_point(g(p["pos"][0], grid), g(p["pos"][1], grid), w, h, r)
                    ux, uy = rot_dir(p["dir"][0], p["dir"][1], r)
                    ports[pn] = (px, py, ux, uy, g(p["z"], grid))
                zones = []
                for zx, zy, zw, zh in v["service_zones"]:
                    ax, ay = rot_point(g(zx, grid), g(zy, grid), w, h, r)
                    bx, by = rot_point(g(zx + zw, grid), g(zy + zh, grid), w, h, r)
                    zones.append((min(ax, bx), min(ay, by), abs(bx - ax), abs(by - ay)))
                poses.append({"variant": vname, "rot": r, "W": W, "H": H, "ports": ports, "zones": zones})
        fixed = tuple(g(c, grid) for c in b["fixed"]) if "fixed" in b else None
        blocks.append({"id": b["id"], "name": b["name"], "poses": poses, "fixed": fixed,
                       "copy_group": b.get("copy_group"),
                       "footprint": g(poses[0]["W"] * grid, grid) * poses[0]["H"]})
    idx = {b["id"]: i for i, b in enumerate(blocks)}
    nets = []
    for n in data["nets"]:
        terms = []
        for t in n["terminals"]:
            bid, pn = t.split(".")
            terms.append((idx[bid], pn))
        nets.append({"id": n["id"], "terms": terms})
    if "service_zones_inside_footprint" not in P:
        raise KeyError("缺少必填参数 params.service_zones_inside_footprint（检修区是否必须在占地矩形内），"
                       "请向用户确认：矩形边界外是否可用作检修空间")
    params = {"grid": grid, "delta": g(P["delta_ee_mm"], grid), "lmin": g(P["l_min_mm"], grid),
              "kappa": P["kappa"], "w": P["weights"],
              "zones_inside": bool(P["service_zones_inside_footprint"])}
    A0 = sum(b["footprint"] for b in blocks)
    params.update(A0=A0, L0=len(nets) * math.sqrt(A0), B0=len(nets))
    return blocks, nets, params


# ============================================================ 下界定义（模型与校验器共用的纯函数）
def bend_lb(p, q):
    """p、q 为 (x, y, ux, uy, z)。"""
    if p[4] != q[4]:
        return 2
    d0 = (p[2], p[3])
    d1 = (-q[2], -q[3])
    if d0 == d1:
        return 0
    if d0 == (-d1[0], -d1[1]):
        return 2
    return 1


# ============================================================ CP-SAT 模型
def build_and_solve(blocks, nets, P, weights, limit, label, trace=None, workers=8, seed=0):
    m = cp_model.CpModel()
    n = len(blocks)
    d, lmin, k = P["delta"], P["lmin"], P["kappa"]
    Xmax = sum(max(max(p["W"], p["H"]) for p in b["poses"]) for b in blocks) + d * n
    Ymax = Xmax

    X, Y, B = [], [], []
    raw, infl, zones = [], [], []
    for i, b in enumerate(blocks):
        if b["fixed"]:
            X.append(m.NewConstant(b["fixed"][0])); Y.append(m.NewConstant(b["fixed"][1]))
        else:
            X.append(m.NewIntVar(0, Xmax, f"x_{b['id']}")); Y.append(m.NewIntVar(0, Ymax, f"y_{b['id']}"))
        bs = [m.NewBoolVar(f"pose_{b['id']}_{k_}") for k_ in range(len(b["poses"]))]
        m.AddExactlyOne(bs)
        B.append(bs)
        raw_i, infl_i, zone_i = [], [], []
        for p, bp in zip(b["poses"], bs):
            raw_i.append((m.NewOptionalIntervalVar(X[i], p["W"], X[i] + p["W"], bp, ""),
                          m.NewOptionalIntervalVar(Y[i], p["H"], Y[i] + p["H"], bp, "")))
            infl_i.append((m.NewOptionalIntervalVar(X[i], p["W"] + d, X[i] + p["W"] + d, bp, ""),
                           m.NewOptionalIntervalVar(Y[i], p["H"] + d, Y[i] + p["H"] + d, bp, "")))
            for zx, zy, zw, zh in p["zones"]:
                zone_i.append((m.NewOptionalIntervalVar(X[i] + zx, zw, X[i] + zx + zw, bp, ""),
                               m.NewOptionalIntervalVar(Y[i] + zy, zh, Y[i] + zy + zh, bp, "")))
        raw.append(raw_i); infl.append(infl_i); zones.append(zone_i)

    # 复制组：同一内部排法
    groups = {}
    for i, b in enumerate(blocks):
        if b["copy_group"]:
            groups.setdefault(b["copy_group"], []).append(i)
    for members in groups.values():
        vnames = sorted({p["variant"] for p in blocks[members[0]]["poses"]})
        vb = {v: m.NewBoolVar(f"variant_{v}") for v in vnames}
        m.AddExactlyOne(vb.values())
        for i in members:
            for v in vnames:
                m.Add(sum(bp for p, bp in zip(blocks[i]["poses"], B[i]) if p["variant"] == v) == vb[v])

    # 不重叠：膨胀 δ 后两两不重叠
    m.AddNoOverlap2D([r[0] for ri in infl for r in ri], [r[1] for ri in infl for r in ri])
    # 检修区不与其他块重叠
    for i in range(n):
        if zones[i]:
            xs = [z[0] for z in zones[i]] + [r[0] for j in range(n) if j != i for r in raw[j]]
            ys = [z[1] for z in zones[i]] + [r[1] for j in range(n) if j != i for r in raw[j]]
            m.AddNoOverlap2D(xs, ys)

    # 占地矩形、长宽比、面积
    W = m.NewIntVar(1, Xmax, "W"); H = m.NewIntVar(1, Ymax, "H")
    for i, b in enumerate(blocks):
        for p, bp in zip(b["poses"], B[i]):
            m.Add(X[i] + p["W"] <= W).OnlyEnforceIf(bp)
            m.Add(Y[i] + p["H"] <= H).OnlyEnforceIf(bp)
    if P["zones_inside"]:
        for i, b in enumerate(blocks):
            for p, bp in zip(b["poses"], B[i]):
                for zx, zy, zw, zh in p["zones"]:
                    m.Add(X[i] + zx >= 0).OnlyEnforceIf(bp)
                    m.Add(Y[i] + zy >= 0).OnlyEnforceIf(bp)
                    m.Add(X[i] + zx + zw <= W).OnlyEnforceIf(bp)
                    m.Add(Y[i] + zy + zh <= H).OnlyEnforceIf(bp)
    m.Add(W <= k * H); m.Add(H <= k * W)
    A = m.NewIntVar(1, Xmax * Ymax, "A")
    m.AddMultiplicationEquality(A, [W, H])

    # 管道下界
    def port_expr(i, pn, stub):
        ex = X[i] + sum(bp * (p["ports"][pn][0] + (lmin * p["ports"][pn][2] if stub else 0))
                        for p, bp in zip(blocks[i]["poses"], B[i]))
        ey = Y[i] + sum(bp * (p["ports"][pn][1] + (lmin * p["ports"][pn][3] if stub else 0))
                        for p, bp in zip(blocks[i]["poses"], B[i]))
        return ex, ey

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
                v = m.NewIntVar(-Xmax, 2 * Xmax, "")
                m.Add(v == port_expr(i, pn, stub)[axis])
                vs.append(v)
            hi = m.NewIntVar(-Xmax, 2 * Xmax, ""); lo = m.NewIntVar(-Xmax, 2 * Xmax, "")
            m.AddMaxEquality(hi, vs); m.AddMinEquality(lo, vs)
            rng.append(hi - lo)
        L_terms.append(rng[0] + rng[1] + const)
        if len(terms) == 2:
            (i, pn), (j, qn) = terms
            be = m.NewIntVar(0, 2, f"bend_{e['id']}")
            for p, bp in zip(blocks[i]["poses"], B[i]):
                for q, bq in zip(blocks[j]["poses"], B[j]):
                    c = bend_lb(p["ports"][pn], q["ports"][qn])
                    if c:
                        m.Add(be >= c * (bp + bq - 1))
            B_terms.append(be)
    L = sum(L_terms)
    Bn = sum(B_terms)

    wA, wL, wB = weights["area"], weights["length"], weights["bends"]
    m.Minimize(wA / P["A0"] * A + wL / P["L0"] * L + wB / P["B0"] * Bn)

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = limit
    s.parameters.num_workers = workers
    s.parameters.random_seed = seed
    t0 = time.time()

    class CB(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            if trace is not None:
                trace.append((time.time() - t0, self.ObjectiveValue()))

    st = s.Solve(m, CB())
    el = time.time() - t0
    assert st in (cp_model.OPTIMAL, cp_model.FEASIBLE), s.StatusName(st)
    sol = []
    for i, b in enumerate(blocks):
        pi = next(k_ for k_, bp in enumerate(B[i]) if s.Value(bp))
        sol.append({"x": s.Value(X[i]), "y": s.Value(Y[i]), "pose": pi})
    return {"label": label, "status": s.StatusName(st), "time_s": round(el, 1),
            "objective": s.ObjectiveValue(), "bound": s.BestObjectiveBound(),
            "model_W": s.Value(W), "model_H": s.Value(H), "model_L": s.Value(L), "model_B": s.Value(Bn),
            "solution": sol}


# ============================================================ 独立校验器
def validate(blocks, nets, P, sol, weights):
    d, lmin, k = P["delta"], P["lmin"], P["kappa"]
    rects, zs = [], []
    for b, s_ in zip(blocks, sol):
        p = b["poses"][s_["pose"]]
        if b["fixed"]:
            assert (s_["x"], s_["y"]) == b["fixed"], b["id"]
        rects.append((s_["x"], s_["y"], p["W"], p["H"]))
        zs.append([(s_["x"] + zx, s_["y"] + zy, zw, zh) for zx, zy, zw, zh in p["zones"]])
    n = len(blocks)
    for i in range(n):
        for j in range(i + 1, n):
            a, c = rects[i], rects[j]
            gx = max(c[0] - (a[0] + a[2]), a[0] - (c[0] + c[2]))
            gy = max(c[1] - (a[1] + a[3]), a[1] - (c[1] + c[3]))
            assert max(gx, gy) >= d, f"净距不足：{blocks[i]['id']} {blocks[j]['id']}"
    for i in range(n):
        for z in zs[i]:
            for j in range(n):
                if j == i:
                    continue
                c = rects[j]
                ox = min(z[0] + z[2], c[0] + c[2]) - max(z[0], c[0])
                oy = min(z[1] + z[3], c[1] + c[3]) - max(z[1], c[1])
                assert not (ox > 0 and oy > 0), f"检修区冲突：{blocks[i]['id']} 与 {blocks[j]['id']}"
    for members in {b["copy_group"] for b in blocks if b["copy_group"]}:
        vs = {blocks[i]["poses"][sol[i]["pose"]]["variant"] for i in range(n) if blocks[i]["copy_group"] == members}
        assert len(vs) == 1, "复制组内部排法不一致"
    extent = rects + ([z for zl in zs for z in zl] if P["zones_inside"] else [])
    Wx = max(r[0] + r[2] for r in extent)
    Hy = max(r[1] + r[3] for r in extent)
    W, H = max(Wx, -(-Hy // k)), max(Hy, -(-Wx // k))
    A = W * H
    zones_outside = sum(1 for zl in zs for z in zl
                        if z[0] < 0 or z[1] < 0 or z[0] + z[2] > W or z[1] + z[3] > H)
    if P["zones_inside"]:
        assert zones_outside == 0, "检修区超出占地矩形"

    def port(i, pn):
        p = blocks[i]["poses"][sol[i]["pose"]]["ports"][pn]
        return (sol[i]["x"] + p[0], sol[i]["y"] + p[1], p[2], p[3], p[4])

    L = Bsum = 0
    per_net = []
    for e in nets:
        ps = [port(i, pn) for i, pn in e["terms"]]
        if len(ps) >= 3:
            pts = [(q[0] + lmin * q[2], q[1] + lmin * q[3], q[4]) for q in ps]
            l = len(ps) * lmin
        else:
            pts = [(q[0], q[1], q[4]) for q in ps]
            l = 0
        l += sum(max(t[a] for t in pts) - min(t[a] for t in pts) for a in range(3))
        bnd = bend_lb(ps[0], ps[1]) if len(ps) == 2 else 0
        L += l; Bsum += bnd
        per_net.append({"net": e["id"], "L_lb_m": l * P["grid"] / 1000, "bends_lb": bnd})
    J = weights["area"] * A / P["A0"] + weights["length"] * L / P["L0"] + weights["bends"] * Bsum / P["B0"]
    return {"W_m": W * P["grid"] / 1000, "H_m": H * P["grid"] / 1000, "area_m2": round(A * P["grid"] ** 2 / 1e6, 1),
            "L_lb_m": round(L * P["grid"] / 1000, 1), "bends_lb": Bsum,
            "zones_outside_footprint": zones_outside, "J_full_weights": round(J, 4),
            "per_net": per_net, "_A": A, "_L": L, "_B": Bsum}


# ============================================================ 画图
def plot(blocks, nets, P, results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    fig, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 8))
    colors = plt.cm.tab20.colors
    for ax, r in zip(axes, results):
        sol = r["solution"]
        for b, s_ in zip(blocks, sol):
            p = b["poses"][s_["pose"]]
            ax.add_patch(Rectangle((s_["x"], s_["y"]), p["W"], p["H"], fill=False, lw=1.5))
            for zx, zy, zw, zh in p["zones"]:
                ax.add_patch(Rectangle((s_["x"] + zx, s_["y"] + zy), zw, zh, fill=True, alpha=0.12,
                                       color="orange", lw=0))
            ax.text(s_["x"] + p["W"] / 2, s_["y"] + p["H"] / 2, f"{b['name']}\n{p['variant']} {p['rot']}°"
                    if b["copy_group"] else b["name"], ha="center", va="center", fontsize=7)
        for c, e in enumerate(nets):
            pts = []
            for i, pn in e["terms"]:
                p = blocks[i]["poses"][sol[i]["pose"]]["ports"][pn]
                px, py = sol[i]["x"] + p[0], sol[i]["y"] + p[1]
                ax.annotate("", xy=(px + 4 * p[2], py + 4 * p[3]), xytext=(px, py),
                            arrowprops=dict(arrowstyle="->", color=colors[c % 20], lw=1))
                pts.append((px, py))
            cx = sum(q[0] for q in pts) / len(pts); cy = sum(q[1] for q in pts) / len(pts)
            for q in pts:
                ax.plot([q[0], cx], [q[1], cy], color=colors[c % 20], lw=0.8, alpha=0.6)
        v = r["validated"]
        ax.set_title(f"{r['label']}\n占地 {v['area_m2']} m²（{v['W_m']}×{v['H_m']} m）  "
                     f"管长下界 {v['L_lb_m']} m  弯头下界 {v['bends_lb']}", fontsize=10)
        ax.set_aspect("equal"); ax.autoscale()
        ax.set_xlabel("×100 mm")
    fig.suptitle("连线为管网示意（端点到管网中心），不是真实管路；橙色为检修区", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


# ============================================================ 主程序
def main():
    blocks, nets, P = load(HERE / "算例.json")
    print(f"块 {len(blocks)} 个，姿态共 {sum(len(b['poses']) for b in blocks)} 个，管网 {len(nets)} 个；"
          f"A₀={P['A0']}，L₀={P['L0']:.1f}，B₀={P['B0']}（网格单位 100 mm）")
    full_w = P["w"]
    runs = [("仅最小占地", {"area": 1.0, "length": 0.0, "bends": 0.0}),
            ("占地 + 管长 + 弯头下界", full_w)]
    results = []
    for label, wts in runs:
        r = build_and_solve(blocks, nets, P, wts, TIME_LIMIT, label)
        v = validate(blocks, nets, P, r["solution"], full_w)
        # 校验器与模型内部计算一致
        # 管长下界由 max/min 等式精确定义，必须一致；弯头变量只有下界约束，权重为 0 时求解器不会压低它
        assert v["_L"] == r["model_L"], (v["_L"], r["model_L"])
        assert (v["_B"] == r["model_B"]) if wts["bends"] > 0 else (r["model_B"] >= v["_B"]), (v["_B"], r["model_B"])
        r["validated"] = v
        results.append(r)
        gap = (r["objective"] - r["bound"]) / r["objective"] * 100 if r["objective"] else 0
        print(f"\n[{label}] 状态={r['status']}  用时={r['time_s']} s  本模型目标={r['objective']:.4f}  "
              f"下界={r['bound']:.4f}  差距={gap:.2f}%")
        print(f"  占地 {v['area_m2']} m²（{v['W_m']}×{v['H_m']} m）  管长下界 {v['L_lb_m']} m  "
              f"弯头下界 {v['bends_lb']}  按完整权重的 J={v['J_full_weights']}")
        for b, s_ in zip(blocks, r["solution"]):
            p = b["poses"][s_["pose"]]
            print(f"    {b['name']:<6} ({s_['x'] * P['grid']:>6}, {s_['y'] * P['grid']:>6}) mm  "
                  f"{p['variant']:<6} {p['rot']:>3}°")
    for r in results:
        for key in ("_A", "_L", "_B"):
            r["validated"].pop(key)
    (HERE / "results.json").write_text(json.dumps({"time_limit_s": TIME_LIMIT, "results": results},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    plot(blocks, nets, P, results, HERE / "摆放结果.png")
    print("\n已保存 results.json、摆放结果.png")


if __name__ == "__main__":
    main()
