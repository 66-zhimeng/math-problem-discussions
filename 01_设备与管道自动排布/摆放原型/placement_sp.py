"""块级摆放：序列对搜索 + LP 定位。问题定义、评分与 placement_cpsat.py 相同（按 v2 下界评分）。

解的表示：(Γ⁺, Γ⁻, 每块旋转, 每个复制组的内部排法, 两端点管网的“直连”标记)
  - i 在 j 左 ⇔ 两个序列中 i 都在 j 前；i 在 j 下 ⇔ Γ⁺ 中 i 在 j 后、Γ⁻ 中 i 在 j 前

评估一个候选（内层）：
  1. 最长路：由序列对得到差分约束，算出该骨架下最小 W*、H*（考虑 κ 补足）。
     固定块位置与最长路冲突 → 候选不可行，直接拒绝。
  2. 快速下界：J ≥ 面积项(W*, H*) + 管长常数项 + 已确定的弯头项。退火中先抽接受阈值，
     下界超过阈值就不解 LP（结果与解完 LP 再拒绝完全相同，只是更快）。
  3. LP（HiGHS）：变量为块坐标、W、H、管网辅助变量；目标为
       面积在 (W*, H*) 处的切平面 + 管长下界（与校验器 v2 定义相同的线性化）。
     两端点管网：姿态对能直连（同高、正对）且直连标记为真 → 加“轴线对齐、间距 ≥ ℓ_min”，
       长度 = 曼哈顿距离；否则长度 ≥ max(曼哈顿距离, 2ℓ_min + 伸出点曼哈顿距离)。
     多端点管网：默认为伸出点坐标跨度（max − min）；trunk=True 时改用“主干 + 分支”估计
    （主干沿一条直线，长度 = 主干轴跨度 + Σ 各端点到主干的垂距；恒 ≥ 跨度估计，仍是有效下界）。
  4. 以 placement_cpsat.validate(..., lb="v2") 计算最终 J；所有比较以校验器为准。

检修区处理（交接文档 4.3 第 2 条方式 (a)）：
  对每对 (i, j)，按序列对给出的方向加间距 max(块+净距, i 检修区伸出量, j 检修区伸出量)。
  这是充分条件；检修区在块宽度范围之外伸出、且在另一轴上错开的布局会被排除（启发式限制）。

外层：模拟退火，多进程并行；每个冷却周期结束时从全局最好解重新加热。

布管引导（可选，run(..., guide=...)）：每隔 every_s 秒对当前解调用 guide["fn"]（粗网格快速布管，约 1 s）：
  - 路由目标 J_route = 校验器 J − 管长下界项 + 粗网格管长项；粗网格不可布 → J_route = inf。
  - 各管网长度权重 w_e ← (1 − ema)·w_e + ema·clamp(粗网格长度 / 下界长度, 1, w_max)，
    用于 LP 与快速下界（下界严重低估的管网，例如多端点总管，被拉得更近）。
  - 进程间共享、重新加热、最终返回的“最好解”都改按 J_route 比较。

运行：.venv/Scripts/python placement_sp.py [总秒数，默认 60] [进程数，默认 12] [种子，默认 0]
"""
import math
import multiprocessing as mp
import random
import sys
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

import placement_cpsat as base

# ============================================================ 预处理
class Problem:
    def __init__(self, blocks, nets, P, weights, access=None, margins=None, trunk=False):
        self.blocks, self.nets, self.P, self.w = blocks, nets, P, weights
        self.trunk = trunk
        self.access = access_rects_factory(P, access) if access else None
        self.margins = side_margins_factory(P, margins) if margins else None
        self.traffic = None
        if margins and margins.get("traffic"):
            need = ["heights_mm", "z_max_mm", "K"]
            miss = [k for k in need if k not in margins]
            if miss:
                raise KeyError(f"缝隙穿行容量缺少参数：{miss}")
            g = P["grid"]
            D, dep, dpp = margins["D_mm"], margins["delta_ep_mm"], margins["delta_pp_mm"]

            def levels(h):
                """设备上方可放的管道层数：管中心可在 [h + δ_ep + D/2, z_max − D/2]，层距 D + δ_pp。
                越过设备要升降各一次，K < 2 时不能越过。"""
                lo_c, hi_c = h + dep + D / 2, margins["z_max_mm"] - D / 2
                if margins["K"] < 2 or hi_c < lo_c:
                    return 0
                return int((hi_c - lo_c) // (D + dpp)) + 1
            self.traffic = {
                "levels": [levels(margins["heights_mm"][b["id"]]) for b in blocks],
                "lane": -(-int(D + dpp) // g),
                "stub": -(-int(P["lmin"] * g + margins["c_rho"] * D) // g),
            }
        self.n = len(blocks)
        self.d, self.lmin, self.k = P["delta"], P["lmin"], P["kappa"]
        # 姿态索引：(内部排法, 旋转) → pose 下标
        self.variants, self.rots, self.pose_of = [], [], []
        for b in blocks:
            vs = sorted({p["variant"] for p in b["poses"]})
            rs = sorted({p["rot"] for p in b["poses"]})
            self.variants.append(vs); self.rots.append(rs)
            self.pose_of.append({(p["variant"], p["rot"]): k for k, p in enumerate(b["poses"])})
        groups = {}
        for i, b in enumerate(blocks):
            key = b["copy_group"] or f"__solo_{i}"
            groups.setdefault(key, []).append(i)
        self.groups = list(groups.values())            # 每组共用内部排法；非复制块自成一组
        self.group_of = [0] * self.n
        for gi, mem in enumerate(self.groups):
            for i in mem:
                self.group_of[i] = gi
        self.fixed = [b["fixed"] for b in blocks]
        self.net_w = [1.0] * len(nets)                 # 各管网长度权重（布管引导时更新；否则恒为 1）
        # 可直连标记只对两端点管网有意义
        self.two = [e for e, net in enumerate(nets) if len(net["terms"]) == 2]
        zs = []
        for net in nets:
            z = [blocks[i]["poses"][0]["ports"][pn][4] for i, pn in net["terms"]]
            zs.append(max(z) - min(z))
        self.dz = zs

    def pose(self, st, i):
        v = self.variants[i][st["var"][self.group_of[i]]]
        r = self.rots[i][st["rot"][i]]
        return self.blocks[i]["poses"][self.pose_of[i][(v, r)]], self.pose_of[i][(v, r)]


def access_rects_factory(P, access):
    """端口接入区（主文档 12.3 的必要条件）：管道离开端口须先直行 ≥ ℓ_min。
    返回函数 pose → (raw_rects, pipe_rects)，局部坐标 (x0, y0, x1, y1)，网格单位：
      raw_rects  管段盒按 δ_ep 膨胀，不得与其他块重叠
      pipe_rects 管段盒本身（宽 D），不得与其他块的检修区重叠"""
    grid = P["grid"]
    if access["D_mm"] % (2 * grid) or access["delta_ep_mm"] % grid:
        raise ValueError("端口接入区：D/2 与 δ_ep 须为网格整数倍")
    hw, dep, L = access["D_mm"] // 2 // grid, access["delta_ep_mm"] // grid, P["lmin"]
    cache = {}

    def rects(p):
        key = id(p)
        if key not in cache:
            raw, pipe = [], []
            for x, y, ux, uy, _z in p["ports"].values():
                # 管段方形截面在末端也外伸 D/2（与布管校验同口径）
                for depth, side, out in ((L + hw + dep, hw + dep, raw), (L + hw, hw, pipe)):
                    ax, ay = x + ux * depth, y + uy * depth
                    if ux:
                        out.append((min(x, ax), y - side, max(x, ax), y + side))
                    else:
                        out.append((x - side, min(y, ay), x + side, max(y, ay)))
            cache[key] = (raw, pipe)
        return cache[key]
    return rects


def side_margins_factory(P, cfg):
    """按侧出管留空（网格单位）：某侧有 k 个端口时，
       m = ℓ_min + ρ + (k−1)·(D + δ_pp) + D/2 + δ_ep，向上取整到网格；无端口为 0。
    含义：管道离开端口直行到转弯点，再沿该侧并排走 k 条车道，最外侧与相邻设备保持 δ_ep。
    相邻两块之间的缝隙 ≥ 两侧留空之和，因此正面相对的端口需求会叠加。"""
    grid = P["grid"]
    need = ["D_mm", "delta_pp_mm", "delta_ep_mm", "c_rho"]
    miss = [k for k in need if k not in cfg]
    if miss:
        raise KeyError(f"出管留空缺少参数：{miss}")
    lmin_mm = P["lmin"] * grid
    rho = cfg["c_rho"] * cfg["D_mm"]
    cache = {}

    def margins(p):
        key = id(p)
        if key not in cache:
            cnt = {"L": 0, "R": 0, "B": 0, "T": 0}
            for _x, _y, ux, uy, _z in p["ports"].values():
                cnt["L" if ux < 0 else "R" if ux > 0 else "B" if uy < 0 else "T"] += 1
            m = {}
            for side, k in cnt.items():
                mm = 0 if k == 0 else (lmin_mm + rho + (k - 1) * (cfg["D_mm"] + cfg["delta_pp_mm"])
                                       + cfg["D_mm"] / 2 + cfg["delta_ep_mm"])
                m[side] = -(-int(math.ceil(mm)) // grid)
            cache[key] = (m["L"], m["R"], m["B"], m["T"])
        return cache[key]
    return margins


def extents(p, zones_inside):
    """块在局部坐标中需要的范围：(左, 右, 下, 上)，以及检修区伸出量。"""
    zl = min([0] + [zx for zx, _, _, _ in p["zones"]])
    zr = max([p["W"]] + [zx + zw for zx, _, zw, _ in p["zones"]])
    zb = min([0] + [zy for _, zy, _, _ in p["zones"]])
    zt = max([p["H"]] + [zy + zh for _, zy, _, zh in p["zones"]])
    return zl, zr, zb, zt


# ============================================================ 评估
def _bounds(zones):
    """(x, y, w, h) 列表的包围范围 (左, 右, 下, 上)；空列表返回 None。"""
    if not zones:
        return None
    return (min(z[0] for z in zones), max(z[0] + z[2] for z in zones),
            min(z[1] for z in zones), max(z[1] + z[3] for z in zones))


def relations(gp, gm, n):
    pp = [0] * n; pm = [0] * n
    for k, b in enumerate(gp):
        pp[b] = k
    for k, b in enumerate(gm):
        pm[b] = k
    return pp, pm


def skeleton(pr, st):
    """最长路：返回 (poses, 约束列表, W*, H*) 或 None（与固定块冲突）。"""
    n = pr.n
    poses = [pr.pose(st, i) for i in range(n)]
    ps = [p for p, _ in poses]
    ext = [extents(p, pr.P["zones_inside"]) for p in ps]
    inside = pr.P["zones_inside"]
    pp, pm = relations(st["gp"], st["gm"], n)
    if pr.access:
        acc = [pr.access(p) for p in ps]
        # 与其他块比较的范围：检修区 ∪ 端口接入区（膨胀 δ_ep）
        ext_raw = [(min(e[0], *[r[0] for r in a[0]]), max(e[1], *[r[2] for r in a[0]]),
                    min(e[2], *[r[1] for r in a[0]]), max(e[3], *[r[3] for r in a[0]])) if a[0] else e
                   for e, a in zip(ext, acc)]
        zb = [_bounds(p["zones"]) for p in ps]                          # 检修区本身的范围
        ab = [_bounds([(r[0], r[1], r[2] - r[0], r[3] - r[1]) for r in a[1]]) for a in acc]
    else:
        ext_raw = ext
    if pr.margins:
        mg = [pr.margins(p) for p in ps]                                # (左, 右, 下, 上)
        zb_m = [_bounds(p["zones"]) for p in ps]
    cons = []
    for i in range(n):
        for j in range(n):
            if i == j or pm[i] > pm[j]:
                continue
            if pp[i] < pp[j]:                               # i 在 j 左
                g = max(ps[i]["W"] + pr.d, ext_raw[i][1], ps[i]["W"] - ext_raw[j][0])
                if pr.access:                               # 接入区不压检修区（两个方向）
                    if ab[i] and zb[j]:
                        g = max(g, ab[i][1] - zb[j][0])
                    if zb[i] and ab[j]:
                        g = max(g, zb[i][1] - ab[j][0])
                if pr.margins:                              # 出管留空：两侧叠加；留空不压对方检修区
                    g = max(g, ps[i]["W"] + mg[i][1] + mg[j][0])
                    if zb_m[j]:
                        g = max(g, ps[i]["W"] + mg[i][1] - zb_m[j][0])
                    if zb_m[i]:
                        g = max(g, zb_m[i][1] + mg[j][0])
                cons.append((i, j, 0, g))
            else:                                           # i 在 j 下
                g = max(ps[i]["H"] + pr.d, ext_raw[i][3], ps[i]["H"] - ext_raw[j][2])
                if pr.access:
                    if ab[i] and zb[j]:
                        g = max(g, ab[i][3] - zb[j][2])
                    if zb[i] and ab[j]:
                        g = max(g, zb[i][3] - ab[j][2])
                if pr.margins:
                    g = max(g, ps[i]["H"] + mg[i][3] + mg[j][2])
                    if zb_m[j]:
                        g = max(g, ps[i]["H"] + mg[i][3] - zb_m[j][2])
                    if zb_m[i]:
                        g = max(g, zb_m[i][3] + mg[j][2])
                cons.append((i, j, 1, g))
    lo = [[(-e[0] if inside else 0) for e in ext], [(-e[2] if inside else 0) for e in ext]]
    if pr.margins:                                          # 外圈出管留空也在占地矩形内
        lo = [[max(lo[0][i], mg[i][0]) for i in range(n)], [max(lo[1][i], mg[i][2]) for i in range(n)]]
    pos = _longest(pr, st, cons, lo)
    if pos is None:
        return None
    detour = 0
    if pr.traffic:
        extra, detour = traffic_demand(pr, ps, pos, cons)
        if extra:
            lane = pr.traffic["lane"]
            cons = [(i, j, a, g + lane * extra.get((i, j, a), 0)) for i, j, a, g in cons]
            pos = _longest(pr, st, cons, lo)
            if pos is None:
                return None
    right = [(e[1] if inside else p["W"]) for e, p in zip(ext, ps)]
    top = [(e[3] if inside else p["H"]) for e, p in zip(ext, ps)]
    if pr.margins:
        right = [max(r_, p["W"] + m_[1]) for r_, p, m_ in zip(right, ps, mg)]
        top = [max(t_, p["H"] + m_[3]) for t_, p, m_ in zip(top, ps, mg)]
    Wx = max(pos[0][i] + right[i] for i in range(n))
    Hy = max(pos[1][i] + top[i] for i in range(n))
    W, H = max(Wx, -(-Hy // pr.k)), max(Hy, -(-Wx // pr.k))
    return {"poses": poses, "ext": ext, "cons": cons, "W": W, "H": H, "pos": pos, "right": right, "top": top,
            "lo": lo, "detour": detour}


def _longest(pr, st, cons, lo):
    n = pr.n
    pos = [[0] * n, [0] * n]
    preds = [[[] for _ in range(n)], [[] for _ in range(n)]]
    for i, j, a, g in cons:
        preds[a][j].append((i, g))
    for a, order in ((0, st["gp"]), (1, st["gm"])):
        for j in order:
            v = lo[a][j]
            for i, g in preds[a][j]:
                if pos[a][i] + g > v:
                    v = pos[a][i] + g
            f = pr.fixed[j]
            if f is not None:
                if v > f[a]:
                    return None
                v = f[a]
            pos[a][j] = v
    return pos


def traffic_demand(pr, ps, pos, cons):
    """缝隙穿行需求（启发式估计）。
    每个管网取各端口“出管转弯点”（端口 + (ℓ_min+ρ)·u），按曼哈顿最小生成树连接；
    每段连接试两种 L 形走法。沿轴 a 的一段管（另一轴坐标 c）：
      - 压到设备 b：b 上方还有空余层 → 从上方越过，占用一层；
        没有 → 从 b 较近的一侧绕行，绕行长度计 2×距离，并给该侧与 b 相邻的设备之间的缝隙加 1 根管；
      - 在设备 b 与其相邻设备 k 之间的缝隙里平行穿过：两者上方都没有空余层时，给这条缝隙加 1 根管；
        否则改从上方走，占用空余层较多者的一层。
    走法选择：绕行最短、其次缝隙需求最少。管网按输入顺序依次占用层数。
    返回 ({(i, j, 轴): 管数}, 绕行总长（网格单位）)。"""
    n, tf = pr.n, pr.traffic
    free = list(tf["levels"])
    size = [(p["W"], p["H"]) for p in ps]
    lo_ = [(pos[0][i], pos[1][i]) for i in range(n)]
    hi_ = [(pos[0][i] + size[i][0], pos[1][i] + size[i][1]) for i in range(n)]
    before = [[[] for _ in range(n)], [[] for _ in range(n)]]
    after = [[[] for _ in range(n)], [[] for _ in range(n)]]
    for i, j, a, _g in cons:
        before[a][j].append(i)
        after[a][i].append(j)

    def ov(b, k, a):                                        # 沿轴 a 的投影是否重叠
        return lo_[b][a] < hi_[k][a] and lo_[k][a] < hi_[b][a]

    def leg(a, c, l, r, use):
        """沿轴 a、另一轴坐标 c、范围 [l, r] 的管段 → (需求列表, 绕行长度)；use 记录本走法占用的层。"""
        o = 1 - a
        dem, det = [], 0
        for b in range(n):
            if r <= lo_[b][a] or l >= hi_[b][a]:
                continue
            if lo_[b][o] < c < hi_[b][o]:                   # 压到设备 b
                if free[b] - use.get(b, 0) > 0:
                    use[b] = use.get(b, 0) + 1
                    continue
                down, up = c - lo_[b][o], hi_[b][o] - c
                if down <= up:
                    det += 2 * down
                    ks = [k for k in before[o][b] if ov(b, k, a)]
                    if ks:
                        dem.append((max(ks, key=lambda k: hi_[k][o]), b, o))
                else:
                    det += 2 * up
                    ks = [k for k in after[o][b] if ov(b, k, a)]
                    if ks:
                        dem.append((b, min(ks, key=lambda k: lo_[k][o]), o))
            elif c >= hi_[b][o]:                            # 在 b 与相邻设备 k 之间的缝隙里平行穿过
                ks = [k for k in after[o][b] if ov(b, k, a)]
                if not ks:
                    continue
                k = min(ks, key=lambda k: lo_[k][o])
                if not (c <= lo_[k][o] and l < hi_[k][a] and r > lo_[k][a]):
                    continue
                fb, fk = free[b] - use.get(b, 0), free[k] - use.get(k, 0)
                if fb <= 0 and fk <= 0:
                    dem.append((b, k, o))
                else:
                    x = b if fb >= fk else k
                    use[x] = use.get(x, 0) + 1
        return dem, det

    stub = tf["stub"]
    extra, detour = {}, 0
    for net in pr.nets:
        pts = []
        for i, pn in net["terms"]:
            x, y, ux, uy, _z = ps[i]["ports"][pn]
            pts.append((lo_[i][0] + x + ux * stub, lo_[i][1] + y + uy * stub))
        # 曼哈顿最小生成树（Prim）
        inside, conns = {0}, []
        while len(inside) < len(pts):
            best = min(((abs(pts[u][0] - pts[v][0]) + abs(pts[u][1] - pts[v][1]), u, v)
                        for u in inside for v in range(len(pts)) if v not in inside))
            conns.append((best[1], best[2]))
            inside.add(best[2])
        for u, v in conns:
            (xa, ya), (xb, yb) = pts[u], pts[v]
            opts = []
            for legs in (((0, ya, min(xa, xb), max(xa, xb)), (1, xb, min(ya, yb), max(ya, yb))),
                         ((1, xa, min(ya, yb), max(ya, yb)), (0, yb, min(xa, xb), max(xa, xb)))):
                dem, det, use = [], 0, {}
                for lg in legs:
                    d_, t_ = leg(*lg, use)
                    dem += d_; det += t_
                opts.append((det, len(dem), dem, use))
            det, _, dem, use = min(opts, key=lambda o_: (o_[0], o_[1]))
            for bb, cnt in use.items():
                free[bb] -= cnt
            detour += det
            for key in dem:
                extra[key] = extra.get(key, 0) + 1
    return extra, detour


def trunk_axis(sk, terms):
    """主干轴：端口法向多数沿 X 时主干沿 Y，反之沿 X（法向平分时取 Y）。"""
    nx = sum(1 for i, pn in terms if sk["poses"][i][0]["ports"][pn][2])
    return 1 if nx * 2 >= len(terms) else 0


def trunk_extra(pr, sk, sol):
    """“主干 + 分支”估计 − 跨度估计（网格单位，≥ 0）：搜索目标里多算的那部分。"""
    out = 0.0
    lmin = pr.lmin
    for net in pr.nets:
        t = net["terms"]
        if len(t) < 3:
            continue
        a = 1 - trunk_axis(sk, t)
        vs = []
        for i, pn in t:
            pt = sk["poses"][i][0]["ports"][pn]
            vs.append((sol[i]["x"] if a == 0 else sol[i]["y"]) + pt[a] + lmin * pt[2 + a])
        vs.sort()
        med = vs[len(vs) // 2]
        out += sum(abs(v - med) for v in vs) - (vs[-1] - vs[0])
    return out


def quick_lb(pr, sk, st):
    """不解 LP 的 J 下界：面积(W*,H*) + 管长常数 + 确定的弯头数。"""
    wA, wL, wB = pr.w["area"], pr.w["length"], pr.w["bends"]
    P = pr.P
    L = B = 0
    for e, net in enumerate(pr.nets):
        t = net["terms"]
        if len(t) >= 3:
            L += pr.net_w[e] * (len(t) * pr.lmin + pr.dz[e])
        else:
            (i, pn), (j, qn) = t
            p = sk["poses"][i][0]["ports"][pn]; q = sk["poses"][j][0]["ports"][qn]
            c = base.bend_lb(p, q)
            L += pr.net_w[e] * pr.dz[e]
            if c:
                B += c
            elif not st["straight"][e]:
                B += 2
    return (wA * sk["W"] * sk["H"] / P["A0"] + wL * (L + sk["detour"]) / P["L0"] + wB * B / P["B0"])


def solve_lp(pr, sk, st):
    """返回整数坐标解 sol（validate 格式）或 None。"""
    n, lmin, k = pr.n, pr.lmin, pr.k
    P, inside = pr.P, pr.P["zones_inside"]
    wA, wL = pr.w["area"], pr.w["length"]
    rows, cols, vals, rhs = [], [], [], []
    eq_rows, eq_cols, eq_vals, eq_rhs = [], [], [], []
    r = 0

    def le(terms, b):                                   # Σ c·v ≤ b
        nonlocal r
        for c_, v in terms:
            rows.append(r); cols.append(v); vals.append(c_)
        rhs.append(b); r += 1

    re = 0

    def eq(terms, b):
        nonlocal re
        for c_, v in terms:
            eq_rows.append(re); eq_cols.append(v); eq_vals.append(c_)
        eq_rhs.append(b); re += 1

    X = lambda i: i
    Y = lambda i: n + i
    WV, HV = 2 * n, 2 * n + 1
    nv = 2 * n + 2
    cost = [0.0] * nv
    bounds = [(0, None)] * nv
    for i in range(n):
        bounds[X(i)] = (sk["lo"][0][i], None); bounds[Y(i)] = (sk["lo"][1][i], None)
        if pr.fixed[i] is not None:
            bounds[X(i)] = (pr.fixed[i][0], pr.fixed[i][0]); bounds[Y(i)] = (pr.fixed[i][1], pr.fixed[i][1])
    for i, j, a, g in sk["cons"]:
        v = X if a == 0 else Y
        le([(1, v(i)), (-1, v(j))], -g)
    for i in range(n):
        le([(1, X(i)), (-1, WV)], -sk["right"][i])
        le([(1, Y(i)), (-1, HV)], -sk["top"][i])
    le([(1, HV), (-k, WV)], 0)
    le([(1, WV), (-k, HV)], 0)
    # 面积切平面
    cost[WV] = wA * sk["H"] / P["A0"]
    cost[HV] = wA * sk["W"] / P["A0"]
    cL = wL / P["L0"]

    def new_var(c_=0.0, bnd=(None, None)):
        nonlocal nv
        cost.append(c_); bounds.append(bnd); nv += 1
        return nv - 1

    def abs_of(terms, const):
        """t ≥ |Σ terms + const|"""
        t = new_var(0.0, (0, None))
        le([(c_, v) for c_, v in terms] + [(-1, t)], -const)
        le([(-c_, v) for c_, v in terms] + [(-1, t)], const)
        return t

    B = 0
    for e, net in enumerate(pr.nets):
        t = net["terms"]
        cLe = cL * pr.net_w[e]
        if len(t) >= 3:
            axes = (0, 1)
            if pr.trunk:
                ta = trunk_axis(sk, t)                      # 主干轴：与多数端口法向垂直
                tv = new_var(0.0, (None, None))             # 主干在垂直轴上的坐标
                for i, pn in t:                             # 各端点到主干的垂距
                    pt = sk["poses"][i][0]["ports"][pn]
                    a = 1 - ta
                    off = pt[a] + lmin * pt[2 + a]
                    d = new_var(cLe, (0, None))
                    var = X if a == 0 else Y
                    le([(1, var(i)), (-1, tv), (-1, d)], -off)
                    le([(-1, var(i)), (1, tv), (-1, d)], off)
                axes = (ta,)
            for a in axes:                                  # 跨度（trunk 时只算主干轴）
                var = X if a == 0 else Y
                hi = new_var(cLe, (None, None)); lo = new_var(-cLe, (None, None))
                for i, pn in t:
                    pt = sk["poses"][i][0]["ports"][pn]
                    off = pt[a] + lmin * pt[2 + a]
                    le([(1, var(i)), (-1, hi)], -off)       # hi ≥ x_i + off
                    le([(-1, var(i)), (1, lo)], off)        # lo ≤ x_i + off
            continue
        (i, pn), (j, qn) = t
        p = sk["poses"][i][0]["ports"][pn]; q = sk["poses"][j][0]["ports"][qn]
        c = base.bend_lb(p, q)
        L = new_var(cLe, (0, None))
        ax = abs_of([(1, X(i)), (-1, X(j))], p[0] - q[0])
        ay = abs_of([(1, Y(i)), (-1, Y(j))], p[1] - q[1])
        le([(1, ax), (1, ay), (-1, L)], -pr.dz[e])
        if c == 0 and st["straight"][e]:
            if p[2] != 0:                                   # 沿 X 方向：y 对齐，(qx − px)·ux ≥ ℓ_min
                eq([(1, Y(i)), (-1, Y(j))], q[1] - p[1])
                u = p[2]
                le([(u, X(i)), (-u, X(j))], u * (q[0] - p[0]) - lmin)
            else:
                eq([(1, X(i)), (-1, X(j))], q[0] - p[0])
                u = p[3]
                le([(u, Y(i)), (-u, Y(j))], u * (q[1] - p[1]) - lmin)
        else:
            sx = abs_of([(1, X(i)), (-1, X(j))], p[0] + lmin * p[2] - q[0] - lmin * q[2])
            sy = abs_of([(1, Y(i)), (-1, Y(j))], p[1] + lmin * p[3] - q[1] - lmin * q[3])
            le([(1, sx), (1, sy), (-1, L)], -(2 * lmin + pr.dz[e]))

    A_ub = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    kw = {}
    if re:
        kw = dict(A_eq=coo_matrix((eq_vals, (eq_rows, eq_cols)), shape=(re, nv)).tocsr(), b_eq=eq_rhs)
    res = linprog(cost, A_ub=A_ub, b_ub=rhs, bounds=bounds, method="highs-ds", **kw)
    if res.status != 0:
        return None
    xs = res.x[:2 * n]
    ri = np.rint(xs)
    if np.max(np.abs(xs - ri)) > 1e-6:
        pr.stats["fractional"] += 1
    return [{"x": int(ri[X(i)]), "y": int(ri[Y(i)]), "pose": sk["poses"][i][1]} for i in range(n)]


def evaluate(pr, st, threshold=math.inf):
    """返回 (J, sol)；不可行或被阈值剪掉时 J = inf。"""
    pr.stats["evals"] += 1
    sk = skeleton(pr, st)
    if sk is None:
        pr.stats["infeasible_sp"] += 1
        return math.inf, None
    if quick_lb(pr, sk, st) > threshold:
        pr.stats["pruned"] += 1
        return math.inf, None
    sol = solve_lp(pr, sk, st)
    if sol is None:
        pr.stats["infeasible_lp"] += 1
        return math.inf, None
    pr.stats["lp"] += 1
    try:
        v = base.validate(pr.blocks, pr.nets, pr.P, sol, pr.w, "v2")
    except AssertionError:
        pr.stats["invalid"] += 1
        return math.inf, None
    # 启用穿行容量时，搜索目标另加估计的绕行长度；布管引导时另加各管网权重超出 1 的部分（校验器的 J 不含这两项）
    extra = sum((w - 1.0) * x["L_lb_m"] for w, x in zip(pr.net_w, v["per_net"])) * 1000 / pr.P["grid"]
    if pr.trunk:
        extra += trunk_extra(pr, sk, sol)
    return v["J_full_weights"] + pr.w["length"] * (sk["detour"] + extra) / pr.P["L0"], sol


GUIDE_REQUIRED = ["fn", "every_s", "ema", "w_max"]


def route_score(pr, sol, guide):
    """粗网格布管评估一个解：返回 (J_route, 布管结果)。"""
    r = guide["fn"](sol)
    if not r["ok"]:
        return math.inf, r
    v = base.validate(pr.blocks, pr.nets, pr.P, sol, pr.w, "v2")
    L_route = r["L_m"] * 1000 / pr.P["grid"]
    return v["J_full_weights"] + pr.w["length"] * (L_route - v["_L"]) / pr.P["L0"], {**r, "lb": v["per_net"]}


def update_weights(pr, r, guide):
    for e, x in enumerate(r["lb"]):
        Lc = r["L_net_m"].get(x["net"])
        if Lc is None or x["L_lb_m"] <= 0:
            continue
        target = min(guide["w_max"], max(1.0, Lc / x["L_lb_m"]))
        pr.net_w[e] = (1 - guide["ema"]) * pr.net_w[e] + guide["ema"] * target


# ============================================================ 状态与邻域
def random_state(pr, rng):
    n = pr.n
    gp, gm = list(range(n)), list(range(n))
    rng.shuffle(gp); rng.shuffle(gm)
    # 固定块放到 Γ⁻ 最前（固定在原点附近时必要）；是否可行仍由最长路判断
    fixed = [i for i in range(n) if pr.fixed[i] is not None]
    gm = fixed + [i for i in gm if i not in fixed]
    return {"gp": gp, "gm": gm,
            "rot": [rng.randrange(len(pr.rots[i])) for i in range(n)],
            "var": [rng.randrange(len(pr.variants[mem[0]])) for mem in pr.groups],
            "straight": [rng.random() < 0.5 for _ in pr.nets]}


def copy_state(st):
    return {k: list(v) for k, v in st.items()}


def neighbor(pr, st, rng):
    s = copy_state(st)
    n = pr.n
    m = rng.random()
    if m < 0.25:
        a, b = rng.sample(range(n), 2); s["gp"][a], s["gp"][b] = s["gp"][b], s["gp"][a]
    elif m < 0.5:
        a, b = rng.sample(range(n), 2); s["gm"][a], s["gm"][b] = s["gm"][b], s["gm"][a]
    elif m < 0.6:
        i, j = rng.sample(range(n), 2)
        for key in ("gp", "gm"):
            a, b = s[key].index(i), s[key].index(j)
            s[key][a], s[key][b] = s[key][b], s[key][a]
    elif m < 0.7:
        key = rng.choice(("gp", "gm"))
        b = s[key].pop(rng.randrange(n)); s[key].insert(rng.randrange(n), b)
    elif m < 0.85:
        i = rng.randrange(n)
        if len(pr.rots[i]) > 1:
            s["rot"][i] = rng.choice([r for r in range(len(pr.rots[i])) if r != s["rot"][i]])
    elif m < 0.9:
        gi = rng.randrange(len(pr.groups))
        nv = len(pr.variants[pr.groups[gi][0]])
        if nv > 1:
            s["var"][gi] = rng.choice([v for v in range(nv) if v != s["var"][gi]])
    else:
        e = rng.choice(pr.two)
        s["straight"][e] = not s["straight"][e]
    return s


def encode(st):
    return [*st["gp"], *st["gm"], *st["rot"], *st["var"], *map(int, st["straight"])]


def decode(pr, arr):
    n, g, e = pr.n, len(pr.groups), len(pr.nets)
    a = list(arr)
    return {"gp": a[:n], "gm": a[n:2 * n], "rot": a[2 * n:3 * n], "var": a[3 * n:3 * n + g],
            "straight": [bool(v) for v in a[3 * n + g:3 * n + g + e]]}


# ============================================================ 模拟退火（单进程）
def anneal(args):
    (path, weights, budget, t_start, seed, cycle, T0, T1, access, margins, guide, trunk, shared) = args
    blocks, nets, P = base.load(path)
    pr = Problem(blocks, nets, P, weights or P["w"], access, margins, trunk)
    if guide is not None:
        miss = [k for k in GUIDE_REQUIRED if k not in guide]
        if miss:
            raise KeyError(f"布管引导缺少参数：{miss}")
    pr.stats = dict.fromkeys(("evals", "infeasible_sp", "pruned", "infeasible_lp", "lp", "invalid", "fractional"), 0)
    rng = random.Random(seed)
    gJ, gArr, lock = shared
    cur = random_state(pr, rng)
    Jc, solc = evaluate(pr, cur)
    for _ in range(1000):
        if Jc < math.inf:
            break
        cur = random_state(pr, rng); Jc, solc = evaluate(pr, cur)
    best = (Jc, copy_state(cur), solc)
    trace = [(time.time() - t_start, Jc, solc)] if guide is None else []
    cyc0 = time.time()
    rbest = (math.inf, None, None)                          # 布管引导：按 J_route 的最好解
    guide_log = []
    last_guide = -math.inf

    def guide_tick():
        nonlocal Jc, solc, rbest, best, last_guide
        Jr, r = route_score(pr, solc, guide)
        last_guide = time.time()
        guide_log.append({"t": round(last_guide - t_start, 1), "J_route": Jr, "ok": r["ok"],
                          "L_route_m": r.get("L_m"), "w_max_now": round(max(pr.net_w), 2)})
        if Jr < rbest[0]:
            rbest = (Jr, copy_state(cur), solc)
            trace.append((last_guide - t_start, Jr, solc))
            with lock:
                if Jr < gJ.value:
                    gJ.value = Jr
                    gArr[:] = encode(cur)
        if r["ok"]:
            update_weights(pr, r, guide)
            Jc, solc = evaluate(pr, cur)                     # 权重变了：按新目标重算当前解
            best = (Jc, copy_state(cur), solc)

    while True:
        now = time.time()
        if now - t_start > budget:
            break
        if guide is not None and solc is not None and now - last_guide >= guide["every_s"]:
            guide_tick()
            now = time.time()
        frac = (now - cyc0) / cycle
        if frac >= 1:                                       # 周期结束：从全局最好解重新加热
            with lock:
                mine = rbest[0] if guide is not None else Jc
                take = gJ.value < mine
                arr = gArr[:] if take else None
            if take:
                cur = decode(pr, arr); Jc, solc = evaluate(pr, cur)
            cyc0 = now; frac = 0
        T = T0 * (T1 / T0) ** frac
        cand = neighbor(pr, cur, rng)
        u = rng.random()
        thr = Jc - T * math.log(max(u, 1e-300))             # 接受 ⇔ J_cand ≤ thr
        Jn, soln = evaluate(pr, cand, thr)
        if Jn <= thr:
            cur, Jc, solc = cand, Jn, soln
            if Jc < best[0] - 1e-12:
                best = (Jc, copy_state(cur), solc)
                if guide is None:
                    trace.append((time.time() - t_start, Jc, solc))
                    with lock:
                        if Jc < gJ.value:
                            gJ.value = Jc
                            gArr[:] = encode(cur)
    if guide is not None:
        return {"seed": seed, "best_J": rbest[0], "solution": rbest[2], "trace": trace, "stats": pr.stats,
                "guide_log": guide_log, "net_w": [round(w, 2) for w in pr.net_w]}
    return {"seed": seed, "best_J": best[0], "solution": best[2], "trace": trace, "stats": pr.stats}


def run(path, budget, procs=12, seed=0, cycle=10.0, T0=0.05, T1=0.001, weights=None, access=None, margins=None,
        guide=None, trunk=False):
    """并行退火；返回全局最好解与合并后的“时间—最好 J”曲线。
    access = {"D_mm", "delta_ep_mm"} 时启用端口接入区约束（默认不启用，与第 9 节实验一致）。
    margins = {"D_mm", "delta_pp_mm", "delta_ep_mm", "c_rho"} 时启用按侧出管留空（见 side_margins_factory）；
      再加 "traffic": True, "heights_mm": {块 id: 高度}, "z_max_mm", "K" 时启用缝隙穿行容量（见 traffic_demand）。
    guide = {"fn": 解 → {"ok", "L_m", "L_net_m"}（须可跨进程序列化）, "every_s", "ema", "w_max"} 时启用布管引导，
      返回的 J 为 J_route（见模块说明）。
    trunk=True 时多端点管网按“主干 + 分支”估计管长（见模块说明）。"""
    t_start = time.time()
    blocks, nets, P = base.load(path)
    pr = Problem(blocks, nets, P, weights or P["w"])
    size = 3 * pr.n + len(pr.groups) + len(pr.nets)
    ctx = mp.get_context("spawn")
    mgr_lock = ctx.Lock()
    gJ = ctx.Value("d", math.inf, lock=False)
    gArr = ctx.Array("i", size, lock=False)
    jobs = [(path, weights, budget, t_start, seed * 1000 + k, cycle, T0, T1, access, margins, guide, trunk,
             (gJ, gArr, mgr_lock))
            for k in range(procs)]
    if procs == 1:
        outs = [anneal(jobs[0])]
    else:
        with ctx.Pool(procs, initializer=_init, initargs=((gJ, gArr, mgr_lock),)) as pool:
            outs = pool.map(_anneal_shared, [j[:-1] for j in jobs])
    events = sorted((t, J, sol) for o in outs for t, J, sol in o["trace"] if J < math.inf)
    curve, bestJ, bestSol = [], math.inf, None
    for t, J, sol in events:
        if J < bestJ:
            bestJ, bestSol = J, sol
            curve.append((t, J))
    stats = {k: sum(o["stats"][k] for o in outs) for k in outs[0]["stats"]}
    out = {"J": bestJ, "solution": bestSol, "curve": curve, "stats": stats, "elapsed": time.time() - t_start}
    if guide is not None:
        out["guide"] = [{"seed": o["seed"], "log": o["guide_log"], "net_w": o["net_w"]} for o in outs]
    return out


_SHARED = None


def _init(shared):
    global _SHARED
    _SHARED = shared


def _anneal_shared(job):
    return anneal((*job, _SHARED))


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    path = base.HERE / "算例.json"
    r = run(path, budget, procs, seed)
    blocks, nets, P = base.load(path)
    v = base.validate(blocks, nets, P, r["solution"], P["w"], "v2")
    print(f"用时 {r['elapsed']:.1f} s  J={v['J_full_weights']}  占地 {v['area_m2']} m²（{v['W_m']}×{v['H_m']} m）  "
          f"管长下界 {v['L_lb_m']} m  弯头下界 {v['bends_lb']}")
    print("统计：", r["stats"])
    print("时间—最好 J：", [(round(t, 1), round(J, 4)) for t, J in r["curve"]])


if __name__ == "__main__":
    main()
