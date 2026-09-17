"""布管原型：给定设备位置，按主文档 12.3 / 12.4 / 12.6 布管。

搜索空间（非均匀轨道线）：每个轴的候选坐标 = 基础间距 pitch 的整数倍 ∪ 端口坐标
  ∪ 设备盒按 (δ_ep + D/2) 膨胀后的边界 ∪ 检修区按 D/2 膨胀后的边界 ∪ 端口坐标 ± (D + δ_pp)。
  管道中心线只走这些线的交点之间的轴向边。

单根管 A*，状态 = (节点, 方向, 上一个管件是否为端口, 已发生高度变化次数, 是否已有水平段, 自上个管件起的直管长度)
  - 只沿 ±X ±Y ±Z；只有 90° 弯头；不允许 180° 掉头
  - 直管长度：两端管件中心之间 ≥ ρ_a + ρ_b + ℓ_min（端口 ρ = 0，弯头、三通 ρ = c_ρ·D）
  - 高度变化：两个水平段之间的一段连续竖直管计 1 次；> K 剪枝（hard）
  - 代价 = w_L·长度/L₀ + w_B·弯头/B₀ + w_C·高度变化/C₀，另加拥堵代价
多端点管网：先连最近的两个端点，其余端点依次连到已布的树上（在直管段内部、垂直接入处放三通，
  三通两侧直管同样满足最小长度）。
多根管：协商布线（PathFinder 变体；并行时共享拥堵地图 + 冲突感知调度，见 negotiate）。每根管的“占用晕”= 与其中心线 Chebyshev 距离 < D + δ_pp 的所有边；
  其他管走进晕内的边就要付拥堵代价；每轮全部重布，冲突边累积历史代价，直到无冲突或到达轮数上限。

独立校验器 check_routes 只看折线几何，不依赖搜索内部数据。

简化（报告中须注明）：
  - 管道截面按边长 D 的正方形处理（比圆管保守）
  - 管长按中心线折线计（弯头处按直角角点，不扣弯曲弧长）
  - 三通按点处理，两侧直管按 ρ 计；未建斜三通、未处理两入口交换
  - 同一管网内部各支路之间不检查净距
  - 本算例管径全部相同，未实现变径段（遇到不同管径直接报错）
  - 多端点管网的高度变化按支路分别计数
  - astar_weight > 1 时为加权 A*：单根管代价最多为最优的 astar_weight 倍
"""
import bisect
import heapq
import math
import time
from itertools import count

import numpy as np

DIRS = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]      # (轴, 符号)
DIR_OF = {(1, 0, 0): 0, (-1, 0, 0): 1, (0, 1, 0): 2, (0, -1, 0): 3, (0, 0, 1): 4, (0, 0, -1): 5}
REQUIRED = ["D_default_mm", "c_rho", "delta_ep_mm", "delta_pp_mm", "K", "eps_z_mm", "z_max_mm",
            "service_zone_height_mm", "pitch_mm", "margin_mm", "max_iters", "pres_fac_init", "pres_fac_mult",
            "hist_fac", "max_expansions", "astar_weight", "route_workers"]


def check_params(rp, weights):
    miss = [k for k in REQUIRED if k not in rp] + [f"weights.{k}" for k in ("area", "length", "bends", "height_changes")
                                                   if k not in weights]
    if miss:
        raise KeyError(f"布管缺少必填参数：{', '.join(miss)}（不设默认值）")


# ============================================================ 几何输入
class Scene:
    """devices: {id: {"box": (x0,y0,z0,x1,y1,z1), "zones": [(x0,y0,x1,y1)], "ports": {pn: (x,y,z,ux,uy,uz)}}}
    nets: [{"id", "terms": [(dev, port)]}]；单位 mm。"""

    def __init__(self, devices, nets, rp, weights, scale):
        check_params(rp, weights)
        self.dev, self.nets, self.rp, self.w = devices, nets, rp, weights
        self.D = rp["D_default_mm"]
        self.rho = rp["c_rho"] * self.D
        self.lmin = scale["l_min_mm"]
        self.r_ep = rp["delta_ep_mm"] + self.D / 2
        self.r_pp = self.D + rp["delta_pp_mm"]
        self.scale = scale                                      # A0 (mm²)、L0 (mm)、B0、C0、kappa
        for n in nets:
            if n.get("D_mm", self.D) != self.D:
                raise NotImplementedError("原型未实现变径段：所有管网须同径")
        self.boxes = []                                         # (x0,y0,z0,x1,y1,z1, r, owner)
        zh = rp["service_zone_height_mm"]
        for did, d in devices.items():
            self.boxes.append((*d["box"], self.r_ep, did))
            for zx0, zy0, zx1, zy1 in d["zones"]:
                self.boxes.append((zx0, zy0, 0, zx1, zy1, zh, self.D / 2, None))

    def port(self, dev, pn):
        return self.dev[dev]["ports"][pn]


# ============================================================ 网格
class Grid:
    def __init__(self, sc):
        rp, D = sc.rp, sc.D
        p, mg = rp["pitch_mm"], rp["margin_mm"]
        xs0 = min(d["box"][0] for d in sc.dev.values()) - mg
        ys0 = min(d["box"][1] for d in sc.dev.values()) - mg
        xs1 = max(d["box"][3] for d in sc.dev.values()) + mg
        ys1 = max(d["box"][4] for d in sc.dev.values()) + mg
        zlo, zhi = D / 2, rp["z_max_mm"] - D / 2
        lim = [(xs0, xs1), (ys0, ys1), (zlo, zhi)]
        cs = [set(), set(), set()]
        for a in (0, 1):
            lo, hi = lim[a]
            cs[a].update(range(int(math.floor(lo / p)) * p, int(hi) + 1, p))
        cs[2].update(range(int(math.ceil(zlo / p)) * p, int(zhi) + 1, p))
        for b in sc.boxes:
            r = b[6]
            for a in range(3):
                cs[a].update((b[a] - r, b[a + 3] + r))
        for d in sc.dev.values():
            for x, y, z, *_ in d["ports"].values():
                for a, v in enumerate((x, y, z)):
                    cs[a].update((v, v - sc.r_pp, v + sc.r_pp))
        self.c = [sorted(v for v in cs[a] if lim[a][0] - 1e-9 <= v <= lim[a][1] + 1e-9) for a in range(3)]
        self.n = [len(v) for v in self.c]
        self.sc = sc
        nx_, ny_, nz_ = self.n
        self.size = nx_ * ny_ * nz_
        blocked = np.zeros(self.n, dtype=bool)
        eblocked = [np.zeros(self.n, dtype=bool) for _ in range(3)]      # 边 (i→i+1) 沿各轴
        for b in sc.boxes:
            self._mark(b, blocked, eblocked)
        self.blocked = blocked.ravel().tolist()
        self.eblocked = [e.ravel().tolist() for e in eblocked]
        self.stride = (ny_ * nz_, nz_, 1)

    def _open_range(self, a, lo, hi):
        """坐标严格落在 (lo, hi) 内的下标区间 [i0, i1)。"""
        c = self.c[a]
        return bisect.bisect_right(c, lo + 1e-9), bisect.bisect_left(c, hi - 1e-9)

    def _mark(self, b, blocked, eblocked):
        r = b[6]
        rng = [self._open_range(a, b[a] - r, b[a + 3] + r) for a in range(3)]
        sl = tuple(slice(i0, i1) for i0, i1 in rng)
        blocked[sl] = True
        for a in range(3):
            c = self.c[a]
            lo, hi = b[a] - r, b[a + 3] + r
            # 边 (i, i+1) 的开区间与 (lo, hi) 相交 ⇔ c[i] < hi 且 c[i+1] > lo
            i0 = max(0, bisect.bisect_right(c, lo + 1e-9) - 1)
            i1 = bisect.bisect_left(c, hi - 1e-9)
            s = list(sl)
            s[a] = slice(i0, max(i0, min(i1, self.n[a] - 1)))
            eblocked[a][tuple(s)] = True

    def node(self, pt):
        idx = []
        for a in range(3):
            i = bisect.bisect_left(self.c[a], pt[a] - 1e-6)
            if i >= self.n[a] or abs(self.c[a][i] - pt[a]) > 1e-6:
                raise ValueError(f"点 {pt} 不在网格上（轴 {a}）")
            idx.append(i)
        return (idx[0] * self.n[1] + idx[1]) * self.n[2] + idx[2]

    def ijk(self, n):
        return n // self.stride[0], (n // self.n[2]) % self.n[1], n % self.n[2]

    def pt(self, n):
        i, j, k = self.ijk(n)
        return (self.c[0][i], self.c[1][j], self.c[2][k])

    def step(self, n, d):
        """沿方向 d 走一格：返回 (邻点, 边下端点, 轴, 长度) 或 None。"""
        a, s = DIRS[d]
        ijk = self.ijk(n)
        m = ijk[a] + s
        if m < 0 or m >= self.n[a]:
            return None
        nb = n + s * self.stride[a]
        lower = n if s > 0 else nb
        return nb, lower, a, abs(self.c[a][m] - self.c[a][ijk[a]])

    def free_for(self, pt, owner_ok):
        """几何判断点是否不在障碍内（跳过 owner_ok 设备本身）。"""
        for b in self.sc.boxes:
            if b[7] is not None and b[7] == owner_ok:
                continue
            r = b[6]
            if all(b[a] - r < pt[a] < b[a + 3] + r for a in range(3)):
                return False
        return True


# ============================================================ 单根管 / 支路 A*
def port_exemptions(G, sc, terms):
    """端口接入段豁免：从端口沿法向走 δ_ep + D/2 的点与边，只豁免端口所属设备本身。"""
    nodes, edges = set(), set()
    for dev, pn in terms:
        x, y, z, ux, uy, uz = sc.port(dev, pn)
        n = G.node((x, y, z))
        d = DIR_OF[(ux, uy, uz)]
        dist = 0.0
        nodes.add(n)
        while dist < sc.r_ep - 1e-9:
            st = G.step(n, d)
            if st is None:
                break
            nb, lower, a, ln = st
            mid = tuple((G.pt(n)[k] + G.pt(nb)[k]) / 2 for k in range(3))
            if not G.free_for(mid, dev):
                break
            edges.add((lower, a))
            dist += ln
            n = nb
            if G.free_for(G.pt(n), dev):
                nodes.add(n)
    return nodes, edges


def astar(G, sc, start, target, ctx):
    """start = (dev, port)；target = ("port", (dev, port)) 或 ("tree", tree)。
    返回 (折线点列表, 结束信息) 或 None。"""
    rho, lmin, K = sc.rho, sc.lmin, sc.rp["K"]
    cap = 2 * rho + lmin
    cL = sc.w["length"] / sc.scale["L0"]
    cB = sc.w["bends"] / sc.scale["B0"]
    cC = sc.w["height_changes"] / sc.scale["C0"]
    halo, hist, pres = ctx["halo"], ctx["hist"], ctx["pres"]
    ex_nodes, ex_edges = ctx["ex_nodes"], ctx["ex_edges"]
    blocked, eblocked = G.blocked, G.eblocked
    hw = sc.rp["astar_weight"]                                      # > 1：加权 A*，更快但不保证单管最优

    sx, sy, sz, sux, suy, suz = sc.port(*start)
    s = G.node((sx, sy, sz))
    d0 = DIR_OF[(sux, suy, suz)]
    if target[0] == "port":
        tx, ty, tz, tux, tuy, tuz = sc.port(*target[1])
        t = G.node((tx, ty, tz))
        t_dir = DIR_OF[(-tux, -tuy, -tuz)]
        tpt = (tx, ty, tz)
        ta, ts = DIRS[t_dir]

        def h(p, d):
            """管长曼哈顿距离 + 至少还需的弯头数（可采纳）。"""
            base_ = cL * (abs(p[0] - tpt[0]) + abs(p[1] - tpt[1]) + abs(p[2] - tpt[2]))
            a, s_ = DIRS[d]
            if a == ta:
                if s_ != ts:
                    nb_ = 2
                else:
                    aligned = all(abs(p[b] - tpt[b]) < 1e-9 for b in range(3) if b != a)
                    ahead = (tpt[a] - p[a]) * s_ >= 0
                    nb_ = 0 if (aligned and ahead) else 2
            else:
                nb_ = 1
            return base_ + cB * nb_
        tree_nodes = {}
    else:
        tree = target[1]
        tree_nodes = tree["nodes"]
        bb = tree["bbox"]
        h = lambda p, d: cL * sum(max(0, bb[0][a] - p[a], p[a] - bb[1][a]) for a in range(3))

    # 状态：(节点, 方向, 上个管件是端口, 高度变化次数, 已有水平段) → [(run, g)]
    best = {}
    states = []                                                    # (node, dir, lastport, vr, hh, run, g, parent)
    heap = []
    tie = count()

    def push(n, d, lp, vr, hh, run, g, parent, goal):
        key = (n, d, lp, vr, hh)
        lst = best.get(key)
        if lst is not None:
            for r2, g2 in lst:
                if r2 >= run and g2 <= g + 1e-15:
                    return
            lst[:] = [(r2, g2) for r2, g2 in lst if not (run >= r2 and g <= g2)]
            lst.append((run, g))
        else:
            best[key] = [(run, g)]
        states.append((n, d, lp, vr, hh, run, g, parent, goal))
        heapq.heappush(heap, (g + (0 if goal else hw * h(G.pt(n), d)), next(tie), len(states) - 1))

    push(s, d0, 1, 0, 1 if DIRS[d0][0] != 2 else 0, 0, 0.0, -1, None)
    exp = 0
    while heap:
        _, _, sid = heapq.heappop(heap)
        n, d, lp, vr, hh, run, g, parent, goal = states[sid]
        if goal is not None:
            ctx["stats"]["expansions"] += exp
            return _trace(G, states, sid), goal
        exp += 1
        if exp > sc.rp["max_expansions"]:
            ctx["stats"]["expansions"] += exp
            ctx["stats"]["exhausted"] += 1
            return None
        need_turn = (0 if lp else rho) + rho + lmin
        for nd in range(6):
            if nd != d and DIRS[nd][0] == DIRS[d][0]:
                continue                                           # 不允许掉头
            turning = nd != d
            if turning and run < need_turn - 1e-9:
                continue
            st = G.step(n, nd)
            if st is None:
                continue
            nb, lower, a, ln = st
            if eblocked[a][lower] and (lower, a) not in ex_edges:
                continue
            if blocked[nb] and nb not in ex_nodes:
                continue
            ng = g
            nlp, nvr, nhh = lp, vr, hh
            if turning:
                ng += cB
                nlp = 0
                nrun = ln
                if DIRS[d][0] == 2 and a != 2 and hh:
                    nvr += 1
                    ng += cC
                    if nvr > K:
                        continue
            else:
                nrun = min(run + ln, cap)
            if a != 2:
                nhh = 1
            ei = lower * 3 + a                                         # 边下标：共享占用 / 历史代价表
            ng += cL * ln * (1 + hist[ei]) * (1 + pres * halo[ei])
            goal_info = None
            if target[0] == "port":
                if nb == t:
                    if nd == t_dir and nrun >= (0 if nlp else rho) + lmin - 1e-9:
                        goal_info = ("port", target[1])
                    else:
                        continue
            elif nb in tree_nodes:
                seg_axis, dA, kA, dB, kB, seg_id = tree_nodes[nb]
                ok = (a != seg_axis and nrun >= (0 if nlp else rho) + rho + lmin - 1e-9
                      and dA >= (0 if kA == "port" else rho) + rho + lmin - 1e-9
                      and dB >= (0 if kB == "port" else rho) + rho + lmin - 1e-9)
                if not ok:
                    continue
                goal_info = ("tee", seg_id)
            push(nb, nd, nlp, nvr, nhh, nrun, ng, sid, goal_info)
    ctx["stats"]["expansions"] += exp
    return None


def _trace(G, states, sid):
    seq = []
    while sid >= 0:
        n, d, *_rest, parent, _ = states[sid]
        seq.append((n, d))
        sid = parent
    seq.reverse()
    pts = [G.pt(seq[0][0])]
    for k in range(1, len(seq)):
        if seq[k][1] != seq[k - 1][1]:
            pts.append(G.pt(seq[k - 1][0]))                        # 在上一节点转弯
    pts.append(G.pt(seq[-1][0]))
    return _dedupe(pts)


def _dedupe(pts):
    out = [pts[0]]
    for p in pts[1:]:
        if p != out[-1]:
            out.append(p)
    return out


# ============================================================ 管网树
def segments_of(points):
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


def axis_of(p, q):
    diff = [a for a in range(3) if abs(p[a] - q[a]) > 1e-9]
    if len(diff) != 1:
        raise ValueError(f"非轴向管段：{p} → {q}")
    return diff[0]


def build_tree(G, branches):
    """由已布支路生成可接三通的树：{节点: (段轴, 到端A距离, 端A类型, 到端B距离, 端B类型, 段号)}。"""
    segs = tree_segments(branches)
    nodes, pts = {}, []
    for sid, (p, q, kp, kq) in enumerate(segs):
        a = axis_of(p, q)
        lo, hi = sorted((p[a], q[a]))
        c = G.c[a]
        for i in range(bisect.bisect_left(c, lo - 1e-9), bisect.bisect_right(c, hi + 1e-9)):
            v = c[i]
            pt = list(p); pt[a] = v
            n = G.node(tuple(pt))
            dp, dq = abs(v - p[a]), abs(v - q[a])
            if dp < 1e-9 or dq < 1e-9:
                nodes[n] = (a, -1, "x", -1, "x", sid)              # 段端点：不能接三通
            elif n not in nodes:
                nodes[n] = (a, dp, kp, dq, kq, sid)
        pts += [p, q]
    bbox = ([min(t[a] for t in pts) for a in range(3)], [max(t[a] for t in pts) for a in range(3)])
    return {"nodes": nodes, "bbox": bbox, "segs": segs}


def tree_segments(branches):
    """所有支路 → 直管段列表 (p, q, p 端类型, q 端类型)，三通处拆段。"""
    segs = []
    for br in branches:
        pts = br["points"]
        for i in range(len(pts) - 1):
            kp = "port" if i == 0 else "elbow"
            kq = ("port" if br["end"][0] == "port" else "tee") if i == len(pts) - 2 else "elbow"
            new = (pts[i], pts[i + 1], kp, kq)
            segs.append(new)
        if br["end"][0] == "tee":
            t = pts[-1]
            for k, (p, q, kp_, kq_) in enumerate(segs[:-(len(pts) - 1)]):
                a = axis_of(p, q)
                if all(abs(t[b] - p[b]) < 1e-9 for b in range(3) if b != a) and \
                        min(p[a], q[a]) + 1e-9 < t[a] < max(p[a], q[a]) - 1e-9:
                    segs[k] = (p, t, kp_, "tee")
                    segs.insert(k + 1, (t, q, "tee", kq_))
                    break
            else:
                raise ValueError(f"三通点 {t} 不在已有直管段内部")
    return segs


# ============================================================ 占用晕
def halo_edges(G, sc, branches):
    r = sc.r_pp
    out = set()
    for br in branches:
        for p, q in segments_of(br["points"]):
            lo = [min(p[a], q[a]) for a in range(3)]
            hi = [max(p[a], q[a]) for a in range(3)]
            rng = []
            for a in range(3):
                c = G.c[a]
                rng.append((max(0, bisect.bisect_right(c, lo[a] - r) - 1), min(G.n[a], bisect.bisect_left(c, hi[a] + r) + 1)))
            for i in range(*rng[0]):
                for j in range(*rng[1]):
                    for k in range(*rng[2]):
                        base_pt = (G.c[0][i], G.c[1][j], G.c[2][k])
                        n = (i * G.n[1] + j) * G.n[2] + k
                        ijk = (i, j, k)
                        for a in range(3):
                            if ijk[a] + 1 >= G.n[a]:
                                continue
                            e_lo = list(base_pt); e_hi = list(base_pt)
                            e_hi[a] = G.c[a][ijk[a] + 1]
                            gap = max(max(0.0, e_lo[b] - hi[b], lo[b] - e_hi[b]) for b in range(3))
                            if gap < r - 1e-9:
                                out.add((n, a))
    return out


def path_edges(G, branches):
    out = set()
    for br in branches:
        for p, q in segments_of(br["points"]):
            a = axis_of(p, q)
            lo, hi = sorted((p[a], q[a]))
            c = G.c[a]
            base_pt = list(p)
            for i in range(bisect.bisect_left(c, lo - 1e-9), bisect.bisect_left(c, hi - 1e-9)):
                base_pt[a] = c[i]
                out.add((G.node(tuple(base_pt)), a))
    return out


# ============================================================ 单个管网
def port_access_blocked(G, sc, terms, ex_nodes, ex_edges):
    """必要条件：每个端口正前方至少能直行 ℓ_min。返回被堵的端口列表。"""
    bad = []
    for dev, pn in terms:
        x, y, z, ux, uy, uz = sc.port(dev, pn)
        n, d, dist = G.node((x, y, z)), DIR_OF[(ux, uy, uz)], 0.0
        while dist < sc.lmin - 1e-9:
            st = G.step(n, d)
            if st is None:
                bad.append((dev, pn)); break
            nb, lower, a, ln = st
            if (G.eblocked[a][lower] and (lower, a) not in ex_edges) or (G.blocked[nb] and nb not in ex_nodes):
                bad.append((dev, pn)); break
            dist += ln
            n = nb
    return bad


def route_net(G, sc, net, ctx):
    """返回 (支路列表, None) 或 (None, 失败原因)。"""
    terms = net["terms"]
    ex_nodes, ex_edges = port_exemptions(G, sc, terms)
    bad = port_access_blocked(G, sc, terms, ex_nodes, ex_edges)
    if bad:
        return None, "端口正前方不足 ℓ_min：" + "、".join(f"{d}.{p}" for d, p in bad)
    ctx = {**ctx, "ex_nodes": ex_nodes, "ex_edges": ex_edges}
    P = {t: sc.port(*t)[:3] for t in terms}
    man = lambda u, v: sum(abs(P[u][a] - P[v][a]) for a in range(3))
    if len(terms) == 2:
        pairs = [(terms[0], terms[1])]
    else:
        pairs = sorted(((u, v) for i, u in enumerate(terms) for v in terms[i + 1:]), key=lambda uv: man(*uv))[:1]
    a, b = pairs[0]
    r = astar(G, sc, a, ("port", b), ctx)
    if r is None:
        return None, "搜索未找到路径（或超过扩展上限）"
    branches = [{"start": a, "points": r[0], "end": ("port", b)}]
    rest = [t for t in terms if t not in (a, b)]
    while rest:
        tree = build_tree(G, branches)
        bb = tree["bbox"]
        rest.sort(key=lambda t: sum(max(0, bb[0][k] - P[t][k], P[t][k] - bb[1][k]) for k in range(3)))
        t = rest.pop(0)
        r = astar(G, sc, t, ("tree", tree), ctx)
        if r is None:
            return None, f"支路 {t[0]}.{t[1]} 未接上（或超过扩展上限）"
        branches.append({"start": t, "points": r[0], "end": ("tee", None)})
    return branches, None


# ============================================================ 协商布线
_W = {}


def _worker_init(devices, nets, rp, weights, scale, halo_name, hist_name):
    from multiprocessing import shared_memory
    sc = Scene(devices, nets, rp, weights, scale)
    _W["sc"], _W["G"] = sc, Grid(sc)
    _W["shm"] = (shared_memory.SharedMemory(name=halo_name), shared_memory.SharedMemory(name=hist_name))
    _W["halo"] = _W["shm"][0].buf.cast("i")
    _W["hist"] = _W["shm"][1].buf.cast("f")


def _worker_ready(_):
    return True


def _worker_route(args):
    return _route_one(_W["G"], _W["sc"], _W["halo"], _W["hist"], *args)


class _Zero:
    def __getitem__(self, _):
        return 0


def _route_one(G, sc, halo, hist, e, pres, never_routed):
    """带拥堵代价搜索（A* 中实时读取共享占用表）；失败且该管网从未布通过时，再做一次无拥堵搜索。
    info["free_fail"] = 无拥堵搜索也失败（或端口被堵）：该布局下布不通，之后不再重试。"""
    t0 = time.time()
    stats = {"expansions": 0, "exhausted": 0}
    br, reason = route_net(G, sc, sc.nets[e], {"halo": halo, "hist": hist, "pres": pres, "stats": stats})
    fallback = free_fail = False
    if br is None and reason.startswith("端口正前方"):
        free_fail = True
    elif br is None and never_routed:
        fallback = True
        br, reason = route_net(G, sc, sc.nets[e], {"halo": _Zero(), "hist": _Zero(), "pres": 0.0, "stats": stats})
        free_fail = br is None
    info = {"net": sc.nets[e]["id"], "time_s": round(time.time() - t0, 2), "fallback": fallback,
            "free_fail": free_fail, **stats}
    if br is None:
        return e, None, reason, None, None, info
    return (e, br, None, [n * 3 + ax for n, ax in halo_edges(G, sc, br)],
            [n * 3 + ax for n, ax in path_edges(G, br)], info)


def negotiate(sc, log=print):
    """有交流的并行协商布线（PathFinder 变体 + 乐观并发）。
    共享状态：每条边的占用计数（各管网“晕”的叠加）与历史拥堵代价，放在共享内存里；
      只有主进程写，工作进程在 A* 中实时读——一根管一提交，之后的搜索立即看到。
    乐观并发：route_workers 个进程同时搜索，不预先排队。某管网搜索完成时，主进程检查它的路径
      是否落入“它开始搜索之后才提交”的其他管网的晕里：没有 → 提交；有 → 丢弃结果、按最新地图重搜
      （每轮每个管网最多重搜 max_requeue 次，超过则照常提交，冲突交给下一轮协商）。
    每轮：第 1 轮布全部管网，之后只重布有冲突或未布通的管网；轮末冲突边累积历史代价。
    带拥堵代价搜索失败时保留上一轮路径；无拥堵也布不通的管网记为该布局下不可布，不再重试。
    route_workers = 1 时串行（无并发冲突，逻辑相同）。"""
    G = Grid(sc)
    rp = sc.rp
    workers = rp["route_workers"]
    size = G.size * 3
    ex = shms = None
    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
        from multiprocessing import shared_memory
        shms = (shared_memory.SharedMemory(create=True, size=size * 4),
                shared_memory.SharedMemory(create=True, size=size * 4))
        for sh in shms:
            sh.buf[:size * 4] = bytes(size * 4)
        halo, hist = shms[0].buf.cast("i"), shms[1].buf.cast("f")
        ex = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_worker_init,
                                 initargs=(sc.dev, sc.nets, rp, sc.w, sc.scale, shms[0].name, shms[1].name))
        list(ex.map(_worker_ready, range(workers * 4)))                 # 预先启动全部工作进程
    else:
        from array import array
        halo, hist = array("i", bytes(size * 4)), array("f", bytes(size * 4))
    max_requeue = 2
    routes, halos, pedges, why, hard = {}, {}, {}, {}, {}
    ever = set()
    pres = rp["pres_fac_init"]
    history = []
    commits = []                                                        # 提交日志：各次提交的晕
    todo = sorted(range(len(sc.nets)), key=lambda e: -len(sc.nets[e]["terms"]))

    def rip(e):
        old = (routes.pop(e, None), halos.pop(e, None), pedges.pop(e, None))
        for ei in old[1] or ():
            halo[ei] -= 1
        return old

    def put(e, br, hl, pe):
        routes[e], halos[e], pedges[e] = br, set(hl), pe
        for ei in halos[e]:
            halo[ei] += 1
        commits.append(halos[e])

    def finish(out, old, times, counters):
        e, br, reason, hl, pe, info = out
        times.append(info)
        if br is not None:
            put(e, br, hl, pe); why.pop(e, None); ever.add(e)
        elif info["free_fail"]:
            hard[e] = reason; why[e] = reason
        elif old[0] is not None:
            put(e, *old); counters["kept"] += 1                         # 保留旧路径
        else:
            why[e] = reason

    try:
        for it in range(rp["max_iters"]):
            t_it = time.time()
            times, counters = [], {"kept": 0, "requeued": 0, "max_parallel": 0}
            queue = [e for e in todo if e not in hard]
            if ex is None:
                for e in queue:
                    old = rip(e)
                    finish(_route_one(G, sc, halo, hist, e, pres, e not in ever), old, times, counters)
            else:
                inflight, requeues = {}, {}
                while queue or inflight:
                    while queue and len(inflight) < workers:
                        e = queue.pop(0)
                        old = rip(e) if e not in requeues else requeues[e][1]
                        fut = ex.submit(_worker_route, (e, pres, e not in ever))
                        inflight[fut] = (e, old, len(commits))
                    counters["max_parallel"] = max(counters["max_parallel"], len(inflight))
                    done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                    for fut in done:
                        e, old, seq0 = inflight.pop(fut)
                        out = fut.result()
                        pe = out[4]
                        n_req = requeues.get(e, (0, None))[0]
                        if pe is not None and n_req < max_requeue and \
                                any(ei in h for h in commits[seq0:] for ei in pe):
                            requeues[e] = (n_req + 1, old)                # 搜索期间别人提交了冲突路径：重搜
                            counters["requeued"] += 1
                            queue.insert(0, e)
                            continue
                        finish(out, old, times, counters)
            conflicted, conflicts = set(), 0
            for e, pe in pedges.items():
                own = halos[e]
                for ei in pe:
                    if halo[ei] - (1 if ei in own else 0) > 0:
                        conflicts += 1
                        conflicted.add(e)
                        hist[ei] += rp["hist_fac"]
            retry = {e for e in why if e not in hard}
            slow = sorted(times, key=lambda x: -x["time_s"])
            cpu = sum(x["time_s"] for x in times)
            el = time.time() - t_it
            history.append({"iter": it + 1, "rerouted": len(queue if ex is None else todo), "kept_old": counters["kept"],
                            "requeued": counters["requeued"], "max_parallel": counters["max_parallel"],
                            "conflict_edges": conflicts, "conflicted_nets": len(conflicted), "time_s": round(el, 1),
                            "search_time_sum_s": round(cpu, 1), "net_times": slow,
                            "failed": {sc.nets[e]["id"]: r for e, r in why.items()}})
            log(f"  第 {it + 1} 轮：墙钟 {el:.1f} s / 搜索合计 {cpu:.1f} s（同时最多 {counters['max_parallel']}，"
                f"并发冲突重搜 {counters['requeued']}），冲突边 {conflicts}（{len(conflicted)} 个管网），保留旧路径 "
                f"{counters['kept']}，未布通 {len(why)}（其中不可布 {len(hard)}）；最慢："
                + "、".join(f"{x['net']} {x['time_s']}s{'(回退)' if x['fallback'] else ''}" for x in slow[:3]))
            if not conflicted and not retry:
                break
            todo = sorted(conflicted | retry, key=lambda e: -len(sc.nets[e]["terms"]))
            pres *= rp["pres_fac_mult"]
    finally:
        if ex is not None:
            ex.shutdown(wait=True, cancel_futures=True)
            halo.release(); hist.release()
            for sh in shms:
                sh.close(); sh.unlink()
    return {sc.nets[e]["id"]: br for e, br in routes.items()}, history, G


# ============================================================ 独立校验器
def check_routes(sc, routes):
    """只依据折线几何检查规则并计算指标。返回 (违规列表, 指标)。"""
    D, rho, lmin, K = sc.D, sc.rho, sc.lmin, sc.rp["K"]
    eps = sc.rp["eps_z_mm"]
    viol = []
    per_net, all_boxes = {}, []
    net_of = {n["id"]: n for n in sc.nets}
    for nid, n in net_of.items():
        if nid not in routes:
            viol.append(f"{nid}：未布通")
    for nid, brs in routes.items():
        terms = set(net_of[nid]["terms"])
        seen = set()
        L = bends = layers = 0
        for bi, br in enumerate(brs):
            pts = br["points"]
            x, y, z, ux, uy, uz = sc.port(*br["start"])
            if pts[0] != (x, y, z):
                viol.append(f"{nid} 支路 {bi}：起点不在端口")
            seen.add(tuple(br["start"]))
            dirs = []
            for p, q in segments_of(pts):
                a = axis_of(p, q)
                s = 1 if q[a] > p[a] else -1
                dirs.append((a, s))
                L += abs(q[a] - p[a])
            if dirs[0] != (DIRS[DIR_OF[(ux, uy, uz)]]):
                viol.append(f"{nid} 支路 {bi}：离开端口方向不是端口法向")
            for k in range(1, len(dirs)):
                if dirs[k][0] == dirs[k - 1][0]:
                    viol.append(f"{nid} 支路 {bi}：相邻管段不垂直")
            bends += len(pts) - 2
            if br["end"][0] == "port":
                ex, ey, ez, eux, euy, euz = sc.port(*br["end"][1])
                if pts[-1] != (ex, ey, ez) or dirs[-1] != DIRS[DIR_OF[(-eux, -euy, -euz)]]:
                    viol.append(f"{nid} 支路 {bi}：终点未正对端口接入")
                seen.add(tuple(br["end"][1]))
            # 高度变化：两个水平段之间的连续竖直段计 1 次
            nl, had_h, in_v, dz = 0, False, False, 0.0
            for (p, q), (a, _) in zip(segments_of(pts), dirs):
                if a == 2:
                    in_v = True; dz += q[2] - p[2]
                else:
                    if in_v and had_h and abs(dz) > eps:
                        nl += 1
                    in_v, dz, had_h = False, 0.0, True
            if nl > K:
                viol.append(f"{nid} 支路 {bi}：高度变化 {nl} > K={K}")
            layers += nl
        if seen != terms:
            viol.append(f"{nid}：未连接全部端点")
        try:
            segs = tree_segments(brs)
        except ValueError as ex:
            viol.append(f"{nid}：{ex}")
            segs = []
        for p, q, kp, kq in segs:
            need = (0 if kp == "port" else rho) + (0 if kq == "port" else rho) + lmin
            ln = abs(q[axis_of(p, q)] - p[axis_of(p, q)])
            if ln < need - 1e-6:
                viol.append(f"{nid}：直管段 {p}→{q} 长 {ln:.0f} < {need:.0f}（{kp}–{kq}）")
        # 碰撞
        boxes = []
        for br in brs:
            dev0 = br["start"][0]
            dev1 = br["end"][1][0] if br["end"][0] == "port" else None
            sg = segments_of(br["points"])
            for k, (p, q) in enumerate(sg):
                a = axis_of(p, q)
                s = 1 if q[a] > p[a] else -1
                pp, qq = list(p), list(q)
                trim_dev = set()
                if k == 0:
                    pp[a] += s * sc.r_ep; trim_dev.add(dev0)
                if k == len(sg) - 1 and dev1:
                    qq[a] -= s * sc.r_ep; trim_dev.add(dev1)
                full = _box(p, q, D / 2)
                boxes.append(full)
                chk = _box(tuple(pp), tuple(qq), D / 2) if (s * (qq[a] - pp[a]) > 0) else None
                for did, d in sc.dev.items():
                    b = d["box"]
                    tgt = chk if did in trim_dev else full
                    if tgt is not None and _gap(tgt, b) < sc.rp["delta_ep_mm"] - 1e-6:
                        viol.append(f"{nid}：管段 {p}→{q} 与设备 {did} 净距不足")
                    for zx0, zy0, zx1, zy1 in d["zones"]:
                        zb = (zx0, zy0, 0, zx1, zy1, sc.rp["service_zone_height_mm"])
                        if _gap(full, zb) < 0 - 1e-6:
                            viol.append(f"{nid}：管段 {p}→{q} 穿过 {did} 的检修区")
                if full[2] < -1e-6 or full[5] > sc.rp["z_max_mm"] + 1e-6:
                    viol.append(f"{nid}：管段 {p}→{q} 超出高度范围")
        all_boxes.append((nid, boxes))
        per_net[nid] = {"L_mm": L, "bends": bends, "height_changes": layers, "branches": len(brs)}
    for i in range(len(all_boxes)):
        for j in range(i + 1, len(all_boxes)):
            for bi in all_boxes[i][1]:
                for bj in all_boxes[j][1]:
                    if _gap(bi, bj) < sc.rp["delta_pp_mm"] - 1e-6:
                        viol.append(f"{all_boxes[i][0]} 与 {all_boxes[j][0]}：管–管净距不足")
                        break
                else:
                    continue
                break
    # 占地（设备 + 管道），长宽比补足
    xs = [d["box"][0] for d in sc.dev.values()] + [b[0] for _, bs in all_boxes for b in bs]
    ys = [d["box"][1] for d in sc.dev.values()] + [b[1] for _, bs in all_boxes for b in bs]
    xe = [d["box"][3] for d in sc.dev.values()] + [b[3] for _, bs in all_boxes for b in bs]
    ye = [d["box"][4] for d in sc.dev.values()] + [b[4] for _, bs in all_boxes for b in bs]
    W, H = max(xe) - min(xs), max(ye) - min(ys)
    k = sc.scale["kappa"]
    W, H = max(W, H / k), max(H, W / k)
    Lt = sum(v["L_mm"] for v in per_net.values())
    Bt = sum(v["bends"] for v in per_net.values())
    Ct = sum(v["height_changes"] for v in per_net.values())
    s = sc.scale
    J = (sc.w["area"] * W * H / s["A0"] + sc.w["length"] * Lt / s["L0"] + sc.w["bends"] * Bt / s["B0"]
         + sc.w["height_changes"] * Ct / s["C0"])
    metrics = {"W_m": round(W / 1000, 2), "H_m": round(H / 1000, 2), "area_m2": round(W * H / 1e6, 1),
               "L_m": round(Lt / 1000, 1), "bends": Bt, "height_changes": Ct, "J": round(J, 4), "per_net": per_net}
    return sorted(set(viol)), metrics


def _box(p, q, r):
    return (min(p[0], q[0]) - r, min(p[1], q[1]) - r, min(p[2], q[2]) - r,
            max(p[0], q[0]) + r, max(p[1], q[1]) + r, max(p[2], q[2]) + r)


def _gap(a, b):
    """两个轴对齐盒的间隙：各轴间隙的最大值；重叠时为负。"""
    return max(max(b[k] - a[k + 3], a[k] - b[k + 3]) for k in range(3))
