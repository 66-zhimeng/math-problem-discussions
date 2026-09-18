"""布管原型：给定设备位置，按主文档 12.3 / 12.4 / 12.6 布管。

搜索空间（非均匀轨道线）：每个轴的候选坐标 = 基础间距 pitch 的整数倍 ∪ 端口坐标
  ∪ 设备盒按 (δ_ep + D/2) 膨胀后的边界 ∪ 检修区按 D/2 膨胀后的边界 ∪ 端口坐标 ± (D + δ_pp)。
  管道中心线只走这些线的交点之间的轴向边。

单根管 A*，状态 = (节点, 方向, 上一个管件是否为端口, 已发生高度变化次数, 是否已有水平段, 自上个管件起的直管长度)
  - 只沿 ±X ±Y ±Z；只有 90° 弯头；不允许 180° 掉头
  - 直管长度：两端管件中心之间 ≥ ρ_a + ρ_b + ℓ_min（端口 ρ = 0，弯头、三通 ρ = c_ρ·D）
  - 高度变化：两个水平段之间的一段连续竖直管计 1 次；> K 剪枝（hard）
  - 代价 = w_L·长度/L₀ + w_B·弯头/B₀ + w_C·高度变化/C₀，另加拥堵代价
多端点管网：先连最近的两个端点，其余端点依次连到已布的树上（在直管段内部垂直接入，或沿弯头一条腿的延长线接入并把弯头改为三通；
  三通两侧直管同样满足最小长度）。
多根管：协商布线（PathFinder 变体；并行时共享拥堵地图 + 冲突感知调度，见 negotiate）。每根管的“占用晕”= 与其中心线 Chebyshev 距离 < D + δ_pp 的所有边；
  其他管走进晕内的边就要付拥堵代价；每轮全部重布，冲突边累积历史代价，直到无冲突、到达轮数上限，
  或连续 stall_iters 轮没有改进。之后做清理（cleanup）：其他管网固定为硬障碍，只对仍冲突的管网单独重搜。
  某管网连续 freeze_after_exhausted 次搜到扩展上限（保留旧路径）后不再重搜，其冲突留给清理。
  协商中途冲突管网 ≤ cleanup_trigger_nets 时先试清理（同一组冲突管网只试一次）：清干净即结束，否则继续协商。

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
import copy
import heapq
import math
import time
from itertools import count

import numpy as np

import astar_fast as af

import astar_fast as af

DIRS = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]      # (轴, 符号)
DIR_OF = {(1, 0, 0): 0, (-1, 0, 0): 1, (0, 1, 0): 2, (0, -1, 0): 3, (0, 0, 1): 4, (0, 0, -1): 5}
REQUIRED = ["D_default_mm", "c_rho", "delta_ep_mm", "delta_pp_mm", "K", "eps_z_mm", "z_max_mm",
            "service_zone_height_mm", "pitch_mm", "margin_mm", "max_iters", "pres_fac_init", "pres_fac_mult",
            "hist_fac", "max_expansions", "astar_weight", "route_workers", "stall_iters", "cleanup_max_expansions",
            "cleanup_trigger_nets", "freeze_after_exhausted", "port_side_lines"]


def check_params(rp, weights):
    miss = [k for k in REQUIRED if k not in rp] + [f"weights.{k}" for k in ("area", "length", "bends", "height_changes")
                                                   if k not in weights]
    if miss:
        raise KeyError(f"布管缺少必填参数：{', '.join(miss)}（不设默认值）")


# ============================================================ 几何输入
class Scene:
    """devices: {id: {"box": (x0,y0,z0,x1,y1,z1) 或 None, "zones": [(x0,y0,x1,y1)], "ports": {pn: (x,y,z,ux,uy,uz)}}}
      box 为 None 的节点（如三通）本身不是障碍，只提供端口；其实体用 fixed_routes 中的短管表示。
    nets: [{"id", "terms": [(dev, port)], 可选 "D_mm"、"rho_mm"、"lead_mm"、"zc_max_mm"}]；单位 mm。
      D_mm 管外径（缺省 D_default_mm）；rho_mm 弯曲半径（缺省 c_rho × D）；
      lead_mm 端口处直颈 + 变径所需直管（与 ℓ_min 取大）；zc_max_mm 管中心线最高标高（缺省只受 z_max_mm 限制）；
      port_straight_mm {端口名: 从该端口到第一个弯头所需直管}（缺省 rho + lmin；两端直颈不同时分别给出）。
    fixed_routes: {id: {"points": [[x,y,z], ...], "D_mm"}}：不参与布管的已有管道（锁定管、局部优化范围外的管、
      三通内部短管），作为硬障碍，并参与管—管净距校验。允许斜段（按包围盒保守处理）。"""

    def __init__(self, devices, nets, rp, weights, scale, fixed_routes=None):
        check_params(rp, weights)
        self.dev, self.nets, self.rp, self.w = devices, nets, rp, weights
        self.fixed_routes = fixed_routes or {}
        # “管件式”端口：端口外已有管件（如斜支口伸出段末端的 45° 弯），出入时按弯头计直管长度
        self.fitting_ports = {(did, pn) for did, d in devices.items() for pn in d.get("fitting_ports", ())}
        self.lmin = scale["l_min_mm"]
        self.scale = scale                                      # A0 (mm²)、L0 (mm)、B0、C0、kappa
        D0 = rp["D_default_mm"]
        self.np = {}                                            # 每根管的参数
        for n in nets:
            D = n.get("D_mm", D0)
            zc = min(rp["z_max_mm"] - D / 2, n.get("zc_max_mm", math.inf))
            self.np[n["id"]] = {"D": D, "rho": n.get("rho_mm", rp["c_rho"] * D),
                                "lmin": max(self.lmin, n.get("lead_mm", 0.0)), "zc_max": zc,
                                "port_straight": {(dev, pn): n["port_straight_mm"][pn]
                                                  for dev, pn in n["terms"] if pn in n.get("port_straight_mm", {})}}
        # 包络管径：障碍膨胀、占用晕、网格线都按最大管径取，保守
        self.D = max([D0] + [v["D"] for v in self.np.values()] + [f["D_mm"] for f in self.fixed_routes.values()])
        self.rho = rp["c_rho"] * D0
        self.r_ep = rp["delta_ep_mm"] + self.D / 2
        self.r_pp = self.D + rp["delta_pp_mm"]
        self.boxes = []                                         # (x0,y0,z0,x1,y1,z1, r, owner)
        zh = rp["service_zone_height_mm"]
        for did, d in devices.items():
            if d["box"] is not None:
                self.boxes.append((*d["box"], self.r_ep, did))
            for zx0, zy0, zx1, zy1 in d["zones"]:
                self.boxes.append((zx0, zy0, 0, zx1, zy1, zh, self.D / 2, None))

    def net_param(self, nid):
        return self.np[nid]

    def port(self, dev, pn):
        return self.dev[dev]["ports"][pn]


# ============================================================ 网格
class Grid:
    def __init__(self, sc):
        rp, D = sc.rp, sc.D
        p, mg = rp["pitch_mm"], rp["margin_mm"]
        pts = [c for d in sc.dev.values() for c in ([d["box"][:3], d["box"][3:]] if d["box"] is not None else [])]
        pts += [v[:3] for d in sc.dev.values() for v in d["ports"].values()]
        pts += [q for f in sc.fixed_routes.values() for q in f["points"]]
        xs0, ys0 = min(q[0] for q in pts) - mg, min(q[1] for q in pts) - mg
        xs1, ys1 = max(q[0] for q in pts) + mg, max(q[1] for q in pts) + mg
        port_z = [v[2] for d in sc.dev.values() for v in d["ports"].values()]
        zlo, zhi = min([D / 2] + port_z), rp["z_max_mm"] - D / 2
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
        side = rp["port_side_lines"]                                    # 端口两侧加密网格线（规模大时关掉）
        for d in sc.dev.values():
            for x, y, z, *_ in d["ports"].values():
                for a, v in enumerate((x, y, z)):
                    cs[a].update((v, v - sc.r_pp, v + sc.r_pp) if side else (v,))
        self.c = [sorted(v for v in cs[a] if lim[a][0] - 1e-9 <= v <= lim[a][1] + 1e-9) for a in range(3)]
        self.carr = [np.asarray(v, dtype=float) for v in self.c]
        self.n = [len(v) for v in self.c]
        self.sc = sc
        nx_, ny_, nz_ = self.n
        self.size = nx_ * ny_ * nz_
        blocked = np.zeros(self.n, dtype=bool)
        eblocked = [np.zeros(self.n, dtype=bool) for _ in range(3)]      # 边 (i→i+1) 沿各轴
        for b in sc.boxes:
            self._mark(b, blocked, eblocked)
        self.blocked = blocked.ravel().astype(np.uint8)
        self.eblocked = [e.ravel().astype(np.uint8) for e in eblocked]
        self.stride = (ny_ * nz_, nz_, 1)
        for f in sc.fixed_routes.values():                             # 固定管路：其占用晕设为硬障碍
            codes = halo_codes(self, sc, [{"points": [tuple(q) for q in f["points"]], "trim": f.get("trim", (False, False))}],
                               f["D_mm"])
            for c in codes.tolist():
                self.eblocked[c % 3][c // 3] = 1

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
    """端口接入段豁免：从端口沿法向走，直到走出所属设备（膨胀 δ_ep + D/2 后）的盒子，只豁免该设备本身。
    端口可以在设备盒内部（例如接口在设备顶面以下、法向朝上）。
    另返回 inside：端口到自身设备盒面（未膨胀）的距离；端口在盒外为 0。
    这段盒内直管不计入“转弯前所需直管”，保证第一个弯（连同弯头让位）在盒外。"""
    nodes, edges, inside = set(), set(), {}
    for dev, pn in terms:
        x, y, z, ux, uy, uz = sc.port(dev, pn)
        own = sc.dev[dev]["box"]
        if own is not None and all(own[k] - 1e-9 <= (x, y, z)[k] <= own[k + 3] + 1e-9 for k in range(3)):
            k = DIRS[DIR_OF[(ux, uy, uz)]][0]
            inside[(dev, pn)] = (own[k + 3] - (x, y, z)[k]) if (ux, uy, uz)[k] > 0 else ((x, y, z)[k] - own[k])
        x, y, z, ux, uy, uz = sc.port(dev, pn)
        n = G.node((x, y, z))
        d = DIR_OF[(ux, uy, uz)]
        dist = 0.0
        nodes.add(n)
        own = sc.dev[dev]["box"]
        r = sc.r_ep

        def inside_own(pt):
            return own is not None and all(own[k] - r - 1e-9 <= pt[k] <= own[k + 3] + r + 1e-9 for k in range(3))
        while dist < sc.r_ep - 1e-9 or inside_own(G.pt(n)):
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
    return nodes, edges, inside


def tree_heuristic_arr(G, segs, cL, base=None):
    """全部网格节点到树上最近直管段的曼哈顿距离 × cL（按节点编号展平的数组）。
    每段是轴对齐线段，距离 = 各轴到区间 [lo, hi] 的间隙之和，三轴可分离，用 numpy 广播一次算完。
    base 给出已算好的距离场（乘过 cL）时只按新增段更新（三通拆段不改变几何，可增量）。"""
    C = [np.asarray(G.c[a], dtype=float) for a in range(3)]
    best = None if base is None else base.reshape(G.n) / cL
    for p, q, _kp, _kq in segs:
        g = [np.maximum(0.0, np.maximum(min(p[a], q[a]) - C[a], C[a] - max(p[a], q[a]))) for a in range(3)]
        d = g[0][:, None, None] + g[1][None, :, None] + g[2][None, None, :]
        best = d if best is None else np.minimum(best, d)
    return best.ravel() * cL


def tree_heuristic(G, segs, cL):
    """同 tree_heuristic_arr，返回 Python 列表（astar_py 用）。"""
    return tree_heuristic_arr(G, segs, cL).tolist()


def astar(G, sc, start, target, ctx):
    """编译实现（astar_fast.search）。规则与 astar_py 相同，测试中逐例比对。
    start = (dev, port)；target = ("port", (dev, port)) 或 ("tree", tree)。返回 (折线点列表, 结束信息) 或 None。"""
    cL = sc.w["length"] / sc.scale["L0"]
    cB = sc.w["bends"] / sc.scale["B0"]
    cC = sc.w["height_changes"] / sc.scale["C0"]
    halo, hist = ctx["halo"], ctx["hist"]
    ha = halo if isinstance(halo, np.ndarray) else np.zeros(G.size * 3, dtype=np.int32)
    hi_ = hist if isinstance(hist, np.ndarray) else np.zeros(G.size * 3, dtype=np.float32)
    ex_nodes = np.array(sorted(ctx["ex_nodes"]), dtype=np.int64)
    ex_edges = np.array(sorted(n * 3 + a for n, a in ctx["ex_edges"]), dtype=np.int64)
    sx, sy, sz, sux, suy, suz = sc.port(*start)
    s0 = G.node((sx, sy, sz))
    d0 = DIR_OF[(sux, suy, suz)]
    empty_i = np.zeros(0, dtype=np.int64)
    empty_f = np.zeros(0, dtype=np.float64)
    if target[0] == "port":
        tx, ty, tz, tux, tuy, tuz = sc.port(*target[1])
        mode, t, t_dir = 0, G.node((tx, ty, tz)), DIR_OF[(-tux, -tuy, -tuz)]
        tpt = np.array([tx, ty, tz], dtype=np.float64)
        hl = empty_f
        tnode, ttype, tsa, tdA, tdB, tkA, tkB, tcor = (empty_i, empty_i, empty_i, empty_f, empty_f,
                                                       empty_i, empty_i, empty_i)
        goal_info = ("port", target[1])
    else:
        tree = target[1]
        mode, t, t_dir = 1, 0, 0
        tpt = np.zeros(3)
        hl = np.asarray(ctx.get("tree_h_arr") if ctx.get("tree_h_arr") is not None
                        else tree_heuristic_arr(G, tree["segs"], cL))
        tnode, ttype, tsa, tdA, tdB, tkA, tkB, tcor = tree_arrays(tree["nodes"])
        goal_info = ("tee", None)
    out = af.search(G.carr[0], G.carr[1], G.carr[2], G.n[0], G.n[1], G.n[2], G.blocked,
                    G.eblocked[0], G.eblocked[1], G.eblocked[2], ha, hi_, float(ctx["pres"]),
                    ex_nodes, ex_edges, float(ctx["rho"]), float(ctx["lmin"]), int(sc.rp["K"]), float(ctx["zc_max"]),
                    cL, cB, cC, float(sc.rp["astar_weight"]), int(sc.rp["max_expansions"]),
                    int(s0), int(d0), mode, int(t), int(t_dir), tpt, hl,
                    tnode, ttype, tsa, tdA, tdB, tkA, tkB, tcor,
                    0 if tuple(start) in sc.fitting_ports else 1,
                    _target_extra(sc, target, ctx),
                    float(ctx.get("self_clear", 0.0)), _start_run(ctx, start))
    nodes, dirs, par, gid, exp, exhausted = out
    ctx["stats"]["expansions"] += int(exp)
    if gid < 0:
        if exhausted:
            ctx["stats"]["exhausted"] += 1
        return None
    return _trace_arrays(G, nodes, dirs, par, int(gid)), goal_info


def _port_delta(ctx, term):
    """该端口所需直管相对缺省（rho + lmin）的差值。"""
    ps = ctx.get("port_straight", {}).get(tuple(term))
    return 0.0 if ps is None else float(ps) - float(ctx["rho"]) - float(ctx["lmin"])


def _start_run(ctx, start):
    """起点状态的“已走直管”：负值表示转弯前还要多走（盒内长度、端口所需直管的附加量）。"""
    return -float(ctx.get("inside", {}).get(tuple(start), 0.0)) - _port_delta(ctx, start)


def _target_extra(sc, target, ctx):
    """到达目标端口前所需直管的附加量：管件式端口按弯头计；端口在设备盒内时加上盒内长度；两端直颈不同时按本端。"""
    if target[0] != "port":
        return 0.0
    t = tuple(target[1])
    return ((float(ctx["rho"]) if t in sc.fitting_ports else 0.0) + float(ctx.get("inside", {}).get(t, 0.0))
            + _port_delta(ctx, t))


def tree_arrays(nodes):
    """树上可接三通的位置 → 数组（按节点号排序）。类型：0 直管段内部、1 弯头、2 不可接。"""
    ks = sorted(nodes)
    tnode = np.array(ks, dtype=np.int64)
    ttype = np.zeros(len(ks), dtype=np.int64)
    tsa = np.zeros(len(ks), dtype=np.int64)
    tdA = np.zeros(len(ks), dtype=np.float64)
    tdB = np.zeros(len(ks), dtype=np.float64)
    tkA = np.zeros(len(ks), dtype=np.int64)
    tkB = np.zeros(len(ks), dtype=np.int64)
    tcor = np.zeros(len(ks), dtype=np.int64)
    for i, k in enumerate(ks):
        info = nodes[k]
        if info[0] == "mid":
            _k, seg_axis, dA, kA, dB, kB = info
            ttype[i] = 0; tsa[i] = seg_axis; tdA[i] = dA; tdB[i] = dB
            tkA[i] = 0 if kA == "port" else 1
            tkB[i] = 0 if kB == "port" else 1
        elif info[0] == "corner":
            ttype[i] = 1
            tcor[i] = sum(1 << nd for nd in range(6) if DIRS[nd] in info[1])
        else:
            ttype[i] = 2
    return tnode, ttype, tsa, tdA, tdB, tkA, tkB, tcor


def astar_py(G, sc, start, target, ctx):
    """纯 Python 参考实现（测试中与 astar 比对；规则以本函数与文档为准）。
    start = (dev, port)；target = ("port", (dev, port)) 或 ("tree", tree)。返回 (折线点列表, 结束信息) 或 None。"""
    rho, lmin, K, zc_max = ctx["rho"], ctx["lmin"], sc.rp["K"], ctx["zc_max"]
    cap = 2 * rho + lmin                                           # 目标端要求更长时在下方放宽
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

        def h(p, d, n):
            """管长曼哈顿距离 + 至少还需的弯头数（可采纳）。"""
            p = G.pt(n)
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
        hl = ctx.get("tree_h") or tree_heuristic(G, tree["segs"], cL)
        hl = list(hl)

        def h(p, d, n):
            """到树上最近直管段的曼哈顿距离（可采纳）；按节点查表，表由 tree_heuristic 一次算出。"""
            return hl[n]

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
        heapq.heappush(heap, (g + (0 if goal else hw * h(None, d, n)), next(tie), len(states) - 1))

    push(s, d0, 0 if tuple(start) in sc.fitting_ports else 1, 0, 1 if DIRS[d0][0] != 2 else 0,
         _start_run(ctx, start), 0.0, -1, None)
    t_extra = _target_extra(sc, target, ctx)
    self_clear = ctx.get("self_clear", 0.0)
    cap = max(cap, rho + t_extra + lmin)
    exp = 0
    while heap:
        _, _, sid = heapq.heappop(heap)
        n, d, lp, vr, hh, run, g, parent, goal = states[sid]
        if goal is not None:
            pts_ = _trace(G, states, sid)
            if self_clear > 0 and _self_close_py(pts_, self_clear):
                continue                                           # 与编译实现相同：自己挨得太近的路径不接受
            ctx["stats"]["expansions"] += exp
            return pts_, goal
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
            if a == 2 and G.c[2][G.ijk(nb)[2]] > zc_max + 1e-9:
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
                    if nd == t_dir and nrun >= (0 if nlp else rho) + t_extra + lmin - 1e-9:
                        goal_info = ("port", target[1])
                    else:
                        continue
            elif nb in tree_nodes:
                info_ = tree_nodes[nb]
                run_ok = nrun >= (0 if nlp else rho) + rho + lmin - 1e-9
                if info_[0] == "mid":
                    _k, seg_axis, dA, kA, dB, kB = info_
                    ok = (run_ok and a != seg_axis
                          and dA >= (0 if kA == "port" else rho) + rho + lmin - 1e-9
                          and dB >= (0 if kB == "port" else rho) + rho + lmin - 1e-9)
                elif info_[0] == "corner":
                    ok = run_ok and DIRS[nd] in info_[1]
                else:
                    ok = False
                if not ok:
                    continue
                goal_info = ("tee", None)
            push(nb, nd, nlp, nvr, nhh, nrun, ng, sid, goal_info)
    ctx["stats"]["expansions"] += exp
    return None


def _trace_arrays(G, nodes, dirs, par, sid):
    """编译实现返回的状态数组 → 折线点。"""
    seq = []
    while sid >= 0:
        seq.append((int(nodes[sid]), int(dirs[sid])))
        sid = int(par[sid])
    seq.reverse()
    pts = [G.pt(seq[0][0])]
    for k in range(1, len(seq)):
        if seq[k][1] != seq[k - 1][1]:
            pts.append(G.pt(seq[k - 1][0]))                        # 在上一节点转弯
    pts.append(G.pt(seq[-1][0]))
    return _dedupe(pts)


def _self_close_py(pts, clear):
    segs = list(zip(pts, pts[1:]))
    for i in range(len(segs)):
        for j in range(i + 2, len(segs)):
            gap = max(max(min(segs[j][0][a], segs[j][1][a]) - max(segs[i][0][a], segs[i][1][a]),
                          min(segs[i][0][a], segs[i][1][a]) - max(segs[j][0][a], segs[j][1][a])) for a in range(3))
            if gap < clear - 1e-6:
                return True
    return False


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


def build_tree(G, branches, rho, lmin):
    """由已布支路生成可接三通的位置：
      {节点: ("mid", 段轴, 到端A距离, 端A类型, 到端B距离, 端B类型)}  直管段内部，支路须垂直接入
      {节点: ("corner", {(轴, 符号), ...})}                           弯头处，支路沿某条腿的延长线接入
      {节点: ("end",)}                                                 端口、已有三通：不能接
    另返回合法接入点个数。"""
    segs = tree_segments(branches)
    nodes, valid = {}, 0
    ends = {}
    for p, q, kp, kq in segs:
        a = axis_of(p, q)
        for v, other, k in ((p, q, kp), (q, p, kq)):
            ends.setdefault(v, []).append((a, 1 if other[a] > v[a] else -1, k))
    for p, q, kp, kq in segs:
        a = axis_of(p, q)
        lo, hi = sorted((p[a], q[a]))
        c = G.c[a]
        for i in range(bisect.bisect_left(c, lo - 1e-9), bisect.bisect_right(c, hi + 1e-9)):
            v = c[i]
            pt = list(p); pt[a] = v
            pt = tuple(pt)
            n = G.node(pt)
            if pt in ends or n in nodes:
                continue
            dp, dq = abs(v - p[a]), abs(v - q[a])
            nodes[n] = ("mid", a, dp, kp, dq, kq)
            if dp >= (0 if kp == "port" else rho) + rho + lmin - 1e-9 and \
                    dq >= (0 if kq == "port" else rho) + rho + lmin - 1e-9:
                valid += 1
    for v, lst in ends.items():
        n = G.node(v)
        if len(lst) == 2 and all(k == "elbow" for _, _, k in lst):
            # 支路沿腿 (轴, 指向腿的符号) 的延长线进入：移动方向 = (轴, 符号)
            nodes[n] = ("corner", {(ax, sg) for ax, sg, _ in lst})
            valid += 1
        else:
            nodes[n] = ("end",)
    pts = [v for v in ends]
    bbox = ([min(t[a] for t in pts) for a in range(3)], [max(t[a] for t in pts) for a in range(3)])
    return {"nodes": nodes, "bbox": bbox, "segs": segs, "valid": valid}


def tree_segments(branches, stats=None):
    """所有支路 → 直管段列表 (p, q, p 端类型, q 端类型)，三通处拆段。
    三通两种：接在直管段内部（支路须垂直）；接在弯头处（支路与一条腿共线，弯头改为三通，
    stats["elbow_tees"] 计数，校验时从弯头数中扣除）。"""
    segs = []
    for br in branches:
        pts = br["points"]
        nb = len(pts) - 1
        for i in range(nb):
            kp = "port" if i == 0 else "elbow"
            kq = ("port" if br["end"][0] == "port" else "tee") if i == nb - 1 else "elbow"
            segs.append((pts[i], pts[i + 1], kp, kq))
        if br["end"][0] != "tee":
            continue
        t, prev = pts[-1], pts[-2]
        ab = axis_of(prev, t)
        old = segs[:-nb]
        for k, (p, q, kp_, kq_) in enumerate(old):
            a = axis_of(p, q)
            if all(abs(t[c] - p[c]) < 1e-9 for c in range(3) if c != a) and \
                    min(p[a], q[a]) + 1e-9 < t[a] < max(p[a], q[a]) - 1e-9:
                if ab == a:
                    raise ValueError(f"三通点 {t}：支路与主管共线接入直管段内部")
                segs[k] = (p, t, kp_, "tee")
                segs.insert(k + 1, (t, q, "tee", kq_))
                break
        else:
            at = [k for k, (p, q, kp_, kq_) in enumerate(old)
                  if (p == t and kp_ == "elbow") or (q == t and kq_ == "elbow")]
            if len(at) != 2:
                raise ValueError(f"三通点 {t} 既不在已有直管段内部，也不是弯头")
            legs = [old[k] for k in at]
            body = [(axis_of(p, q), q if p == t else p) for p, q, _, _ in legs]
            # 支路须与某条腿共线且从腿的对侧进入
            if not any(ax == ab and (other[ax] - t[ax]) * (t[ax] - prev[ax]) > 0 for ax, other in body):
                raise ValueError(f"三通点 {t}：支路未沿弯头某条腿的延长线接入")
            for k in at:
                p, q, kp_, kq_ = segs[k]
                segs[k] = (p, q, "tee" if p == t else kp_, "tee" if q == t else kq_)
            if stats is not None:
                stats["elbow_tees"] = stats.get("elbow_tees", 0) + 1
    return segs


# ============================================================ 占用晕
def halo_codes(G, sc, branches, D=None):
    """占用晕：与管道中心线（轴对齐盒）各轴距离都 < r_pp 的全部边，返回边编码 节点×3+轴 的有序数组。
    条件在三个轴上可分离：垂直于边的轴看节点坐标，沿边的轴看边区间，各得一个布尔掩码，取笛卡尔积。"""
    r = (sc.D if D is None else D) / 2 + sc.D / 2 + sc.rp["delta_pp_mm"] - 1e-9   # 本管半径 + 最大管半径 + 净距
    C = G.carr
    ny, nz = G.n[1], G.n[2]
    parts = []
    for br in branches:
        for p, q in clash_segments(sc, br):
            lo = [min(p[a], q[a]) for a in range(3)]
            hi = [max(p[a], q[a]) for a in range(3)]
            pm = [(C[b] - hi[b] < r) & (lo[b] - C[b] < r) for b in range(3)]
            for a in range(3):
                em = np.zeros(G.n[a], dtype=bool)
                em[:-1] = (C[a][:-1] - hi[a] < r) & (lo[a] - C[a][1:] < r)
                m = list(pm)
                m[a] = em
                I, J, K = (np.flatnonzero(x) for x in m)
                if I.size and J.size and K.size:
                    ids = (I[:, None, None] * ny + J[None, :, None]) * nz + K[None, None, :]
                    parts.append(ids.ravel() * 3 + a)
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(parts))


def clash_segments(sc, br):
    """参与管—管冲突判断的管段：在“无盒节点”（三通等汇合点）的端口处各去掉一段 r_pp。
    几根管在同一个三通上汇合，口之间只隔几厘米，汇合处附近的相互接近由三通配件本身决定，不算冲突。"""
    segs = [(tuple(p), tuple(q)) for p, q in zip(br["points"], br["points"][1:])]
    trim = br.get("trim")
    if trim is None:
        start = sc.dev[br["start"][0]]["box"] is None if "start" in br else False
        end = br["end"][0] == "port" and sc.dev[br["end"][1][0]]["box"] is None if "end" in br else False
        trim = (start, end)
    L = sc.r_pp
    for k, flag in ((0, trim[0]), (-1, trim[1])):
        if not flag or not segs:
            continue
        p, q = segs[k] if k == 0 else segs[k][::-1]
        n = math.dist(p, q)
        if n <= L + 1e-9:
            segs.pop(k)
            continue
        t = L / n
        cut = tuple(p[i] + (q[i] - p[i]) * t for i in range(3))
        segs[k] = (cut, q) if k == 0 else (q, cut)
    return segs


def halo_edges(G, sc, branches):
    """同 halo_codes，返回 {(节点, 轴)}。"""
    return {(int(c) // 3, int(c) % 3) for c in halo_codes(G, sc, branches)}


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
def port_access_blocked(G, sc, terms, ex_nodes, ex_edges, lmin):
    """必要条件：每个端口正前方至少能直行 ℓ_min（本管的）。返回被堵的端口列表。"""
    bad = []
    for dev, pn in terms:
        x, y, z, ux, uy, uz = sc.port(dev, pn)
        n, d, dist = G.node((x, y, z)), DIR_OF[(ux, uy, uz)], 0.0
        while dist < lmin - 1e-9:
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
    npar = sc.net_param(net["id"])
    ex_nodes, ex_edges, inside = port_exemptions(G, sc, terms)
    bad = port_access_blocked(G, sc, terms, ex_nodes, ex_edges, npar["lmin"])
    if bad:
        return None, "端口正前方不足 ℓ_min：" + "、".join(f"{d}.{p}" for d, p in bad)
    ctx = {**ctx, "ex_nodes": ex_nodes, "ex_edges": ex_edges, "rho": npar["rho"], "lmin": npar["lmin"],
           "zc_max": npar["zc_max"], "self_clear": npar["D"] + sc.rp["delta_pp_mm"], "inside": inside,
           "port_straight": npar["port_straight"]}
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
    cL = sc.w["length"] / sc.scale["L0"]
    hl = None
    while rest:
        tree = build_tree(G, branches, npar["rho"], npar["lmin"])
        if tree["valid"] == 0:
            return None, "已布管道上没有可接三通的位置"
        # 下一个接入的端点：到已布树最近的那个（查表，与 A* 用的是同一张表；比按包围盒距离排更准）
        # 距离场只看几何：新增支路的折线段即可（三通拆段不改变几何）
        new_segs = tree["segs"] if hl is None else [(p_, q_, None, None) for p_, q_ in segments_of(branches[-1]["points"])]
        hl = tree_heuristic_arr(G, new_segs, cL, hl)
        rest.sort(key=lambda t: hl[G.node(P[t])])
        t = rest.pop(0)
        r = astar(G, sc, t, ("tree", tree), {**ctx, "tree_h_arr": hl})
        if r is None:
            return None, f"支路 {t[0]}.{t[1]} 未接上（或超过扩展上限）"
        branches.append({"start": t, "points": r[0], "end": ("tee", None)})
    return branches, None


# ============================================================ 协商布线
_W = {}


def _worker_init(devices, nets, rp, weights, scale, halo_name, hist_name, fixed_routes):
    from multiprocessing import shared_memory
    sc = Scene(devices, nets, rp, weights, scale, fixed_routes)
    _W["sc"], _W["G"] = sc, Grid(sc)
    _W["shm"] = (shared_memory.SharedMemory(name=halo_name), shared_memory.SharedMemory(name=hist_name))
    _W["halo"] = np.frombuffer(_W["shm"][0].buf, dtype=np.int32)
    _W["hist"] = np.frombuffer(_W["shm"][1].buf, dtype=np.float32)


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
    return (e, br, None, halo_codes(G, sc, br, sc.net_param(sc.nets[e]["id"])["D"]).tolist(),
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
    route_workers = 1 时串行（无并发冲突，逻辑相同）。
"""
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
        halo = np.frombuffer(shms[0].buf, dtype=np.int32)
        hist = np.frombuffer(shms[1].buf, dtype=np.float32)
        ex = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_worker_init,
                                 initargs=(sc.dev, sc.nets, rp, sc.w, sc.scale, shms[0].name, shms[1].name,
                                           sc.fixed_routes))
        list(ex.map(_worker_ready, range(workers * 4)))                 # 预先启动全部工作进程
    else:
        halo, hist = np.zeros(size, dtype=np.int32), np.zeros(size, dtype=np.float32)
    max_requeue = 2
    routes, halos, pedges, why, hard = {}, {}, {}, {}, {}
    ever = set()
    pres = rp["pres_fac_init"]
    history = []
    commits = []                                                        # 提交日志：各次提交的晕
    todo = sorted(range(len(sc.nets)), key=lambda e: -len(sc.nets[e]["terms"]))
    best_key, since_best = None, 0
    stuck = {}                                                          # 连续“搜到扩展上限、保留旧路径”的次数
    tried, done_clean = set(), None                                     # 协商中途试清理：已试过的冲突管网集合

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
        stuck[e] = stuck.get(e, 0) + 1 if (br is None and old[0] is not None and info["exhausted"]) else 0
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
            queue = [e for e in todo if e not in hard and stuck.get(e, 0) < rp["freeze_after_exhausted"]]
            if not queue:
                log("  待重布的管网都已冻结（多次搜到扩展上限），停止协商，转入清理")
                break
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
            if not retry and len(conflicted) <= rp["cleanup_trigger_nets"] and frozenset(conflicted) not in tried:
                tried.add(frozenset(conflicted))
                t_c = time.time()
                cand, rec = cleanup(G, sc, {sc.nets[e]["id"]: br for e, br in routes.items()}, log)
                ok_c = not clashes(sc, cand)
                history[-1]["cleanup_try"] = {"nets": len(conflicted), "ok": ok_c, "time_s": round(time.time() - t_c, 1)}
                log(f"  冲突管网 {len(conflicted)} 个，试清理 {time.time() - t_c:.1f} s：{'成功，结束' if ok_c else '未清干净，继续协商'}")
                if ok_c:
                    done_clean = (cand, rec, round(time.time() - t_c, 1))
                    break
            key = (len(retry), conflicts)
            if best_key is None or key < best_key:
                best_key, since_best = key, 0
            else:
                since_best += 1
                if since_best >= rp["stall_iters"]:
                    log(f"  连续 {since_best} 轮无改进，停止协商，转入清理")
                    break
            todo = sorted(conflicted | retry, key=lambda e: -len(sc.nets[e]["terms"]))
            pres *= rp["pres_fac_mult"]
    finally:
        if ex is not None:
            ex.shutdown(wait=True, cancel_futures=True)
            del halo, hist
            for sh in shms:
                sh.close(); sh.unlink()
    if done_clean is not None:
        out, records, tc = done_clean
    else:
        t_c = time.time()
        out, records = cleanup(G, sc, {sc.nets[e]["id"]: br for e, br in routes.items()}, log)
        tc = round(time.time() - t_c, 1)
    history[-1]["cleanup"] = {"records": records, "time_s": tc}
    return out, history, G


# ============================================================ 清理：其他管网固定为硬障碍，单独重布冲突管网
def clashes(sc, routes):
    """独立校验器报出的冲突：[(管网, 对方管网或 None, 原文)]。"""
    viol, _ = check_routes(sc, routes)
    out = []
    for v in viol:
        if "管–管净距不足" in v:
            a, b = v.split("：")[0].split(" 与 ")
            out.append((a, b, v))
        elif "净距不足" in v or "穿过" in v:
            out.append((v.split("：")[0], None, v))
    return out


def route_fixed(G, sc, routes, nid):
    """routes 中其他管网的占用晕设为硬障碍，不加拥堵代价，单独搜索 nid（扩展上限 cleanup_max_expansions）。
    返回 (支路或 None, 说明)。说明区分“搜索空间耗尽（当前网格与其他管道下确实无路）”与“超过扩展上限”。"""
    e = next(i for i, n in enumerate(sc.nets) if n["id"] == nid)
    codes = [halo_codes(G, sc, brs, sc.net_param(other)["D"]) for other, brs in routes.items() if other != nid]
    blocked = [(c // 3, c % 3) for c in (np.unique(np.concatenate(codes)).tolist() if codes else [])]
    saved = [(n, a, G.eblocked[a][n]) for n, a in blocked]
    for n, a in blocked:
        G.eblocked[a][n] = True
    stats = {"expansions": 0, "exhausted": 0}
    rp = {**sc.rp, "max_expansions": sc.rp["cleanup_max_expansions"]}
    sc_c = copy.copy(sc)
    sc_c.rp = rp
    try:
        br, reason = route_net(G, sc_c, sc.nets[e], {"halo": _Zero(), "hist": _Zero(), "pres": 0.0, "stats": stats})
    finally:
        for n, a, v in saved:
            G.eblocked[a][n] = v
    if br is not None:
        return br, f"找到（扩展 {stats['expansions']}）"
    kind = "超过扩展上限，不能下结论" if stats["exhausted"] else "搜索空间耗尽：确实无路"
    return None, f"{kind}（{reason}；扩展 {stats['expansions']}）"


def cleanup(G, sc, routes, log=print):
    """逐个处理冲突。管–管冲突 X 与 Y 依次尝试：只重布 X → 只重布 Y → 先 X 后 Y → 先 Y 后 X；
    接受第一个“该冲突消失且总冲突数不增加”的方案（均以 check_routes 复核）。返回 (路线, 记录)。"""
    records = []
    for a, b, text in clashes(sc, routes):
        now = clashes(sc, routes)
        if not any((x, y) == (a, b) for x, y, _ in now):
            records.append({"clash": text, "result": "已顺带消除"})
            continue
        movable = [x for x in (a, b) if x and x in sc.np]
        if not movable:
            records.append({"clash": text, "result": "未消除（两侧都是固定管路）"})
            continue
        a0, b0 = a, b                                                  # 原冲突对（用于判断是否消除）
        a, b = movable[0], (movable[1] if len(movable) > 1 else None)
        plans = [[a]] + ([[b], [a, b], [b, a]] if b else [])
        attempts, fixed = [], False
        for order in plans:
            trial = {k: v for k, v in routes.items() if k not in order}
            steps = []
            for nid in order:
                br, msg = route_fixed(G, sc, trial, nid)
                steps.append({"net": nid, "result": msg})
                if br is None:
                    break
                trial[nid] = br
            rec = {"reroute": order, "steps": steps}
            attempts.append(rec)
            if len(steps) < len(order) or steps[-1]["result"].startswith(("超过", "搜索空间")):
                continue
            after = clashes(sc, trial)
            rec["clashes_after"] = len(after)
            still = any((x, y) in ((a0, b0), (b0, a0)) if b0 else (x == a0 and y is None) for x, y, _ in after)
            if not still and len(after) <= len(now):
                routes, fixed = trial, True
                break
        records.append({"clash": text, "result": "已消除" if fixed else "未消除", "attempts": attempts})
        log(f"  清理 {text}：{'已消除' if fixed else '未消除'}（" +
            "；".join("重布 " + "→".join(r["reroute"]) + "：" + "，".join(s_["result"] for s_ in r["steps"])
                     for r in attempts) + "）")
    return routes, records


# ============================================================ 独立校验器
def check_routes(sc, routes):
    """只依据折线几何检查规则并计算指标。返回 (违规列表, 指标)。"""
    K = sc.rp["K"]
    eps = sc.rp["eps_z_mm"]
    viol = []
    per_net, all_boxes = {}, []
    net_of = {n["id"]: n for n in sc.nets}
    for nid, n in net_of.items():
        if nid not in routes:
            viol.append(f"{nid}：未布通")
    for nid, brs in routes.items():
        npar = sc.net_param(nid)
        D, rho, lmin = npar["D"], npar["rho"], npar["lmin"]
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
            sg = segments_of(pts)                                       # 自相交：不相邻管段之间也要留净距
            for i in range(len(sg)):
                for j in range(i + 2, len(sg)):
                    if _gap(_box(*sg[i], D / 2), _box(*sg[j], D / 2)) < sc.rp["delta_pp_mm"] - 1e-6:
                        viol.append(f"{nid} 支路 {bi}：管段 {i} 与管段 {j} 自相交或净距不足")
            layers += nl
        if seen != terms:
            viol.append(f"{nid}：未连接全部端点")
        tstats = {}
        try:
            segs = tree_segments(brs, tstats)
        except ValueError as ex:
            viol.append(f"{nid}：{ex}")
            segs = []
        bends -= tstats.get("elbow_tees", 0)                             # 弯头处接三通：不再是 90° 弯头
        fit_pts = {tuple(sc.port(*fp)[:3]) for fp in sc.fitting_ports}
        inside_at = {}                                                  # 端口在设备盒内：盒内那段不算“转弯前直管”
        port_delta = {tuple(sc.port(*term)[:3]): ps - rho - lmin for term, ps in npar["port_straight"].items()}
        for dev, pn in net_of[nid]["terms"]:
            x, y, z, ux, uy, uz = sc.port(dev, pn)
            own = sc.dev[dev]["box"]
            if own is not None and all(own[k] - 1e-9 <= (x, y, z)[k] <= own[k + 3] + 1e-9 for k in range(3)):
                k = DIRS[DIR_OF[(ux, uy, uz)]][0]
                inside_at[(x, y, z)] = (own[k + 3] - (x, y, z)[k]) if (ux, uy, uz)[k] > 0 else ((x, y, z)[k] - own[k])
        segs = [(p, q, "elbow" if kp == "port" and tuple(p) in fit_pts else kp,
                 "elbow" if kq == "port" and tuple(q) in fit_pts else kq) for p, q, kp, kq in segs]
        for p, q, kp, kq in segs:
            need = ((0 if kp == "port" else rho) + (0 if kq == "port" else rho) + lmin
                    + inside_at.get(tuple(p), 0.0) + inside_at.get(tuple(q), 0.0)
                    + (port_delta.get(tuple(p), 0.0) if kp == "port" else 0.0)
                    + (port_delta.get(tuple(q), 0.0) if kq == "port" else 0.0))
            ln = abs(q[axis_of(p, q)] - p[axis_of(p, q)])
            if ln < need - 1e-6:
                viol.append(f"{nid}：直管段 {p}→{q} 长 {ln:.0f} < {need:.0f}（{kp}–{kq}）")
        # 碰撞
        for br in brs:
            dev0 = br["start"][0]
            dev1 = br["end"][1][0] if br["end"][0] == "port" else None
            sg = segments_of(br["points"])
            for k, (p, q) in enumerate(sg):
                a = axis_of(p, q)
                s = 1 if q[a] > p[a] else -1
                pp, qq = list(p), list(q)
                trim_dev = set()
                trim = sc.rp["delta_ep_mm"] + D / 2
                if k == 0:
                    pp[a] += s * _exit_len(sc, dev0, p, a, s, trim); trim_dev.add(dev0)
                if k == len(sg) - 1 and dev1:
                    qq[a] -= s * _exit_len(sc, dev1, q, a, -s, trim); trim_dev.add(dev1)
                full = _box(p, q, D / 2)
                chk = _box(tuple(pp), tuple(qq), D / 2) if (s * (qq[a] - pp[a]) > 0) else None
                for did, d in sc.dev.items():
                    b = d["box"]
                    tgt = chk if did in trim_dev else full
                    if b is not None and tgt is not None and _gap(tgt, b) < sc.rp["delta_ep_mm"] - 1e-6:
                        viol.append(f"{nid}：管段 {p}→{q} 与设备 {did} 净距不足")
                    for zx0, zy0, zx1, zy1 in d["zones"]:
                        zb = (zx0, zy0, 0, zx1, zy1, sc.rp["service_zone_height_mm"])
                        if _gap(full, zb) < 0 - 1e-6:
                            viol.append(f"{nid}：管段 {p}→{q} 穿过 {did} 的检修区")
                if full[5] > sc.rp["z_max_mm"] + 1e-6 or max(p[2], q[2]) > npar["zc_max"] + 1e-6:
                    viol.append(f"{nid}：管段 {p}→{q} 超出高度范围")
        all_boxes.append((nid, [_box(p, q, D / 2) for br in brs for p, q in clash_segments(sc, br)]))
        per_net[nid] = {"L_mm": L, "bends": bends, "height_changes": layers, "branches": len(brs)}
    for fid, f in sc.fixed_routes.items():                             # 固定管路只参与净距，不计指标
        pts = [tuple(q) for q in f["points"]]
        all_boxes.append((fid, [_box(p, q, f["D_mm"] / 2) for p, q in
                                clash_segments(sc, {"points": pts, "trim": f.get("trim", (False, False))})]))
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
    dboxes = [d["box"] for d in sc.dev.values() if d["box"] is not None]
    xs = [b[0] for b in dboxes] + [b[0] for _, bs in all_boxes for b in bs]
    ys = [b[1] for b in dboxes] + [b[1] for _, bs in all_boxes for b in bs]
    xe = [b[3] for b in dboxes] + [b[3] for _, bs in all_boxes for b in bs]
    ye = [b[4] for b in dboxes] + [b[4] for _, bs in all_boxes for b in bs]
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


def _exit_len(sc, dev, p, a, s, trim):
    """端口段的校验豁免长度：走出所属设备（膨胀 trim 后）的盒子为止，至少 trim。"""
    b = sc.dev[dev]["box"]
    if b is None:
        return trim
    edge = (b[a + 3] + trim - p[a]) if s > 0 else (p[a] - (b[a] - trim))
    return max(trim, edge)


def _box(p, q, r):
    return (min(p[0], q[0]) - r, min(p[1], q[1]) - r, min(p[2], q[2]) - r,
            max(p[0], q[0]) + r, max(p[1], q[1]) + r, max(p[2], q[2]) + r)


def _gap(a, b):
    """两个轴对齐盒的间隙：各轴间隙的最大值；重叠时为负。"""
    return max(max(b[k] - a[k + 3], a[k] - b[k + 3]) for k in range(3))
