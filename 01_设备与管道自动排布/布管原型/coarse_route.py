"""粗网格快速布管：秒级判断一个设备摆放能否布下管道、冲突在哪。只用于判断和搜索引导，
最终方案仍由 routing.py 精细布管 + check_routes 校验。

模型（比精细布管粗、偏保守或偏乐观之处见“简化”）：
  - 平面按 coarse_cell_mm 分格，高度按管层间距 D + δ_pp 分层；每个格点最多放 1 根管。
  - 格点中心落在“设备盒按 δ_ep + D/2 膨胀、且低于设备顶 + δ_ep + D/2”的范围内不可走；
    落在检修区（按 D/2 膨胀、高度 service_zone_height_mm）内不可走。
  - 端口：沿法向伸出 ℓ_min + ρ，伸出段经过的格点预留给本管网；伸出段被挡 → 端口被堵。
    伸出段末端所在格点、离端口高度最近的层为该端点的接入点。
  - 管网：从已连通的树多源 Dijkstra，依次接上最近的剩余端点（scipy.sparse.csgraph，C 实现）。
  - 拥堵：PathFinder 式协商，格点代价 = 基础长度 × (1 + 历史) × (1 + pres × 其他管网占用)；
    竖向边另加 coarse_vertical_penalty_mm，粗略代表高度变化代价。
简化：不计弯头与最小直管长度；不查高度变化上限 K；同格点只看占用数，不看管径。

运行：见 粗网格校准.py
"""
import math
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

REQUIRED = ["coarse_cell_mm", "coarse_iters", "coarse_vertical_penalty_mm"]


class CoarseGrid:
    def __init__(self, sc):
        rp = sc.rp
        miss = [k for k in REQUIRED if k not in rp]
        if miss:
            raise KeyError(f"粗网格布管缺少必填参数：{miss}")
        self.sc = sc
        c = rp["coarse_cell_mm"]
        D = sc.D
        mg = rp["margin_mm"]
        x0 = min(d["box"][0] for d in sc.dev.values()) - mg
        y0 = min(d["box"][1] for d in sc.dev.values()) - mg
        x1 = max(d["box"][3] for d in sc.dev.values()) + mg
        y1 = max(d["box"][4] for d in sc.dev.values()) + mg
        self.c, self.x0, self.y0 = c, x0, y0
        self.nx, self.ny = int(math.ceil((x1 - x0) / c)), int(math.ceil((y1 - y0) / c))
        pitch = D + rp["delta_pp_mm"]
        self.zs = np.arange(D / 2, rp["z_max_mm"] - D / 2 + 1e-9, pitch)
        self.nz = len(self.zs)
        self.xc = x0 + (np.arange(self.nx) + 0.5) * c
        self.yc = y0 + (np.arange(self.ny) + 0.5) * c
        blocked = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        X, Y = self.xc[:, None], self.yc[None, :]
        for d in sc.dev.values():
            bx0, by0, _, bx1, by1, h = d["box"]
            r = sc.r_ep
            plan = (X > bx0 - r) & (X < bx1 + r) & (Y > by0 - r) & (Y < by1 + r)
            blocked[plan[:, :, None] & (self.zs[None, None, :] < h + sc.r_ep)] = True
            zh = rp["service_zone_height_mm"]
            for zx0, zy0, zx1, zy1 in d["zones"]:
                plan = (X > zx0 - D / 2) & (X < zx1 + D / 2) & (Y > zy0 - D / 2) & (Y < zy1 + D / 2)
                blocked[plan[:, :, None] & (self.zs[None, None, :] < zh + D / 2)] = True
        self.blocked = blocked
        self._build_graph(rp["coarse_vertical_penalty_mm"], pitch)

    def idx(self, i, j, k):
        return (i * self.ny + j) * self.nz + k

    def cell_of(self, x, y, z):
        i = min(self.nx - 1, max(0, int((x - self.x0) // self.c)))
        j = min(self.ny - 1, max(0, int((y - self.y0) // self.c)))
        k = int(np.argmin(np.abs(self.zs - z)))
        return i, j, k

    def _build_graph(self, vpen, pitch):
        nx, ny, nz = self.nx, self.ny, self.nz
        N = nx * ny * nz
        ids = np.arange(N).reshape(nx, ny, nz)
        rows, cols, w = [], [], []
        for axis, step in ((0, self.c), (1, self.c), (2, pitch + vpen)):
            a = [slice(None)] * 3; b = [slice(None)] * 3
            a[axis] = slice(0, -1); b[axis] = slice(1, None)
            u, v = ids[tuple(a)].ravel(), ids[tuple(b)].ravel()
            rows += [u, v]; cols += [v, u]; w += [np.full(u.size, step), np.full(u.size, step)]
        rows, cols, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w).astype(float)
        self.N = N
        self.base = csr_matrix((w, (rows, cols)), shape=(N, N))
        self.base.sum_duplicates()
        self.col = self.base.indices                          # 每条边的终点 = 进入该格点的代价
        self.base_data = self.base.data.copy()
        self.free = ~self.blocked.ravel()


def port_stub(G, sc, dev, pn):
    """端口伸出段（长 ℓ_min + ρ）按真实几何判断是否被挡（与精细布管同口径）：
    管段方盒（半宽 D/2）与其他设备的间隙须 ≥ δ_ep，且不得进入任何检修区。
    返回伸出段经过的格点（归本管网专用，末个为接入点），被挡返回 None。"""
    x, y, z, ux, uy, _uz = sc.port(dev, pn)
    L = sc.lmin + sc.rho
    h = sc.D / 2
    ex_, ey_ = x + ux * L, y + uy * L
    seg = (min(x, ex_) - h, min(y, ey_) - h, z - h, max(x, ex_) + h, max(y, ey_) + h, z + h)
    zh = sc.rp["service_zone_height_mm"]
    for did, d in sc.dev.items():
        if did != dev and _gap(seg, d["box"]) < sc.rp["delta_ep_mm"]:
            return None
        for zx0, zy0, zx1, zy1 in d["zones"]:
            if _gap(seg, (zx0, zy0, 0, zx1, zy1, zh)) < 0:
                return None
    n = max(1, int(math.ceil(L / (G.c / 2))))
    cells = []
    for t in range(1, n + 1):
        s_ = L * t / n
        i, j, k = G.cell_of(x + ux * s_, y + uy * s_, z)
        cid = G.idx(i, j, k)
        if cid not in cells:
            cells.append(cid)
    return cells


def _gap(a, b):
    """两个轴对齐盒的间隙（各轴间隙取最大；重叠为负）。"""
    return max(max(b[k] - a[k + 3], a[k] - b[k + 3]) for k in range(3))


def coarse_route(sc, log=None):
    """返回 {"ok", "failed": {管网: 原因}, "overflow_cells", "overflow_nets", "length_m", "iters", "time_s"}。"""
    t0 = time.time()
    rp = sc.rp
    G = CoarseGrid(sc)
    nets = sc.nets
    occ = np.zeros(G.N, dtype=np.int32)
    hist = np.zeros(G.N)
    paths, failed, stubs = {}, {}, {}
    for e, net in enumerate(nets):
        cells = []
        for dev, pn in net["terms"]:
            st = port_stub(G, sc, dev, pn)
            if st is None:
                failed[net["id"]] = f"端口被堵：{dev}.{pn}"
                break
            cells.append(st)
        else:
            stubs[e] = cells
    pres = rp["pres_fac_init"]
    todo = [e for e in range(len(nets)) if e in stubs]
    todo.sort(key=lambda e: -len(nets[e]["terms"]))
    history = []
    for it in range(rp["coarse_iters"]):
        for e in todo:
            if e in paths:
                np.subtract.at(occ, list(paths[e]), 1)
                del paths[e]
            own_stub = {c for st in stubs[e] for c in st}
            ok_nodes = G.free.copy()
            ok_nodes[list(own_stub)] = True
            cost_node = (1 + hist) * (1 + pres * occ)
            data = G.base_data * cost_node[G.col]
            data[~ok_nodes[G.col]] = np.inf
            graph = csr_matrix((data, G.col, G.base.indptr), shape=(G.N, G.N))
            terms = [st[-1] for st in stubs[e]]
            tree = set(stubs[e][0])
            rest = list(range(1, len(terms)))
            used = set(own_stub)
            good = True
            while rest:
                dist, pred, _src = dijkstra(graph, indices=list(tree), return_predecessors=True, min_only=True)
                k = min(rest, key=lambda r: dist[terms[r]])
                if not np.isfinite(dist[terms[k]]):
                    good = False
                    break
                v = terms[k]
                while v not in tree and v >= 0:
                    used.add(v); tree.add(v)
                    v = pred[v]
                tree.update(stubs[e][k])
                used.update(stubs[e][k])
                rest.remove(k)
            if not good:
                failed[nets[e]["id"]] = "粗网格上不连通"
                continue
            failed.pop(nets[e]["id"], None)
            paths[e] = used
            np.add.at(occ, list(used), 1)
        over = occ > 1
        n_over = int(over.sum())
        over_nets = {e for e, p in paths.items() if any(over[c] for c in p)}
        history.append(n_over)
        if log:
            log(f"    粗网格第 {it + 1} 轮：超容格点 {n_over}（{len(over_nets)} 个管网），未布通 {len(failed)}")
        if n_over == 0:
            break
        hist[over] += rp["hist_fac"]
        pres *= rp["pres_fac_mult"]
        todo = sorted(over_nets, key=lambda e: -len(nets[e]["terms"]))
    length = sum(len(p) for p in paths.values()) * G.c / 1000
    over = occ > 1
    return {"ok": not failed and int(over.sum()) == 0, "failed": failed, "overflow_cells": int(over.sum()),
            "overflow_nets": sorted(nets[e]["id"] for e, p in paths.items() if any(over[c] for c in p)),
            "length_m_rough": round(length, 1), "iters": len(history), "history": history,
            "grid": [G.nx, G.ny, G.nz], "time_s": round(time.time() - t0, 2),
            "paths": {nets[e]["id"]: sorted(int(c) for c in p) for e, p in paths.items()},
            "spec": {"x0": G.x0, "y0": G.y0, "c": G.c, "nx": G.nx, "ny": G.ny, "nz": G.nz,
                     "zs": [float(z) for z in G.zs]}}


def dilate(spec, cells, r):
    """走廊：粗网格路径格点向外扩 r 格（平面与层都扩）。"""
    nx, ny, nz = spec["nx"], spec["ny"], spec["nz"]
    out = set()
    for cid in cells:
        i, j, k = cid // (ny * nz), (cid // nz) % ny, cid % nz
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                for dk in range(-r, r + 1):
                    a, b, c = i + di, j + dj, k + dk
                    if 0 <= a < nx and 0 <= b < ny and 0 <= c < nz:
                        out.add((a * ny + b) * nz + c)
    return out
