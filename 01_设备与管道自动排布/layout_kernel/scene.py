"""三维场景任务：节点（设备 / 三通）直接给世界坐标下的包围盒与端口，每根管给两端端口。

这是给已有项目接入用的入口（例如 拆件做网页 的系统管网）：对方已经有设备实例、姿态与现有管路，
不需要设备库和摆放，只需要在当前（或候选）姿态下重新布管、比较真实指标。

输入（米制、Y 向上，与网页 networkMILPInput 相同）：
  nodes:  [{id, box: bool, orientations: [{ports: {key: {position, normal}}, box: {min, max} | null,
            spools: [[p, q], ...]}]}]          —— 本函数只用 orientations[0]（当前姿态）
  routes: [{id, code, points, segments: [{r}], from: {key}, to: {key}, leadA, leadB, fixed, low}]
settings（全部必填）：
  routing:   布管参数（毫米，同 routing.REQUIRED；D_default_mm 仅作缺省，实际按每根管的外径）
  weights / scale：目标权重与归一化尺度（同 routing.Scene）
  low_zc_max_mm：low 管中心线最高标高（网页规则：机房低位管道须低于 3.2 m）
  oblique_stub_mm：斜向端口在基线里找不到斜段时，沿法向伸出的短管长度
  pipe_rules：对方的直管规则，用来推算每根管的 rho / lead（见 _straight_rules）：
    {"trim_ratio": 弯头最多占相邻直管的比例, "radius_margin_mm": 有效弯曲半径须超过管半径的量,
     "port_margin_mm": 端口直颈之外的余量, "safety": 放大系数,
     "rule_D_mm": 对方直管规则所用的管径（网页校验一律按 0.22 m 代理管计算，不按实际外径）,
     "rule_R_mm": 对方的名义弯曲半径（网页 0.38 m）}
  管间净距仍按每根管的实际外径（比对方的代理管更保守）。
返回：{"ok", "violations", "metrics", "routes": [{id, code, points}]（米制、Y 向上，含两端端口）, "timing"}
"""
import math
import sys
import time
from pathlib import Path

_R = str(Path(__file__).resolve().parent.parent / "布管原型")
if _R not in sys.path:
    sys.path.insert(0, _R)
import routing as rt  # noqa: E402  只依赖布管模块（numpy + numba），不加载摆放与分块
SETTINGS_REQUIRED = ["routing", "weights", "scale", "low_zc_max_mm", "oblique_stub_mm", "pipe_rules"]
SERIAL_NETS = 8                                                 # 每个并行进程至少分到的管数
RULES_REQUIRED = ["trim_ratio", "radius_margin_mm", "port_margin_mm", "safety", "rule_D_mm", "rule_R_mm"]


def _straight_rules(D_pipe, leads, rules, delta_pp):
    """把对方的直管规则换算成内核的 (rho, lmin)。内核要求：两弯之间直管 ≥ 2·rho + lmin，端口到弯 ≥ rho + lmin。
    对方（网页 pipeSections / dimensionalSections）：弯头让位 trim = min(R, trim_ratio·相邻直管)，有效半径 trim 须
    > D/2 + margin；端口处去掉让位后的直管 ≥ 直颈 + 变径（lead）+ port_margin。于是
      两弯之间最短  S_ee = (D/2 + margin) / trim_ratio
      端口到弯最短  S_pe = lead + R + port_margin          （若此时 trim_ratio·S_pe ≥ R）
                          (lead + port_margin)/(1 − trim_ratio)  （否则）
    U 形回弯两边是否相碰不在这里加严，由自身净距检查处理（routing 的 self_skip_mm 与对方 selfCollision 一致）。"""
    k, t, D, R = rules["safety"], rules["trim_ratio"], rules["rule_D_mm"], rules["rule_R_mm"]
    s_rad = (D / 2 + rules["radius_margin_mm"]) / t                  # 每个弯头两侧直管都要满足（含端口后的第一个弯）
    s_ee = k * s_rad

    def s_pe(lead):
        v = lead + R + rules["port_margin_mm"]
        if t * v < R:
            v = (lead + rules["port_margin_mm"]) / (1 - t)
        return round(k * max(v, s_rad), 1)
    rho = s_ee / 2
    return round(rho, 1), 0.0, [s_pe(x) for x in leads]


def to_k(p):
    """网页坐标（m，Y 上）→ 内核坐标（mm，Z 上），取整到 0.1 mm。"""
    return (round(p[0] * 1000, 1), round(p[2] * 1000, 1), round(p[1] * 1000, 1))


def to_w(p):
    return [p[0] / 1000, p[2] / 1000, p[1] / 1000]


def _axis_dir(v, where):
    """法向量 → 轴向单位向量（内核坐标）；斜向返回 None。"""
    k = (v[0], v[2], v[1])
    big = [i for i in range(3) if abs(k[i]) > 1e-6]
    if len(big) != 1:
        return None
    i = big[0]
    if abs(abs(k[i]) - 1) > 1e-6:
        raise ValueError(f"{where}：法向量不是单位向量 {v}")
    out = [0, 0, 0]
    out[i] = 1 if k[i] > 0 else -1
    return tuple(out)


def _box_k(b):
    lo, hi = to_k(b["min"]), to_k(b["max"])
    return tuple(min(lo[i], hi[i]) for i in range(3)) + tuple(max(lo[i], hi[i]) for i in range(3))


def _oblique_stub(port_w, normal_w, baseline_points, stub_mm):
    """斜向端口：沿法向伸出一段斜直管，之后接正交布管。
    优先沿用基线里的斜段（长度与之后的正交方向都照旧）；没有则伸出 stub_mm，之后方向取法向的最大水平分量。"""
    P = to_k(port_w)
    n = (normal_w[0], normal_w[2], normal_w[1])
    if baseline_points and len(baseline_points) >= 3:
        a, b, c = (to_k(q) for q in baseline_points[:3])
        if a == P and sum(1 for i in range(3) if abs(a[i] - b[i]) > 0.5) > 1:
            d = [c[i] - b[i] for i in range(3)]
            i = max(range(3), key=lambda k: abs(d[k]))
            out = [0, 0, 0]
            out[i] = 1 if d[i] > 0 else -1
            return b, tuple(out), list(baseline_points[1])
    end = tuple(round(P[i] + n[i] * stub_mm, 1) for i in range(3))
    i = max((0, 1), key=lambda k: abs(n[k]))
    out = [0, 0, 0]
    out[i] = 1 if n[i] > 0 else -1
    return end, tuple(out), to_w(end)


def build_scene(inp, settings, fixed_ids=(), deltas=None):
    """网页优化输入 → routing.Scene。返回 (Scene, 元数据)。fixed_ids 中的管保持原路径，作为障碍。
    deltas = {节点 id: [dX, dY, dZ]}（网页坐标，米、Y 向上）：该节点的端口与包围盒整体平移。平移过的节点，
    其斜向端口不再沿用基线里的斜段，按法向重新伸出。"""
    deltas = deltas or {}
    miss = [k for k in SETTINGS_REQUIRED if k not in settings]
    if miss:
        raise KeyError(f"场景布管缺少设置：{miss}（不设默认值）")
    rp = dict(settings["routing"])
    miss = [k for k in RULES_REQUIRED if k not in settings["pipe_rules"]]
    if miss:
        raise KeyError(f"pipe_rules 缺少：{miss}")
    port_owner, port_w = {}, {}
    devices, fixed = {}, {}
    for n in inp["nodes"]:
        o = n["orientations"][0]
        d = deltas.get(n["id"])
        dw = tuple(d) if d else (0.0, 0.0, 0.0)
        box = None
        if o.get("box"):
            box = _box_k({"min": [o["box"]["min"][i] + dw[i] for i in range(3)],
                          "max": [o["box"]["max"][i] + dw[i] for i in range(3)]})
        devices[n["id"]] = {"box": box, "zones": [], "ports": {}}
        for key, p in o["ports"].items():
            port_owner[key] = n["id"]
            port_w[key] = {"position": [p["position"][i] + dw[i] for i in range(3)], "normal": p["normal"]} if d else p
    by_key = {}
    for r in inp["routes"]:
        for end, pts in (("from", r["points"]), ("to", r["points"][::-1])):
            by_key.setdefault(r[end]["key"], pts)
    stubs = {}                                                  # 端口 key → (斜段终点, 其后方向)
    for key, p in port_w.items():
        d = _axis_dir(p["normal"], f"端口 {key}")
        owner = port_owner[key]
        if d is None:
            hint = None if owner in deltas else by_key.get(key)
            end, d2, exact = _oblique_stub(p["position"], p["normal"], hint, settings["oblique_stub_mm"])
            stubs[key] = exact
            devices[owner]["ports"][key] = (*end, *d2)
            devices[owner].setdefault("fitting_ports", []).append(key)   # 斜段末端是 45° 弯，按弯头计直管
        else:
            devices[owner]["ports"][key] = (*to_k(p["position"]), *d)
    nets, meta = [], {}
    for r in inp["routes"]:
        D = 2000 * max(s["r"] for s in r["segments"])
        if r["id"] in fixed_ids or r.get("fixed"):
            ends = [devices[port_owner[r[k]["key"]]]["box"] is None for k in ("from", "to")]
            fixed[r["id"]] = {"points": [to_k(q) for q in r["points"]], "D_mm": round(D, 1), "trim": ends}
            continue
        rho, lmin, (ps_a, ps_b) = _straight_rules(D, (1000 * r["leadA"], 1000 * r["leadB"]), settings["pipe_rules"],
                                                  rp["delta_pp_mm"])
        ka, kb = r["from"]["key"], r["to"]["key"]
        net = {"id": r["id"], "terms": [(port_owner[ka], ka), (port_owner[kb], kb)],
               "D_mm": round(D, 1), "rho_mm": rho, "lead_mm": lmin,
               "port_straight_mm": {k_: v for k_, v in ((ka, ps_a), (kb, ps_b)) if k_ not in stubs}}
        if r.get("low"):
            net["zc_max_mm"] = settings["low_zc_max_mm"]
        nets.append(net)
        meta[r["id"]] = {"code": r.get("code"), "from": r["from"]["key"], "to": r["to"]["key"]}
    # 节点内部短管（三通内的接管等）：对不接在该节点上的管是障碍；随节点平移
    spools = []
    attached = {}
    for r in inp["routes"]:
        for end in ("from", "to"):
            attached.setdefault(port_owner[r[end]["key"]], []).append(2000 * max(s["r"] for s in r["segments"]))
    for n in inp["nodes"]:
        d = deltas.get(n["id"]) or (0.0, 0.0, 0.0)
        for p_, q_ in n["orientations"][0].get("spools") or []:
            spools.append({"owner": n["id"], "D_mm": round(max(attached.get(n["id"], [rp["D_default_mm"]])), 1),
                           "points": [to_k([p_[i] + d[i] for i in range(3)]), to_k([q_[i] + d[i] for i in range(3)])]})
    # 并行进程各自要建一遍网格：管少时并行不划算，按每进程至少 SERIAL_NETS 根管分配
    rp["route_workers"] = max(1, min(rp["route_workers"], len(nets) // SERIAL_NETS))
    sc = rt.Scene(devices, nets, rp, settings["weights"], settings["scale"], fixed, spools)
    return sc, {"stubs": stubs, "port_w": port_w, "meta": meta}


def _route_once(inp, settings, fixed_ids=(), deltas=None, log=None):
    sc, meta = build_scene(inp, settings, fixed_ids, deltas)
    routes, history, G = rt.negotiate(sc, log=log or (lambda _l: None))
    viol, met = rt.check_routes(sc, routes)
    ok = not viol and len(routes) == len(sc.nets)
    w, s_ = sc.w, sc.scale
    cost = sum(w["length"] * v["L_mm"] / s_["L0"] + w["bends"] * v["bends"] / s_["B0"]
               + w["height_changes"] * v["height_changes"] / s_["C0"] for v in met["per_net"].values())
    return {"sc": sc, "meta": meta, "routes": routes, "viol": viol, "met": met, "ok": ok,
            "cost": cost if ok else math.inf, "history": history, "G": G}


def _to_web_routes(meta, routes):
    out = []
    for nid, brs in routes.items():
        m = meta["meta"][nid]
        pts = [tuple(q) for q in brs[0]["points"]]
        a, b = brs[0]["start"][1], brs[0]["end"][1][1]
        if a != m["from"]:                                      # 统一成原管道的 from → to 方向
            pts, a, b = pts[::-1], b, a
        A, B = list(meta["port_w"][a]["position"]), list(meta["port_w"][b]["position"])
        SA, SB = meta["stubs"].get(a), meta["stubs"].get(b)
        inner = [to_w(q) for q in pts[1:-1]]
        # 内核坐标取整到 0.1 mm；与端口（或斜段终点）共线的坐标改回精确值，避免对方看到微斜的管段
        refs = [c for c in (A, B, SA, SB) if c is not None]
        for q in inner:
            for k in range(3):
                for c in refs:
                    if abs(q[k] - c[k]) < 6e-4:
                        q[k] = c[k]
                        break
        full = [A] + ([SA] if SA else []) + inner + ([SB] if SB else []) + [B]
        out.append({"id": nid, "code": m["code"], "points": full})
    return out


def route_scene(inp, settings, fixed_ids=(), log=None):
    """设备不动，重布（未固定的）全部管道。返回网页坐标下的结果。"""
    t0 = time.time()
    r = _route_once(inp, settings, fixed_ids, None, log)
    return {"ok": r["ok"], "violations": r["viol"],
            "metrics": {k: v for k, v in r["met"].items() if k != "per_net"},
            "routes": _to_web_routes(r["meta"], r["routes"]), "offsets": {}, "grid_nodes": r["G"].size,
            "iterations": len(r["history"]), "timing": {"route_s": round(time.time() - t0, 1)}}


# ================================================================== 设备移动
def _k_ports(node):
    """节点当前姿态下的端口（内核坐标）：{key: (位置, 轴向法向或 None)}。"""
    return {k: (to_k(p["position"]), _axis_dir(p["normal"], k)) for k, p in node["orientations"][0]["ports"].items()}


def _inline_axis(node):
    """两口、法向相反且沿同一坐标轴的节点（在线传感器、阀门等）→ 该轴；否则 None。"""
    ps = list(_k_ports(node).values())
    if len(ps) != 2 or ps[0][1] is None or ps[1][1] is None:
        return None
    a = [i for i in range(3) if ps[0][1][i]]
    return a[0] if a and ps[1][1][a[0]] == -ps[0][1][a[0]] else None


def _candidates(inp, meta0, movable, radius_m):
    """候选移动：[(说明, {节点: delta})]。
    1. 串联拉直：相连的在线设备（同一轴向）整串平移到同一条直线上。候选直线取自串外端点（斜口取斜段终点）
       与串内各端口的横向坐标。
    2. 单个对齐：每个可移动节点对齐到相连管道另一端的端口直线（两个横向坐标都对齐，或只对齐其一）。"""
    nodes = {n["id"]: n for n in inp["nodes"]}
    owner = {k: n["id"] for n in inp["nodes"] for k in n["orientations"][0]["ports"]}
    kdir = {}                                                     # 端口 → 内核里的出管方向（斜口为斜段之后的方向）
    for nid, dev in meta0["sc"].dev.items():
        for k, v in dev["ports"].items():
            kdir[k] = v[3:]
    exact = {k: (meta0["stubs"].get(k) or p["position"]) for k, p in meta0["port_w"].items()}   # 网页坐标（米）

    def wdir(k):                                                  # 内核方向 → 网页坐标轴下标
        d = kdir[k]
        return {0: 0, 1: 2, 2: 1}[[i for i in range(3) if d[i]][0]]
    pipes = [r for r in inp["routes"] if not r.get("fixed")]
    out = []
    # ---------------- 串联拉直
    inline = {nid: _inline_axis(nodes[nid]) for nid in movable}
    inline = {k: {0: 0, 1: 2, 2: 1}[v] for k, v in inline.items() if v is not None}    # 转成网页坐标轴
    adj = {nid: set() for nid in inline}
    for r in pipes:
        a, b = owner[r["from"]["key"]], owner[r["to"]["key"]]
        if a in inline and b in inline and inline[a] == inline[b]:
            adj[a].add(b); adj[b].add(a)
    seen = set()
    for start in inline:
        if start in seen:
            continue
        comp, stack = set(), [start]
        while stack:
            x = stack.pop()
            if x not in comp:
                comp.add(x); stack.extend(adj[x] - comp)
        seen |= comp
        ax = inline[start]
        perp = [i for i in range(3) if i != ax]
        lines = set()
        for r in pipes:
            for mine, other in (("from", "to"), ("to", "from")):
                if owner[r[mine]["key"]] in comp and owner[r[other]["key"]] not in comp:
                    k = r[other]["key"]
                    if wdir(k) == ax:                                # 串外端点朝向与串同轴：可以直接对齐
                        lines.add(tuple(exact[k][i] for i in perp))
        for nid in comp:
            for k in nodes[nid]["orientations"][0]["ports"]:
                lines.add(tuple(exact[k][i] for i in perp))
        for line in sorted(lines):
            move = {}
            for nid in comp:
                pos = exact[next(iter(nodes[nid]["orientations"][0]["ports"]))]
                d = [0.0, 0.0, 0.0]
                for j, i in enumerate(perp):
                    d[i] = line[j] - pos[i]
                if any(abs(v) > radius_m + 1e-9 for v in d):
                    break
                if any(abs(v) > 1e-7 for v in d):
                    move[nid] = tuple(d)
            else:
                if move:
                    out.append((f"串联拉直 {len(comp)} 个设备到直线 {line}", move))
    # ---------------- 单个对齐
    for nid in movable:
        for r in pipes:
            for mine, other in (("from", "to"), ("to", "from")):
                if owner[r[mine]["key"]] != nid or owner[r[other]["key"]] == nid:
                    continue
                k_me, k_ot = r[mine]["key"], r[other]["key"]
                if _axis_dir(nodes[nid]["orientations"][0]["ports"][k_me]["normal"], k_me) is None:
                    continue
                ax = wdir(k_me)
                perp = [i for i in range(3) if i != ax]
                for use in (perp, perp[:1], perp[1:]):
                    d = [0.0, 0.0, 0.0]
                    for i in use:
                        d[i] = exact[k_ot][i] - exact[k_me][i]
                    if any(abs(v) > 1e-7 for v in d) and all(abs(v) <= radius_m + 1e-9 for v in d):
                        out.append((f"对齐 {nid} 到 {r.get('code')} 另一端", {nid: tuple(d)}))
    uniq, keys = [], set()
    for name, move in out:
        key = tuple(sorted(move.items()))
        if key not in keys:
            keys.add(key); uniq.append((name, move))
    return uniq


def _net_costs(r):
    """每根重布管的代价（与 _route_once 的总代价同一口径）。"""
    w, s_ = r["sc"].w, r["sc"].scale
    return {nid: w["length"] * v["L_mm"] / s_["L0"] + w["bends"] * v["bends"] / s_["B0"]
            + w["height_changes"] * v["height_changes"] / s_["C0"] for nid, v in r["met"]["per_net"].items()}


def optimize_scene(inp, settings, log=None):
    """布管 + 设备移动（局部或全局，范围由输入的 move 标记与 fixed 管决定）。只做平移，不改朝向。
      1. 按当前姿态重布范围内全部管，得到基准；
      2. 逐个试候选移动（串联拉直、单个对齐，见 _candidates）。每个候选只重布与被移动对象相连的管，
         其余范围内的管按当前路径固定为障碍；这几根管的代价下降就接受，并以新姿态重新生成候选；
      3. 没有改进或到达时间上限后，按最终姿态把范围内全部管再联合重布一次，取两者中更好的。
    所有比较都用内核自己的指标（校验器口径）；最终是否采用由调用方的校验决定。"""
    log = log or (lambda _l: None)
    t0 = time.time()
    limit = float(inp.get("seconds") or math.inf)
    radius = float(inp.get("radius") or 0)                        # 各轴移动范围（米，网页坐标）
    movable = [n["id"] for n in inp["nodes"] if n.get("move")]
    owner = {k: n["id"] for n in inp["nodes"] for k in n["orientations"][0]["ports"]}
    scope = [r for r in inp["routes"] if not r.get("fixed")]
    first = _route_once(inp, settings, (), None)
    log(f"当前姿态重布：{'通过' if first['ok'] else '有违规'}，代价 {first['cost']:.3f}；可移动节点 {len(movable)} 个，移动范围 ±{radius} m")
    if not first["ok"]:
        return _result(first, {}, [], 0, first["cost"], log)
    routes_w = {r["id"]: r["points"] for r in _to_web_routes(first["meta"], first["routes"])}
    cost = _net_costs(first)
    deltas, tried, accepted, meta_now = {}, 0, [], first
    improved = True
    while improved and movable and radius > 0 and time.time() - t0 < limit:
        improved = False
        for name, move in _candidates(_shifted(inp, deltas), meta_now["meta"] | {"sc": meta_now["sc"]}, movable, radius):
            if time.time() - t0 > limit:
                break
            trial = dict(deltas)
            for nid, d in move.items():
                acc = tuple(trial.get(nid, (0.0, 0.0, 0.0))[i] + d[i] for i in range(3))
                if any(abs(v) > radius + 1e-9 for v in acc):
                    break
                trial[nid] = acc
            else:
                moved = set(move)
                inc = {r["id"] for r in scope if owner[r["from"]["key"]] in moved or owner[r["to"]["key"]] in moved}
                if not inc:
                    continue
                tried += 1
                sub = {**inp, "routes": [r if r["id"] in inc or r.get("fixed") else {**r, "points": routes_w[r["id"]], "fixed": True}
                                         for r in inp["routes"]]}
                r = _route_once(sub, settings, (), trial)
                if not r["ok"]:
                    continue
                new = _net_costs(r)
                gain = sum(cost[x] for x in inc) - sum(new[x] for x in inc)
                if gain > 1e-9:
                    deltas, meta_now = trial, r
                    for x in _to_web_routes(r["meta"], r["routes"]):
                        routes_w[x["id"]] = x["points"]
                    cost.update(new)
                    accepted.append({"move": name, "gain": round(gain, 4)})
                    log(f"接受：{name}，代价 -{gain:.3f}")
                    improved = True
                    break                                              # 姿态变了，重新生成候选
    total_inc = sum(cost.values())
    if accepted:
        final = _route_once(inp, settings, (), deltas)                # 最终姿态下联合重布全部范围内的管
        if final["ok"] and final["cost"] <= total_inc + 1e-9:
            return _result(final, deltas, accepted, tried, first["cost"], log, time.time() - t0)
        log(f"联合重布未更好（{final['cost']:.3f} ≥ {total_inc:.3f}），采用逐步移动的结果")
        out = _result(first, deltas, accepted, tried, first["cost"], log, time.time() - t0)
        out.update(routes=[{"id": k, "code": next(r.get("code") for r in scope if r["id"] == k), "points": v}
                           for k, v in routes_w.items()], cost=round(total_inc, 4), metrics=None)
        return out
    return _result(first, deltas, accepted, tried, first["cost"], log, time.time() - t0)


def _result(r, deltas, accepted, tried, base_cost, log, seconds=0.0):
    offsets = {nid: list(d) for nid, d in deltas.items() if any(abs(v) > 1e-9 for v in d)}
    return {"ok": r["ok"], "violations": r["viol"],
            "metrics": {k: v for k, v in r["met"].items() if k != "per_net"},
            "routes": _to_web_routes(r["meta"], r["routes"]), "offsets": offsets,
            "moves": accepted, "candidates_tried": tried,
            "base_cost": round(base_cost, 4) if base_cost < math.inf else None,
            "cost": round(r["cost"], 4) if r["cost"] < math.inf else None,
            "iterations": len(r["history"]), "grid_nodes": r["G"].size,
            "timing": {"route_s": round(seconds, 1)}}


def _shifted(inp, deltas):
    """按已接受的平移更新节点端口 / 包围盒（网页坐标），供生成下一轮候选。"""
    if not deltas:
        return inp
    nodes = []
    for n in inp["nodes"]:
        d = deltas.get(n["id"])
        if not d:
            nodes.append(n)
            continue
        dw = tuple(d)
        o = n["orientations"][0]
        ports = {k: {"position": [p["position"][i] + dw[i] for i in range(3)], "normal": p["normal"]} for k, p in o["ports"].items()}
        box = {"min": [o["box"]["min"][i] + dw[i] for i in range(3)], "max": [o["box"]["max"][i] + dw[i] for i in range(3)]} if o.get("box") else None
        nodes.append({**n, "orientations": [{**o, "ports": ports, "box": box}] + n["orientations"][1:]})
    return {**inp, "nodes": nodes}
