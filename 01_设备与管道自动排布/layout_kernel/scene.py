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
import time

from . import build

rt = build.rt
SETTINGS_REQUIRED = ["routing", "weights", "scale", "low_zc_max_mm", "oblique_stub_mm", "pipe_rules"]
RULES_REQUIRED = ["trim_ratio", "radius_margin_mm", "port_margin_mm", "safety", "rule_D_mm", "rule_R_mm"]


def _straight_rules(D_pipe, leads, rules, delta_pp):
    """把对方的直管规则换算成内核的 (rho, lmin)。内核要求：两弯之间直管 ≥ 2·rho + lmin，端口到弯 ≥ rho + lmin。
    对方（网页 pipeSections / dimensionalSections）：弯头让位 trim = min(R, trim_ratio·相邻直管)，有效半径 trim 须
    > D/2 + margin；端口处去掉让位后的直管 ≥ 直颈 + 变径（lead）+ port_margin。于是
      两弯之间最短  S_ee = (D/2 + margin) / trim_ratio
      端口到弯最短  S_pe = lead + R + port_margin          （若此时 trim_ratio·S_pe ≥ R）
                          (lead + port_margin)/(1 − trim_ratio)  （否则）
    另外两弯之间不小于 实际外径 + 管间净距，避免 U 形回弯两边相碰（对方校验会查自相交）。"""
    k, t, D, R = rules["safety"], rules["trim_ratio"], rules["rule_D_mm"], rules["rule_R_mm"]
    s_rad = (D / 2 + rules["radius_margin_mm"]) / t                  # 每个弯头两侧直管都要满足（含端口后的第一个弯）
    s_ee = k * max(s_rad, D_pipe + delta_pp)

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


def build_scene(inp, settings, fixed_ids=()):
    """网页优化输入 → routing.Scene。返回 (Scene, 元数据)。fixed_ids 中的管保持原路径，作为障碍。"""
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
        devices[n["id"]] = {"box": _box_k(o["box"]) if o.get("box") else None, "zones": [], "ports": {}}
        for key, p in o["ports"].items():
            port_owner[key] = n["id"]
            port_w[key] = p
    by_key = {}
    for r in inp["routes"]:
        for end, pts in (("from", r["points"]), ("to", r["points"][::-1])):
            by_key.setdefault(r[end]["key"], pts)
    stubs = {}                                                  # 端口 key → (斜段终点, 其后方向)
    for key, p in port_w.items():
        d = _axis_dir(p["normal"], f"端口 {key}")
        owner = port_owner[key]
        if d is None:
            end, d2, exact = _oblique_stub(p["position"], p["normal"], by_key.get(key), settings["oblique_stub_mm"])
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
    sc = rt.Scene(devices, nets, rp, settings["weights"], settings["scale"], fixed)
    return sc, {"stubs": stubs, "port_w": port_w, "meta": meta}


def route_scene(inp, settings, fixed_ids=(), log=None):
    """设备不动，重布（未固定的）全部管道。返回网页坐标下的结果。"""
    log = log or (lambda _l: None)
    t0 = time.time()
    sc, meta = build_scene(inp, settings, fixed_ids)
    t1 = time.time()
    routes, history, G = rt.negotiate(sc, log=log)
    t2 = time.time()
    viol, met = rt.check_routes(sc, routes)
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
    return {"ok": not viol and len(routes) == len(sc.nets), "violations": viol,
            "metrics": {k: v for k, v in met.items() if k != "per_net"},
            "routes": out, "grid_nodes": G.size, "iterations": len(history),
            "timing": {"build_s": round(t1 - t0, 1), "route_s": round(t2 - t1, 1), "check_s": round(time.time() - t2, 1)}}
