"""分块：设备级算例 → 块级算例（供 摆放原型/placement_*.py 使用）+ 按连接关系聚簇。

对应主文档第 8 节与 4.2 节，分两类，不能混淆：
  1. 标准模块（刚性）：内部相对位置固定，整体平移 / 旋转。来源：
       a. 输入 JSON 的 "modules" 标注（优先）；
       b. 自动识别的候选（auto_module_policy = "accept" 时使用，"report_only" 时只报告）。
     内部排法：标注中给出 "layouts" 就用；否则用 CP-SAT 按几个长宽比区间各求一个候选。
  2. 临时簇：按连接关系聚簇，只是求解手段，不锁定块的相对位置。

自动识别（启发式，不保证找全、不保证与设计意图一致，结果须由人确认）：
  - 并联阵列：同类型、且每个端口接在同一管网上的设备（例：冷却塔组、并联空调箱）。
  - 链式模块：从同类型设备出发，沿两端点管网同步生长；只有当所有副本都能以同样的
    (成员, 端口 → 邻居类型, 邻居端口) 扩展、且邻居互不相同且未被占用时才扩展。
    遇到被多个副本共享的设备（如分集水器）自然停止。按类型出现次数从多到少依次尝试。
"""
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
from ortools.sat.python import cp_model

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "摆放原型"))
import placement_cpsat as base  # noqa: E402

REQUIRED_PARAMS = ["grid_mm", "delta_ee_mm", "l_min_mm", "kappa", "service_zones_inside_footprint", "weights"]
REQUIRED_BLOCKING = ["auto_module_policy", "min_copies", "min_members", "module_rotations",
                     "module_aspect_bands", "module_solve_s", "cluster_max_units", "cluster_resolution", "seed"]


# ============================================================ 读取与检查
def load_devices(src):
    data = json.loads(Path(src).read_text(encoding="utf-8")) if not isinstance(src, dict) else src
    P = data.get("params", {})
    miss = [k for k in REQUIRED_PARAMS if k not in P] + \
           [f"blocking.{k}" for k in REQUIRED_BLOCKING if k not in P.get("blocking", {})]
    if miss:
        raise KeyError(f"缺少必填参数：{', '.join(miss)}（不设默认值，请在算例 params 中给出）")
    if P["blocking"]["auto_module_policy"] not in ("accept", "report_only"):
        raise ValueError("blocking.auto_module_policy 只能是 accept 或 report_only")
    types, devs = data["device_types"], {}
    for d in data["devices"]:
        if d["id"] in devs:
            raise ValueError(f"设备 id 重复：{d['id']}")
        if d["type"] not in types:
            raise ValueError(f"设备 {d['id']} 的类型 {d['type']} 未定义")
        if "." in d["id"]:
            raise ValueError(f"设备 id 不能含“.”：{d['id']}")
        devs[d["id"]] = d
    port_net, nets = {}, []
    for n in data["nets"]:
        terms = []
        for t in n["terminals"]:
            did, pn = t.split(".")
            if did not in devs:
                raise ValueError(f"管网 {n['id']} 引用了不存在的设备 {did}")
            if pn not in types[devs[did]["type"]]["ports"]:
                raise ValueError(f"管网 {n['id']}：设备 {did} 没有端口 {pn}")
            if (did, pn) in port_net:
                raise ValueError(f"端口 {t} 同时接在 {port_net[(did, pn)]} 与 {n['id']} 上")
            port_net[(did, pn)] = n["id"]
            terms.append((did, pn))
        if len(terms) < 2:
            raise ValueError(f"管网 {n['id']} 少于 2 个端点")
        nets.append({"id": n["id"], "terms": terms})
    unconnected = [d for d in devs if not any(k[0] == d for k in port_net)]
    return {"data": data, "P": P, "bp": P["blocking"], "types": types, "devs": devs,
            "nets": nets, "net_by_id": {n["id"]: n for n in nets}, "port_net": port_net,
            "unconnected": unconnected}


# ============================================================ 模块：标注
def annotated_modules(inst):
    out, used = [], {}
    for m in inst["data"].get("modules", []):
        roles = None
        for ins in m["instances"]:
            rt = {r: inst["devs"][d]["type"] for r, d in ins["members"].items()}
            for r, d in ins["members"].items():
                if d not in inst["devs"]:
                    raise ValueError(f"模块 {m['template']} 实例 {ins['id']}：设备 {d} 不存在")
                if d in used:
                    raise ValueError(f"设备 {d} 同时属于 {used[d]} 与 {m['template']}")
                if "fixed" in inst["devs"][d]:
                    raise ValueError(f"固定设备 {d} 不能放进模块")
                used[d] = m["template"]
            if roles is None:
                roles = rt
            elif rt != roles:
                raise ValueError(f"模块 {m['template']} 各实例的成员角色或类型不一致")
        out.append({"template": m["template"], "source": "标注", "roles": roles,
                    "instances": [{"id": ins["id"], "members": dict(ins["members"])} for ins in m["instances"]],
                    "layouts": m.get("layouts")})
    return out


# ============================================================ 模块：自动识别
def detect_arrays(inst, taken):
    groups = defaultdict(list)
    for did, d in inst["devs"].items():
        if did in taken or "fixed" in d:
            continue
        ports = inst["types"][d["type"]]["ports"]
        conn = tuple(sorted((pn, inst["port_net"].get((did, pn))) for pn in ports))
        if all(net is None for _, net in conn):
            continue
        groups[(d["type"], conn)].append(did)
    by_tpl = defaultdict(list)
    for (typ, _), members in sorted(groups.items(), key=lambda kv: kv[1][0]):
        if len(members) >= inst["bp"]["min_copies"]:
            by_tpl[(typ, len(members))].append(sorted(members))
    out = []
    for (typ, k), insts in sorted(by_tpl.items()):
        roles = {f"{typ}#{i}": typ for i in range(k)}
        out.append({"template": f"阵列_{typ}×{k}", "source": "自动-并联阵列", "roles": roles, "layouts": None,
                    "instances": [{"id": f"阵列_{typ}×{k}_{j + 1}", "members": {f"{typ}#{i}": d for i, d in enumerate(ms)}}
                                  for j, ms in enumerate(insts)]})
    return out


def _neighbor(inst, did, pn):
    net = inst["port_net"].get((did, pn))
    if net is None or len(inst["net_by_id"][net]["terms"]) != 2:
        return None
    a, b = inst["net_by_id"][net]["terms"]
    return b if a == (did, pn) else a


def detect_chains(inst, taken):
    bp, types, devs = inst["bp"], inst["types"], inst["devs"]
    free = lambda d: d not in taken and "fixed" not in devs[d]
    count = Counter(devs[d]["type"] for d in devs if free(d))
    out = []
    for typ, c in sorted(count.items(), key=lambda kv: (-kv[1], kv[0])):
        if c < bp["min_copies"]:
            continue
        seeds = [d for d in sorted(devs) if devs[d]["type"] == typ and free(d)]
        parts = defaultdict(list)
        for d in seeds:
            sig = []
            for pn in sorted(types[typ]["ports"]):
                nb = _neighbor(inst, d, pn)
                sig.append((pn, devs[nb[0]]["type"] if nb else "*"))
            parts[tuple(sig)].append(d)
        for members in parts.values():
            members = [d for d in members if free(d)]
            if len(members) < bp["min_copies"]:
                continue
            copies = [[d] for d in members]
            roles = [(typ, None)]                                    # 角色：(类型, 来源描述)
            grown = True
            while grown:
                grown = False
                in_any = {d for cp in copies for d in cp}
                for r in range(len(copies[0])):
                    rtype = devs[copies[0][r]]["type"]
                    for pn in sorted(types[rtype]["ports"]):
                        nbs = [_neighbor(inst, cp[r], pn) for cp in copies]
                        if any(nb is None for nb in nbs):
                            continue
                        us = [nb[0] for nb in nbs]
                        if any(u in cp for u, cp in zip(us, copies)):
                            continue                                 # 已在本副本内（成环）
                        if len({(devs[u]["type"], q) for u, q in nbs}) != 1:
                            continue
                        if len(set(us)) != len(us) or any((not free(u)) or u in in_any for u in us):
                            continue
                        for cp, u in zip(copies, us):
                            cp.append(u)
                        roles.append((devs[us[0]]["type"], f"{r}.{pn}"))
                        in_any.update(us)
                        grown = True
            if len(copies[0]) < bp["min_members"]:
                continue
            names, seen = [], Counter()
            for rtype, _ in roles:
                seen[rtype] += 1
                names.append(rtype if seen[rtype] == 1 else f"{rtype}{seen[rtype]}")
            tpl = "链_" + "+".join(names)
            out.append({"template": tpl, "source": "自动-链式", "roles": {nm: rt for nm, (rt, _) in zip(names, roles)},
                        "layouts": None,
                        "instances": [{"id": f"{tpl}_{j + 1}", "members": dict(zip(names, cp))}
                                      for j, cp in enumerate(copies)]})
            taken.update(d for cp in copies for d in cp)
    return out


def find_modules(inst):
    """返回 (采用的模块, 自动候选报告)。标注优先；自动候选按 auto_module_policy 决定是否采用。"""
    ann = annotated_modules(inst)
    taken = {d for m in ann for ins in m["instances"] for d in ins["members"].values()}
    auto_taken = set(taken)
    arrays = detect_arrays(inst, auto_taken)
    auto_taken.update(d for m in arrays for ins in m["instances"] for d in ins["members"].values())
    chains = detect_chains(inst, auto_taken)
    auto = arrays + chains
    used = ann + (auto if inst["bp"]["auto_module_policy"] == "accept" else [])
    for m in used:
        check_module_consistency(inst, m)
    return used, auto


def check_module_consistency(inst, m):
    """同一模块各实例的内部管网结构必须一致；记录内部管网与对外端口。"""
    sigs = []
    for ins in m["instances"]:
        role_of = {d: r for r, d in ins["members"].items()}
        internal, external = set(), set()
        for n in inst["nets"]:
            inside = [(role_of[d], pn) for d, pn in n["terms"] if d in role_of]
            if not inside:
                continue
            if len(inside) == len(n["terms"]):
                internal.add(frozenset(inside))
            else:
                external.update(inside)
        sigs.append((frozenset(internal), frozenset(external)))
    if len(set(sigs)) != 1:
        raise ValueError(f"模块 {m['template']} 各实例的内部连接或对外端口不一致，不能作为同一刚性模块")
    m["internal_nets"] = [sorted(s) for s in sigs[0][0]]
    m["external_ports"] = sorted(sigs[0][1])


# ============================================================ 模块内部排法
def member_poses(inst, typ):
    t = inst["types"][typ]
    return [base.make_pose(t, r, inst["P"]["grid_mm"], "默认") for r in t["rotations"]]


def given_layouts(inst, m):
    grid, d = inst["P"]["grid_mm"], base.g(inst["P"]["delta_ee_mm"], inst["P"]["grid_mm"])
    out = []
    for lay in m["layouts"]:
        items = []
        for role, typ in m["roles"].items():
            if role not in lay["members"]:
                raise ValueError(f"模块 {m['template']} 排法 {lay['name']} 缺少成员 {role}")
            spec = lay["members"][role]
            t = inst["types"][typ]
            if spec["rot"] not in t["rotations"]:
                raise ValueError(f"模块 {m['template']} 排法 {lay['name']}：{role} 不允许旋转 {spec['rot']}")
            pose = base.make_pose(t, spec["rot"], grid, "默认")
            items.append((role, base.g(spec["pos"][0], grid), base.g(spec["pos"][1], grid), pose))
        _check_layout(m, lay["name"], items, d)
        out.append(_normalize(lay["name"], items))
    return out


def _check_layout(m, name, items, d):
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            ra, rb = _rect(items[a]), _rect(items[b])
            gx = max(rb[0] - (ra[0] + ra[2]), ra[0] - (rb[0] + rb[2]))
            gy = max(rb[1] - (ra[1] + ra[3]), ra[1] - (rb[1] + rb[3]))
            if max(gx, gy) < d:
                raise ValueError(f"模块 {m['template']} 排法 {name}：{items[a][0]} 与 {items[b][0]} 净距不足")
    for a in items:
        for zx, zy, zw, zh in a[3]["zones"]:
            z = (a[1] + zx, a[2] + zy, zw, zh)
            for b in items:
                if b is a:
                    continue
                r = _rect(b)
                if min(z[0] + z[2], r[0] + r[2]) > max(z[0], r[0]) and min(z[1] + z[3], r[1] + r[3]) > max(z[1], r[1]):
                    raise ValueError(f"模块 {m['template']} 排法 {name}：{a[0]} 的检修区压到 {b[0]}")


def _rect(item):
    return (item[1], item[2], item[3]["W"], item[3]["H"])


def _normalize(name, items):
    x0 = min(i[1] for i in items); y0 = min(i[2] for i in items)
    items = [(r, x - x0, y - y0, p) for r, x, y, p in items]
    return {"name": name, "W": max(x + p["W"] for _, x, _, p in items),
            "H": max(y + p["H"] for _, _, y, p in items), "items": items}


def solve_layouts(inst, m):
    """每个长宽比区间求一个排法：min 面积/A₀ + 块内管长下界/L₀（端口坐标跨度）。"""
    P, bp = inst["P"], inst["bp"]
    grid = P["grid_mm"]
    d = base.g(P["delta_ee_mm"], grid)
    roles = list(m["roles"])
    poses = {r: member_poses(inst, m["roles"][r]) for r in roles}
    A0 = sum(poses[r][0]["W"] * poses[r][0]["H"] for r in roles)
    L0 = max(1, len(m["internal_nets"])) * math.sqrt(A0)
    D = sum(max(max(p["W"], p["H"]) for p in poses[r]) for r in roles) + d * len(roles)
    out, seen = [], set()
    for lo, hi in bp["module_aspect_bands"]:
        mdl = cp_model.CpModel()
        X, Y, B, EX, EY, rx, ry = {}, {}, {}, {}, {}, {}, {}
        for r in roles:
            X[r] = mdl.NewIntVar(0, D, ""); Y[r] = mdl.NewIntVar(0, D, "")
            B[r] = [mdl.NewBoolVar("") for _ in poses[r]]
            mdl.AddExactlyOne(B[r])
            sx = sum(b * p["W"] for b, p in zip(B[r], poses[r])); sy = sum(b * p["H"] for b, p in zip(B[r], poses[r]))
            EX[r] = mdl.NewIntVar(0, D, ""); EY[r] = mdl.NewIntVar(0, D, "")
            mdl.Add(EX[r] == X[r] + sx); mdl.Add(EY[r] == Y[r] + sy)
            wv = mdl.NewIntVar(0, D, ""); hv = mdl.NewIntVar(0, D, "")
            mdl.Add(wv == sx); mdl.Add(hv == sy)
            rx[r] = (mdl.NewIntervalVar(X[r], wv, EX[r], ""), mdl.NewIntervalVar(X[r], wv + d, EX[r] + d, ""))
            ry[r] = (mdl.NewIntervalVar(Y[r], hv, EY[r], ""), mdl.NewIntervalVar(Y[r], hv + d, EY[r] + d, ""))
        mdl.AddNoOverlap2D([rx[r][1] for r in roles], [ry[r][1] for r in roles])
        for r in roles:
            zx_, zy_ = [], []
            for p, b in zip(poses[r], B[r]):
                for zx, zy, zw, zh in p["zones"]:
                    zx_.append(mdl.NewOptionalIntervalVar(X[r] + zx, zw, X[r] + zx + zw, b, ""))
                    zy_.append(mdl.NewOptionalIntervalVar(Y[r] + zy, zh, Y[r] + zy + zh, b, ""))
            if zx_:
                mdl.AddNoOverlap2D(zx_ + [rx[o][0] for o in roles if o != r], zy_ + [ry[o][0] for o in roles if o != r])
        W = mdl.NewIntVar(1, D, ""); H = mdl.NewIntVar(1, D, "")
        for r in roles:
            mdl.Add(EX[r] <= W); mdl.Add(EY[r] <= H)
        mdl.Add(W * 100 >= int(lo * 100) * H); mdl.Add(W * 100 <= int(hi * 100) * H)
        A = mdl.NewIntVar(1, D * D, "")
        mdl.AddMultiplicationEquality(A, [W, H])
        Ls = []
        for net in m["internal_nets"]:
            for axis in (0, 1):
                vs = []
                for r, pn in net:
                    v = mdl.NewIntVar(-D, 2 * D, "")
                    mdl.Add(v == (X[r] if axis == 0 else Y[r]) + sum(b * p["ports"][pn][axis] for b, p in zip(B[r], poses[r])))
                    vs.append(v)
                hi_ = mdl.NewIntVar(-D, 2 * D, ""); lo_ = mdl.NewIntVar(-D, 2 * D, "")
                mdl.AddMaxEquality(hi_, vs); mdl.AddMinEquality(lo_, vs)
                Ls.append(hi_ - lo_)
        wA, wL = P["weights"]["area"], P["weights"]["length"]
        mdl.Minimize(wA / A0 * A + wL / L0 * sum(Ls))
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = bp["module_solve_s"]
        s.parameters.num_workers = 8
        s.parameters.random_seed = bp["seed"]
        st = s.Solve(mdl)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            continue
        items = []
        for r in roles:
            k = next(i for i, b in enumerate(B[r]) if s.Value(b))
            items.append((r, s.Value(X[r]), s.Value(Y[r]), poses[r][k]))
        _check_layout(m, f"宽高比{lo}–{hi}", items, d)                   # 独立复核求解结果
        lay = _normalize(f"宽高比{lo}–{hi}", items)
        key = (lay["W"], lay["H"])                                       # 尺寸相同的只留一个
        if key in seen:
            continue
        seen.add(key)
        lay["status"] = s.StatusName(st)
        out.append(lay)
    if not out:
        raise RuntimeError(f"模块 {m['template']} 在所有长宽比区间内都没有求出排法")
    return out


def layout_to_variant(inst, m, lay):
    grid = inst["P"]["grid_mm"]
    ext = set(m["external_ports"])
    ports, zones = {}, []
    for r, x, y, p in lay["items"]:
        for pn, (px, py, ux, uy, z) in p["ports"].items():
            if (r, pn) in ext:
                ports[f"{r}__{pn}"] = {"pos": [(x + px) * grid, (y + py) * grid], "dir": [ux, uy], "z": z * grid}
        for zx, zy, zw, zh in p["zones"]:
            zones.append([(x + zx) * grid, (y + zy) * grid, zw * grid, zh * grid])
    return {"size": [lay["W"] * grid, lay["H"] * grid], "ports": ports, "service_zones": zones,
            "members": {r: {"pos": [x * grid, y * grid], "rot": p["rot"]} for r, x, y, p in lay["items"]}}


# ============================================================ 设备级 → 块级
def build_block_instance(inst, modules):
    """返回 (块级算例 dict, 设备 → (块 id, 端口前缀) 映射, 统计)。"""
    P, bp = inst["P"], inst["bp"]
    t0 = time.time()
    templates, blocks, owner = {}, [], {}
    layout_stats = []
    for m in modules:
        t1 = time.time()
        lays = given_layouts(inst, m) if m.get("layouts") else solve_layouts(inst, m)
        variants = {lay["name"]: layout_to_variant(inst, m, lay) for lay in lays}
        layout_stats.append({"template": m["template"], "source": m["source"], "instances": len(m["instances"]),
                             "members": len(m["roles"]),
                             "variants": {lay["name"]: [*variants[lay["name"]]["size"], lay.get("status", "给定")]
                                          for lay in lays},
                             "layout_from": "给定" if m.get("layouts") else "CP-SAT 求解",
                             "time_s": round(time.time() - t1, 2)})
        templates[m["template"]] = {"variants": variants}
        for ins in m["instances"]:
            blocks.append({"id": ins["id"], "name": ins["id"], "template": m["template"],
                           "copy_group": m["template"] if len(m["instances"]) > 1 else None,
                           "rotations": bp["module_rotations"], "members": ins["members"]})
            for r, d in ins["members"].items():
                owner[d] = (ins["id"], f"{r}__")
    for did, d in inst["devs"].items():
        if did in owner:
            continue
        t = inst["types"][d["type"]]
        b = {"id": did, "name": did, "rotations": t["rotations"],
             "variants": {"默认": {k: t[k] for k in ("size", "ports", "service_zones")}}}
        if "fixed" in d:
            b["fixed"] = d["fixed"]
        blocks.append(b)
        owner[did] = (did, "")
    nets, internal = [], 0
    for n in inst["nets"]:
        terms = [f"{owner[d][0]}.{owner[d][1]}{pn}" for d, pn in n["terms"]]
        if len({t.split(".")[0] for t in terms}) == 1 and n["terms"][0][0] in owner and owner[n["terms"][0][0]][1]:
            internal += 1                                                  # 模块内部管网：随模块刚性复制
            continue
        nets.append({"id": n["id"], "terminals": terms})
    params = {k: P[k] for k in REQUIRED_PARAMS}
    data = {"说明": "由 分块原型/blocking.py 从设备级算例生成", "params": params, "templates": templates,
            "blocks": [{k: v for k, v in b.items() if k != "members" and v is not None} for b in blocks],
            "nets": nets}
    stats = {"devices": len(inst["devs"]), "blocks": len(blocks), "module_blocks": sum(1 for b in blocks if "template" in b),
             "nets_in": len(inst["nets"]), "nets_out": len(nets), "internal_nets": internal,
             "layouts": layout_stats, "time_s": round(time.time() - t0, 2)}
    return data, owner, stats


# ============================================================ 临时聚簇
def unit_graph(bdata):
    G = nx.Graph()
    G.add_nodes_from(b["id"] for b in bdata["blocks"])
    for n in bdata["nets"]:
        us = sorted({t.split(".")[0] for t in n["terminals"]})
        if len(us) < 2:
            continue
        w = 1.0 / (len(us) - 1)                                           # 团展开：每个管网总权重与端点数无关
        for i in range(len(us)):
            for j in range(i + 1, len(us)):
                a, b = us[i], us[j]
                G.add_edge(a, b, weight=G.get_edge_data(a, b, {"weight": 0})["weight"] + w)
    return G


def cluster(bdata, max_units, resolution, seed):
    G = unit_graph(bdata)
    comms = [set(c) for c in nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)]
    done = []
    while comms:
        c = comms.pop()
        if len(c) <= max_units:
            done.append(c)
            continue
        sub = G.subgraph(c)
        parts = [set(p) for p in nx.community.louvain_communities(sub, weight="weight", resolution=resolution * 2,
                                                                  seed=seed)]
        if len(parts) <= 1:
            a, b = nx.algorithms.community.kernighan_lin_bisection(sub, weight="weight", seed=seed)
            parts = [set(a), set(b)]
        comms.extend(parts)
    return sorted((sorted(c) for c in done), key=lambda c: (-len(c), c[0]))


def cluster_metrics(bdata, clusters):
    where = {u: k for k, c in enumerate(clusters) for u in c}
    cut = sum(1 for n in bdata["nets"] if len({where[t.split(".")[0]] for t in n["terminals"]}) > 1)
    G = unit_graph(bdata)
    tot = sum(d["weight"] for _, _, d in G.edges(data=True))
    cutw = sum(d["weight"] for a, b, d in G.edges(data=True) if where[a] != where[b])
    return {"clusters": len(clusters), "sizes": sorted((len(c) for c in clusters), reverse=True),
            "cut_nets": cut, "cut_nets_pct": round(100 * cut / len(bdata["nets"]), 1),
            "cut_weight_pct": round(100 * cutw / tot, 1) if tot else 0.0}


def random_partition(clusters, seed):
    units = [u for c in clusters for u in c]
    random.Random(seed).shuffle(units)
    out, k = [], 0
    for c in clusters:
        out.append(units[k:k + len(c)]); k += len(c)
    return out


def cluster_instance(bdata, cluster_units):
    """把一个簇导出为独立块级算例：簇外端点删掉；剩余端点 < 2 的管网丢弃。"""
    keep = set(cluster_units)
    nets = []
    for n in bdata["nets"]:
        ts = [t for t in n["terminals"] if t.split(".")[0] in keep]
        if len({t.split(".")[0] for t in ts}) >= 2 or (len(ts) >= 2 and len(ts) == len(n["terminals"])):
            nets.append({"id": n["id"], "terminals": ts})
    blocks = [b for b in bdata["blocks"] if b["id"] in keep]
    used = {b["template"] for b in blocks if "template" in b}
    return {**bdata, "说明": "簇子问题（簇外端点已删除）", "blocks": blocks, "nets": nets,
            "templates": {k: v for k, v in bdata["templates"].items() if k in used}}
