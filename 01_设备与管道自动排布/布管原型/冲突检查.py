"""冲突点单独检查：判断精细布管剩下的冲突是“空间不够”还是“协商没找到解”。

对每个摆放：先按原流程协商布线（结果存盘，重跑时直接读取），再逐个处理校验器报出的冲突：
  其他管网全部固定，它们的占用晕设为硬障碍（不加价、不协商），只对冲突管网单独重搜，扩展上限放大。
  管–管冲突 X 与 Y 依次尝试：只重布 X → 只重布 Y → 两根都拆掉后先 X 后 Y → 先 Y 后 X。
  某次尝试后整体用独立校验器 check_routes 复核，冲突数不增加且该冲突消失才接受。
单管搜索的结果分三种：
  找到路径；搜索空间耗尽（在当前网格与其他管道位置下确实没有路径）；超过扩展上限（不能下结论）。

运行：.venv/Scripts/python 冲突检查.py [方法前缀，默认 B] [扩展上限，默认 3000000]
输出：冲突检查结果_{方法}.json、冲突检查_路线_{方法}_种子{s}.json
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
import 布管实验 as ex  # noqa: E402

rt = ex.rt
sys.stdout.reconfigure(encoding="utf-8")


def dump_routes(routes):
    return {nid: [{"start": list(b["start"]), "end": [b["end"][0], list(b["end"][1]) if b["end"][1] else None],
                   "points": [list(p) for p in b["points"]]} for b in brs] for nid, brs in routes.items()}


def load_routes(data):
    return {nid: [{"start": tuple(b["start"]), "end": (b["end"][0], tuple(b["end"][1]) if b["end"][1] else None),
                   "points": [tuple(p) for p in b["points"]]} for b in brs] for nid, brs in data.items()}


def clashes(sc, routes):
    """返回 (冲突列表 [(管网, 对方或 None, 原文)], 全部违规)。"""
    viol, _ = rt.check_routes(sc, routes)
    out = []
    for v in viol:
        if "管–管净距不足" in v:
            a, b = v.split("：")[0].split(" 与 ")
            out.append((a, b, v))
        elif "净距不足" in v or "穿过" in v:
            out.append((v.split("：")[0], None, v))
    return out, viol


def route_fixed(G, sc, routes, nid):
    """其他管网固定为硬障碍，单独搜索 nid。返回 (支路或 None, 结果说明, 用时)。"""
    t0 = time.time()
    e = next(i for i, n in enumerate(sc.nets) if n["id"] == nid)
    blocked = set()
    for other, brs in routes.items():
        if other != nid:
            blocked |= rt.halo_edges(G, sc, brs)
    saved = [(n, a, G.eblocked[a][n]) for n, a in blocked]
    for n, a in blocked:
        G.eblocked[a][n] = True
    stats = {"expansions": 0, "exhausted": 0}
    try:
        br, reason = rt.route_net(G, sc, sc.nets[e], {"halo": rt._Zero(), "hist": rt._Zero(), "pres": 0.0,
                                                      "stats": stats})
    finally:
        for n, a, v in saved:
            G.eblocked[a][n] = v
    el = round(time.time() - t0, 1)
    if br is not None:
        return br, f"找到（扩展 {stats['expansions']}）", el
    kind = "超过扩展上限，不能下结论" if stats["exhausted"] else "搜索空间耗尽：确实无路"
    return None, f"{kind}（{reason}；扩展 {stats['expansions']}）", el


def try_fix(G, sc, routes, a, b, n_before):
    """按顺序尝试，接受第一个“该冲突消失且总冲突数不增加”的方案。返回 (新路线或 None, 尝试记录)。"""
    plans = [[a]] + ([[b], [a, b], [b, a]] if b else [])
    log = []
    for order in plans:
        trial = {k: v for k, v in routes.items() if k not in order}
        steps, ok = [], True
        for nid in order:
            br, msg, el = route_fixed(G, sc, trial, nid)
            steps.append({"net": nid, "result": msg, "time_s": el})
            if br is None:
                ok = False
                break
            trial[nid] = br
        rec = {"reroute": order, "steps": steps}
        log.append(rec)
        print(f"    重布 {'→'.join(order)}：" + "；".join(f"{s['net']} {s['result']} {s['time_s']} s" for s in steps),
              flush=True)
        if not ok:
            continue
        cl, _ = clashes(sc, trial)
        still = any({x, y} == {a, b} if b else (x == a and y is None) for x, y, _ in cl)
        rec["clashes_after"] = len(cl)
        print(f"      复核：冲突 {len(cl)}（原 {n_before}），{'该冲突仍在' if still else '该冲突已消失'}", flush=True)
        if not still and len(cl) <= n_before:
            return trial, log
    return None, log


def one(inst, c, max_exp):
    placed = {k: tuple(v) for k, v in c["placement"].items()}
    blocks, nets, P, sol = ex.device_level(inst, placed)
    sc = ex.scene(inst, blocks, nets, P, sol)
    cache = HERE / f"冲突检查_路线_{c['method'][0]}_种子{c['seed']}.json"
    print(f"\n===== {c['method']} 种子 {c['seed']} =====", flush=True)
    if cache.exists():
        routes = load_routes(json.loads(cache.read_text(encoding="utf-8")))
        route_s = None
        print("读取已存协商结果", flush=True)
    else:
        t0 = time.time()
        routes, hist, _ = rt.negotiate(sc, log=lambda x: None)
        route_s = round(time.time() - t0, 1)
        cache.write_text(json.dumps(dump_routes(routes), ensure_ascii=False), encoding="utf-8")
        print(f"协商布线 {route_s} s，{len(hist)} 轮，布通 {len(routes)}/{len(sc.nets)}", flush=True)
    sc.rp["max_expansions"] = max_exp
    G = rt.Grid(sc)
    cl0, viol0 = clashes(sc, routes)
    print(f"协商后：违规 {len(viol0)}，其中冲突 {len(cl0)}：" + "；".join(v for *_, v in cl0), flush=True)
    records = []
    t_all = time.time()
    for a, b, text in cl0:
        cl, _ = clashes(sc, routes)
        if not any((x, y) == (a, b) for x, y, _ in cl):
            print(f"  {text}：前面的修复已顺带消除", flush=True)
            records.append({"clash": text, "result": "已顺带消除"})
            continue
        print(f"  处理：{text}", flush=True)
        new, log = try_fix(G, sc, routes, a, b, len(cl))
        if new is not None:
            routes = new
        records.append({"clash": text, "result": "已消除" if new is not None else "未消除", "attempts": log})
    cl1, viol1 = clashes(sc, routes)
    _, met = rt.check_routes(sc, routes)
    print(f"检查后：违规 {len(viol1)}，冲突 {len(cl1)}，用时 {time.time() - t_all:.0f} s；"
          f"管长 {met['L_m']} m，弯头 {met['bends']}，J={met['J']}" + ("  【全部无违规】" if not viol1 else ""), flush=True)
    for v in viol1:
        print(f"    {v}")
    return {"method": c["method"], "seed": c["seed"], "negotiate_s": route_s, "max_expansions": max_exp,
            "before": {"violations": viol0, "clashes": len(cl0)}, "records": records,
            "after": {"violations": viol1, "clashes": len(cl1), "all_ok": not viol1,
                      "metrics": {k: v for k, v in met.items() if k != "per_net"}},
            "check_s": round(time.time() - t_all, 1)}


def main():
    pick = sys.argv[1] if len(sys.argv) > 1 else "B"
    max_exp = int(sys.argv[2]) if len(sys.argv) > 2 else 3_000_000
    d = json.loads((ex.bk.HERE / "设备算例_小.json").read_text(encoding="utf-8"))
    d["params"]["blocking"]["auto_module_policy"] = "report_only"
    inst = ex.bk.load_devices(d)
    cases = json.loads((HERE / "联合验证结果.json").read_text(encoding="utf-8")) + \
        json.loads((HERE / "联合验证结果_C.json").read_text(encoding="utf-8"))
    rows = []
    for c in [c for c in cases if c["method"].startswith(pick)]:
        rows.append(one(inst, c, max_exp))
        (HERE / f"冲突检查结果_{pick}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n汇总：")
    for r in rows:
        print(f"  {r['method']} 种子 {r['seed']}：冲突 {r['before']['clashes']} → {r['after']['clashes']}，"
              f"违规 {len(r['before']['violations'])} → {len(r['after']['violations'])}，检查用时 {r['check_s']} s")


if __name__ == "__main__":
    main()
