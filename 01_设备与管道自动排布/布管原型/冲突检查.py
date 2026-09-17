"""冲突点单独检查（清理逻辑已并入 routing.cleanup）：判断精细布管剩下的冲突是“空间不够”还是“协商没找到解”。

对每个摆放：先按原流程协商布线（结果存盘，重跑时直接读取），再逐个处理校验器报出的冲突：
  其他管网全部固定，它们的占用晕设为硬障碍（不加价、不协商），只对冲突管网单独重搜，扩展上限放大。
  管–管冲突 X 与 Y 依次尝试：只重布 X → 只重布 Y → 两根都拆掉后先 X 后 Y → 先 Y 后 X。
  某次尝试后整体用独立校验器 check_routes 复核，冲突数不增加且该冲突消失才接受。
单管搜索的结果分三种：
  找到路径；搜索空间耗尽（在当前网格与其他管道位置下确实没有路径）；超过扩展上限（不能下结论）。

注：negotiate 现已自带清理；本脚本只在读取已存协商结果（清理前）时才重现本次检查。

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
    sc.rp["cleanup_max_expansions"] = max_exp
    G = rt.Grid(sc)
    viol0, _ = rt.check_routes(sc, routes)
    cl0 = rt.clashes(sc, routes)
    print(f"协商后：违规 {len(viol0)}，其中冲突 {len(cl0)}：" + "；".join(v for *_, v in cl0), flush=True)
    t_all = time.time()
    routes, records = rt.cleanup(G, sc, routes, log=lambda x: print(x, flush=True))
    cl1 = rt.clashes(sc, routes)
    viol1, met = rt.check_routes(sc, routes)
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
