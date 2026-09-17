"""方案一：用粗网格路径作走廊引导精细布管，判断剩余冲突是“空间不够”还是“精细协商没找到解”。

在已有摆放上对比：同一摆放、同样布管参数，精细布管 不引导（已有结果）vs 走廊引导。
运行：.venv/Scripts/python 走廊引导.py [方法前缀，默认 B] [走廊扩展格数，默认 1] [走出走廊代价倍数，默认 3]
输出：走廊引导结果_{方法}.json、日志重定向到 走廊引导_log.txt
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
import 布管实验 as ex  # noqa: E402
import coarse_route as cr  # noqa: E402

rt = ex.rt
sys.stdout.reconfigure(encoding="utf-8")


def main():
    pick = sys.argv[1] if len(sys.argv) > 1 else "B"
    dil = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    pen = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    d = json.loads((ex.bk.HERE / "设备算例_小.json").read_text(encoding="utf-8"))
    d["params"]["blocking"]["auto_module_policy"] = "report_only"
    inst = ex.bk.load_devices(d)
    cases = json.loads((HERE / "联合验证结果.json").read_text(encoding="utf-8")) + \
        json.loads((HERE / "联合验证结果_C.json").read_text(encoding="utf-8"))
    cases = [c for c in cases if c["method"].startswith(pick)]
    rows = []
    for c in cases:
        placed = {k: tuple(v) for k, v in c["placement"].items()}
        blocks, nets, P, sol = ex.device_level(inst, placed)
        sc = ex.scene(inst, blocks, nets, P, sol)
        coarse = cr.coarse_route(sc)
        cells = {nid: cr.dilate(coarse["spec"], cs, dil) for nid, cs in coarse["paths"].items()}
        print(f"\n===== {c['method']} 种子 {c['seed']}：粗网格 {'可布' if coarse['ok'] else '不可布'}，"
              f"{coarse['time_s']} s；不引导时精细结果：布通 {c['routed']}/{c['nets']}，冲突对 {c['clash_pairs']}，"
              f"冲突边 {c['final_conflict_edges']} =====", flush=True)
        t0 = time.time()
        routes, hist, G = rt.negotiate(sc, log=lambda x: print(x, flush=True),
                                       corridors={"spec": coarse["spec"], "cells": cells, "penalty": pen})
        el = time.time() - t0
        viol, met = rt.check_routes(sc, routes)
        failed = hist[-1]["failed"]
        clash = [v for v in viol if "净距不足" in v or "穿过" in v]
        ok = not failed and not viol
        print(f"走廊引导后：{el:.0f} s，{len(hist)} 轮，布通 {len(routes)}/{len(sc.nets)}，冲突对 {len(clash)}，"
              f"最后冲突边 {hist[-1]['conflict_edges']}；{'【全部布通且无违规】' if ok else '【未达成】'}")
        print(f"  含管道占地 {met['area_m2']} m²，管长 {met['L_m']} m，弯头 {met['bends']}，高度变化 {met['height_changes']}，"
              f"J={met['J']}" + ("" if ok else "（未完整）"))
        if clash:
            print("  冲突：" + "；".join(clash))
        rows.append({"method": c["method"], "seed": c["seed"], "dilate": dil, "penalty": pen,
                     "unguided": {k: c[k] for k in ("routed", "clash_pairs", "final_conflict_edges", "route_s")},
                     "guided": {"routed": len(routes), "clash_pairs": len(clash), "conflict_edges": hist[-1]["conflict_edges"],
                                "all_ok": ok, "route_s": round(el, 1), "iterations": len(hist), "violations": viol,
                                "metrics": {k: v for k, v in met.items() if k != "per_net"}},
                     "conflict_history": [h["conflict_edges"] for h in hist]})
        (HERE / f"走廊引导结果_{pick}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n汇总：")
    for r in rows:
        u, g = r["unguided"], r["guided"]
        print(f"  {r['method']} 种子 {r['seed']}：不引导 冲突对 {u['clash_pairs']}（{u['route_s']} s）→ 引导 冲突对 "
              f"{g['clash_pairs']}，{'成功' if g['all_ok'] else '未达成'}（{g['route_s']} s，{g['iterations']} 轮）")


if __name__ == "__main__":
    main()
