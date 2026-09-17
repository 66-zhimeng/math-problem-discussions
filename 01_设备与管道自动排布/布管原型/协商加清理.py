"""验证 negotiate 新流程：协商（连续 stall_iters 轮无改进即停）+ 清理（其他管网作硬障碍，单独重布冲突管网）。

在已有 B、C 摆放（各 3 种子）上重跑精细布管，与旧流程（固定 30 轮、无清理）的结果对照。
运行：.venv/Scripts/python 协商加清理.py [方法字母，默认 BC]
输出：协商加清理结果.json；日志重定向到 协商加清理_log.txt
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
import 布管实验 as ex  # noqa: E402

rt = ex.rt
sys.stdout.reconfigure(encoding="utf-8")


def main():
    pick = sys.argv[1] if len(sys.argv) > 1 else "BC"
    d = json.loads((ex.bk.HERE / "设备算例_小.json").read_text(encoding="utf-8"))
    d["params"]["blocking"]["auto_module_policy"] = "report_only"
    inst = ex.bk.load_devices(d)
    cases = json.loads((HERE / "联合验证结果.json").read_text(encoding="utf-8")) + \
        json.loads((HERE / "联合验证结果_C.json").read_text(encoding="utf-8"))
    rows = []
    for c in [c for c in cases if c["method"][0] in pick]:
        placed = {k: tuple(v) for k, v in c["placement"].items()}
        blocks, nets, P, sol = ex.device_level(inst, placed)
        sc = ex.scene(inst, blocks, nets, P, sol)
        print(f"\n===== {c['method']} 种子 {c['seed']}（旧流程：{c['route_s']} s，{c['iterations']} 轮，"
              f"布通 {c['routed']}/{c['nets']}，冲突对 {c['clash_pairs']}）=====", flush=True)
        t0 = time.time()
        routes, hist, _ = rt.negotiate(sc, log=lambda x: print(x, flush=True))
        el = round(time.time() - t0, 1)
        viol, met = rt.check_routes(sc, routes)
        cl = hist[-1]["cleanup"]
        n_neg = sum(h["time_s"] for h in hist)
        print(f"新流程：{el} s（协商 {len(hist)} 轮 {n_neg:.0f} s，清理 {cl['time_s']} s），布通 {len(routes)}/{len(sc.nets)}，"
              f"违规 {len(viol)}；管长 {met['L_m']} m，弯头 {met['bends']}，高度变化 {met['height_changes']}，J={met['J']}"
              + ("  【全部无违规】" if not viol and len(routes) == len(sc.nets) else ""), flush=True)
        for v in viol:
            print(f"    {v}")
        rows.append({"method": c["method"], "seed": c["seed"],
                     "old": {k: c[k] for k in ("route_s", "iterations", "routed", "clash_pairs", "all_ok")},
                     "new": {"route_s": el, "iterations": len(hist), "cleanup_s": cl["time_s"],
                             "cleanup": cl["records"], "routed": len(routes), "nets": len(sc.nets),
                             "violations": viol, "all_ok": not viol and len(routes) == len(sc.nets),
                             "metrics": {k: v for k, v in met.items() if k != "per_net"}},
                     "conflict_history": [h["conflict_edges"] for h in hist]})
        (HERE / "协商加清理结果.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n汇总：")
    for r in rows:
        o, n = r["old"], r["new"]
        print(f"  {r['method']} 种子 {r['seed']}：旧 {o['route_s']} s / {o['iterations']} 轮 / 冲突对 {o['clash_pairs']}"
              f" → 新 {n['route_s']} s / {n['iterations']} 轮 / 违规 {len(n['violations'])}"
              f"{'（全部无违规）' if n['all_ok'] else ''}")


if __name__ == "__main__":
    main()
