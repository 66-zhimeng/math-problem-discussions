"""少协商、多清理：协商轮数上限取不同值（之后都做清理），比较精细布管总用时与结果。

运行：.venv/Scripts/python 协商轮数对比.py [方法字母，默认 B] [轮数列表，默认 1,3]
输出：协商轮数对比结果.json；日志重定向到 协商轮数对比_log.txt
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
import 布管实验 as ex  # noqa: E402

rt = ex.rt


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pick = sys.argv[1] if len(sys.argv) > 1 else "B"
    iters = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,3").split(",")]
    d = json.loads((ex.bk.HERE / "设备算例_小.json").read_text(encoding="utf-8"))
    d["params"]["blocking"]["auto_module_policy"] = "report_only"
    inst = ex.bk.load_devices(d)
    cases = json.loads((HERE / "联合验证结果.json").read_text(encoding="utf-8")) + \
        json.loads((HERE / "联合验证结果_C.json").read_text(encoding="utf-8"))
    rows = []
    for c in [c for c in cases if c["method"][0] in pick]:
        placed = {k: tuple(v) for k, v in c["placement"].items()}
        blocks, nets, P, sol = ex.device_level(inst, placed)
        for m in iters:
            sc = ex.scene(inst, blocks, nets, P, sol)
            sc.rp = {**sc.rp, "max_iters": m}
            t0 = time.time()
            routes, hist, _ = rt.negotiate(sc, log=lambda x: None)
            el = round(time.time() - t0, 1)
            viol, met = rt.check_routes(sc, routes)
            cl = hist[-1]["cleanup"]
            ok = not viol and len(routes) == len(sc.nets)
            n_fix = sum(1 for r in cl["records"] if r["result"] == "已消除")
            n_left = sum(1 for r in cl["records"] if r["result"] == "未消除")
            print(f"{c['method']} 种子 {c['seed']}，协商 {m} 轮：总 {el} s（清理 {cl['time_s']} s，冲突 {len(cl['records'])} 处，"
                  f"消除 {n_fix}，未消除 {n_left}），布通 {len(routes)}/{len(sc.nets)}，违规 {len(viol)}，"
                  f"管长 {met['L_m']} m，弯头 {met['bends']}，J={met['J']}" + ("  【全部无违规】" if ok else ""), flush=True)
            rows.append({"method": c["method"], "seed": c["seed"], "max_iters": m, "route_s": el,
                         "cleanup_s": cl["time_s"], "cleanup": cl["records"], "routed": len(routes),
                         "violations": viol, "all_ok": ok, "metrics": {k: v for k, v in met.items() if k != "per_net"}})
            (HERE / "协商轮数对比结果.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
