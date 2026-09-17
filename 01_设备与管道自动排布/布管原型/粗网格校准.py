"""校准：在已有 9 个摆放（A/B/C 各 3 种子）上跑粗网格布管，与精细布管结果对照。

运行：.venv/Scripts/python 粗网格校准.py [格子尺寸 mm，默认取算例参数]
输出：粗网格校准结果.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
import 布管实验 as ex  # noqa: E402
import coarse_route as cr  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def main():
    d = json.loads((ex.bk.HERE / "设备算例_小.json").read_text(encoding="utf-8"))
    d["params"]["blocking"]["auto_module_policy"] = "report_only"
    inst = ex.bk.load_devices(d)
    cases = json.loads((HERE / "联合验证结果.json").read_text(encoding="utf-8")) + \
        json.loads((HERE / "联合验证结果_C.json").read_text(encoding="utf-8"))
    rows = []
    for c in cases:
        placed = {k: tuple(v) for k, v in c["placement"].items()}
        blocks, nets, P, sol = ex.device_level(inst, placed)
        sc = ex.scene(inst, blocks, nets, P, sol)
        if len(sys.argv) > 1:                                         # 可覆盖格子尺寸做对比
            sc.rp["coarse_cell_mm"] = int(sys.argv[1])
        r = cr.coarse_route(sc)
        fine = f"精细：布通 {c['routed']}/{c['nets']}，冲突对 {c['clash_pairs']}，冲突边 {c['final_conflict_edges']}，用时 {c['route_s']} s"
        print(f"{c['method']} 种子 {c['seed']}\n  {fine}\n  粗网格：{'可布' if r['ok'] else '不可布'}，"
              f"未布通 {len(r['failed'])} {r['failed'] if r['failed'] else ''}，超容格点 {r['overflow_cells']}"
              f"（{len(r['overflow_nets'])} 个管网 {r['overflow_nets']}），轮数 {r['iters']}，用时 {r['time_s']} s，"
              f"网格 {r['grid']}", flush=True)
        rows.append({"method": c["method"], "seed": c["seed"], "fine": {k: c[k] for k in (
            "routed", "nets", "clash_pairs", "final_conflict_edges", "route_s")}, "coarse": r})
    (HERE / "粗网格校准结果.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
