"""提速对比：原模型 / 提速模型 / LNS，在相同时间预算下多种子比较。

记录每个方法在 10、20、30、60 秒时的最好目标 J（完整权重，越小越好）。
运行：.venv/Scripts/python 提速对比.py [预算秒数，默认 60] [种子数，默认 3]
"""
import json
import statistics
import sys
import time

import placement_cpsat as base
import placement_fast as fast
import placement_lns as lns

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
CHECK = [t for t in (10, 20, 30, 60, 120, 300) if t <= BUDGET]


def best_at(trace, t):
    vals = [o for tt, o in trace if tt <= t]
    return min(vals) if vals else None


def main():
    blocks, nets, P = base.load(base.HERE / "算例.json")
    w = P["w"]
    methods = {
        "原模型(12线程)": lambda seed, tr: base.build_and_solve(blocks, nets, P, w, BUDGET, "", tr, 12, seed),
        "提速模型": lambda seed, tr: fast.run(blocks, nets, P, w, BUDGET, 5, tr, seed),
        "LNS": lambda seed, tr: lns.run_lns(blocks, nets, P, w, BUDGET, seed, tr),
    }
    rows = []
    for name, fn in methods.items():
        for seed in range(SEEDS):
            tr = []
            t0 = time.time()
            fn(seed, tr)
            tr = [(t, o) for t, o, *_ in tr]
            row = {"method": name, "seed": seed, "time_s": round(time.time() - t0, 1),
                   **{f"J@{t}s": (round(best_at(tr, t), 4) if best_at(tr, t) is not None else None) for t in CHECK}}
            rows.append(row)
            print(row, flush=True)
            (base.HERE / "提速对比结果.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n中位数：")
    for name in methods:
        rs = [r for r in rows if r["method"] == name]
        med = {f"J@{t}s": statistics.median([r[f"J@{t}s"] for r in rs if r[f"J@{t}s"] is not None] or [float("nan")])
               for t in CHECK}
        print(name, {k: round(v, 4) for k, v in med.items()})


if __name__ == "__main__":
    main()
