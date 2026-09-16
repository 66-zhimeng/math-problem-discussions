"""第 1、2 步效果：提速模型分别使用 v1 / v2 两端点下界，所有解统一按 v2 定义评分。

v1 模型的下界是 J_v1 的下界；因 J_v1 ≤ J_v2 逐点成立，它也是 J_v2 最优值的有效下界，可直接比较差距。
运行：.venv/Scripts/python 下界对比.py [预算秒数，默认 60] [种子数，默认 3]
"""
import json
import statistics
import sys
import time

import placement_cpsat as base
import placement_fast as fast

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
CHECK = [t for t in (5, 10, 20, 30, 60, 120, 300) if t <= BUDGET]


def main():
    blocks, nets, P = base.load(base.HERE / "算例.json")
    w = P["w"]
    rows = []
    for lb in ("v1", "v2"):
        for seed in range(SEEDS):
            tr = []
            t0 = time.time()
            r = fast.run(blocks, nets, P, w, BUDGET, 5, tr, seed, lb)
            scored = []
            for t, _, bnd, snap in tr:
                scored.append((t, base.validate(blocks, nets, P, snap, w, "v2")["J_full_weights"]))
            final = base.validate(blocks, nets, P, r["solution"], w, "v2")["J_full_weights"]
            best = min([final] + [j for _, j in scored])
            row = {"lb": lb, "seed": seed, "time_s": round(time.time() - t0, 1),
                   **{f"J@{c}s": round(min([j for t, j in scored if t <= c], default=float("nan")), 4) for c in CHECK},
                   "J_final": round(best, 4), "bound": round(r["bound"], 4),
                   "gap_pct": round((best - r["bound"]) / best * 100, 1), "status": r["status"]}
            rows.append(row)
            print(row, flush=True)
            (base.HERE / "下界对比结果.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n中位数（按 v2 定义评分）：")
    for lb in ("v1", "v2"):
        rs = [r for r in rows if r["lb"] == lb]
        keys = [f"J@{c}s" for c in CHECK] + ["J_final", "gap_pct"]
        print(lb, {k: statistics.median([r[k] for r in rs]) for k in keys})


if __name__ == "__main__":
    main()
