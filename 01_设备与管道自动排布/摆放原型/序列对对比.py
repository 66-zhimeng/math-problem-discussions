"""序列对搜索 vs 提速 CP-SAT 模型：同一算例、同一机器、同样预算，统一按 v2 定义用校验器评分。

方法：
  fast      placement_fast.run(lb="v2")，12 线程
  sp        placement_sp.run，12 进程，冷却周期 10 s（周期结束从全局最好解重新加热）
  sp_nocyc  placement_sp.run，12 进程，冷却周期 = 总预算（不重新加热）
时间从调用开始计（含 sp 的进程启动开销）。

运行：.venv/Scripts/python 序列对对比.py [预算秒数，默认 60] [种子数，默认 3]
"""
import json
import statistics
import sys
import time

import placement_cpsat as base
import placement_fast as fast
import placement_sp as sp

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
CHECK = [t for t in (5, 10, 20, 30, 60, 120, 300) if t <= BUDGET]
PATH = base.HERE / "算例.json"


def at(curve, c):
    return min([j for t, j in curve if t <= c], default=float("nan"))


def run_fast(blocks, nets, P, seed):
    tr = []
    r = fast.run(blocks, nets, P, P["w"], BUDGET, 5, tr, seed, "v2")
    curve = [(t, base.validate(blocks, nets, P, snap, P["w"], "v2")["J_full_weights"]) for t, _, _, snap in tr]
    final = base.validate(blocks, nets, P, r["solution"], P["w"], "v2")["J_full_weights"]
    return curve, final, {"bound": round(r["bound"], 4), "status": r["status"]}


def run_sp(blocks, nets, P, seed, cycle):
    r = sp.run(PATH, BUDGET, 12, seed, cycle=cycle)
    final = base.validate(blocks, nets, P, r["solution"], P["w"], "v2")["J_full_weights"]   # 独立复核
    assert abs(final - r["J"]) < 1e-9
    return r["curve"], final, {"stats": r["stats"]}


def main():
    blocks, nets, P = base.load(PATH)
    methods = {"fast": lambda s: run_fast(blocks, nets, P, s),
               "sp": lambda s: run_sp(blocks, nets, P, s, 10.0),
               "sp_nocyc": lambda s: run_sp(blocks, nets, P, s, BUDGET)}
    rows = []
    for seed in range(SEEDS):
        for name, fn in methods.items():
            t0 = time.time()
            curve, final, extra = fn(seed)
            best = min([final] + [j for _, j in curve])
            row = {"method": name, "seed": seed, "time_s": round(time.time() - t0, 1),
                   "first_solution_s": round(curve[0][0], 2) if curve else None,
                   **{f"J@{c}s": round(at(curve, c), 4) for c in CHECK},
                   "J_final": round(best, 4), **extra}
            rows.append(row)
            print(row, flush=True)
            (base.HERE / "序列对对比结果.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
    print("\n中位数 / 最好（按 v2 定义评分）：")
    for name in methods:
        rs = [r for r in rows if r["method"] == name]
        keys = [f"J@{c}s" for c in CHECK] + ["J_final"]
        print(name, {k: statistics.median([r[k] for r in rs]) for k in keys},
              "最好", min(r["J_final"] for r in rs))


if __name__ == "__main__":
    main()
