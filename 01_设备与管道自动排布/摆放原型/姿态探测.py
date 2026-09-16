"""第 3 步：按下界删除姿态（探测）。

对每个块的每个姿态：固定该姿态，其余不变，CP-SAT 短时求解得到有效下界 LB。
若 LB > 已知最好解 J_best，则该姿态不可能出现在最优解中，可删除（证明来自 CP-SAT 下界的有效性）。
运行：.venv/Scripts/python 姿态探测.py [每次探测秒数，默认 2] [J_best，默认取下界对比结果中的最好值]
"""
import json
import sys

import placement_cpsat as base
import placement_fast as fast

T = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
rows = json.loads((base.HERE / "下界对比结果.json").read_text(encoding="utf-8"))
J_BEST = float(sys.argv[2]) if len(sys.argv) > 2 else min(r["J_final"] for r in rows if r["lb"] == "v2")

blocks, nets, P = base.load(base.HERE / "算例.json")
out, pruned = [], 0
for i, b in enumerate(blocks):
    if len(b["poses"]) == 1:
        continue
    for k, p in enumerate(b["poses"]):
        rb = [dict(bb) for bb in blocks]
        rb[i] = {**b, "poses": [p]}
        r = fast.solve(rb, nets, P, P["w"], T, J_ub=J_BEST, lb="v2")
        lb = r["bound"] if r else None           # 无解：在 J ≤ J_best 范围内不可行，同样可删
        cut = (r is None) or (lb > J_BEST)
        pruned += cut
        out.append({"block": b["id"], "variant": p["variant"], "rot": p["rot"],
                    "bound": None if lb is None else round(lb, 4), "prunable": cut})
        print(out[-1], flush=True)
print(f"\nJ_best={J_BEST}  探测 {len(out)} 个姿态，可删除 {pruned} 个")
(base.HERE / "姿态探测结果.json").write_text(json.dumps({"J_best": J_BEST, "probe_s": T, "rows": out},
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
