"""块级摆放原型：大邻域搜索（LNS）+ CP-SAT 精确子问题。问题定义与 placement_cpsat.py 相同。

流程
  1. 初始解：粗网格 CP-SAT（placement_fast.run 的第一阶段）
  2. 循环直到时间用完：
       选邻域：随机一个管网的全部块 / 某块及其空间最近邻 / 随机若干块
       其余块的位置与姿态固定，只对邻域内的块建 CP-SAT 子问题（含 W、H、全部管网下界）
       子问题以当前解为提示，短时间求解；目标严格下降则接受
  3. 每次接受后用独立校验器重算目标，保证与完整模型一致

只保证“持续改进的可行解”，不给最优性下界。

运行：.venv/Scripts/python placement_lns.py [总秒数，默认 60] [随机种子，默认 0]
"""
import json
import random
import sys
import time

import placement_cpsat as base
import placement_fast as fast

TOTAL = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0


def restrict(blocks, sol, free):
    """非邻域块：固定位置与姿态；复制组若有固定成员，自由成员只能选同一内部排法。"""
    fixed_variant = {}
    for i, b in enumerate(blocks):
        if i not in free and b["copy_group"]:
            fixed_variant[b["copy_group"]] = b["poses"][sol[i]["pose"]]["variant"]
    rb, pose_map = [], []
    for i, b in enumerate(blocks):
        if i in free and not b["fixed"]:
            idx = [k for k, p in enumerate(b["poses"])
                   if b["copy_group"] not in fixed_variant or p["variant"] == fixed_variant[b["copy_group"]]]
            rb.append({**b, "poses": [b["poses"][k] for k in idx]})
        else:
            idx = [sol[i]["pose"]]
            rb.append({**b, "poses": [b["poses"][idx[0]]], "fixed": (sol[i]["x"], sol[i]["y"]),
                       "copy_group": None})
        pose_map.append(idx)
    # 自由成员与固定成员同组时，已通过姿态过滤保证一致；全部自由的组仍由 copy_group 约束
    return rb, pose_map


def neighborhood(rng, blocks, nets, sol, size):
    movable = [i for i, b in enumerate(blocks) if not b["fixed"]]
    kind = rng.random()
    groups = {}
    for i in movable:
        if blocks[i]["copy_group"]:
            groups.setdefault(blocks[i]["copy_group"], []).append(i)
    if groups and kind < 0.15:
        # 整个复制组 + 与其相连的块，允许切换组内部排法
        members = set(rng.choice(list(groups.values())))
        linked = {i for e in nets if members & {t for t, _ in e["terms"]} for i, _ in e["terms"]
                  if not blocks[i]["fixed"] and len(e["terms"]) == 2}
        free = members | set(rng.sample(sorted(linked - members), min(2, len(linked - members))))
        return free
    if kind < 0.45:
        e = rng.choice(nets)
        free = {i for i, _ in e["terms"] if not blocks[i]["fixed"]}
    elif kind < 0.8:
        c = rng.choice(movable)
        cx, cy = sol[c]["x"], sol[c]["y"]
        near = sorted(movable, key=lambda j: abs(sol[j]["x"] - cx) + abs(sol[j]["y"] - cy))
        free = set(near[:size])
    else:
        free = set(rng.sample(movable, min(size, len(movable))))
    while len(free) > size + 2:
        free.discard(rng.choice(sorted(free)))
    return free


def run_lns(blocks, nets, P, weights, total, seed=0, trace=None):
    rng = random.Random(seed)
    t0 = time.time()
    tr0 = []
    cb, cP = fast.coarsen(blocks, P, 5)
    rc = fast.solve(cb, nets, cP, weights, min(5.0, total * 0.2), seed=seed)
    sol = [{"x": h["x"] * 5, "y": h["y"] * 5, "pose": h["pose"]} for h in rc["solution"]]
    J = base.validate(blocks, nets, P, sol, weights)["J_full_weights"]
    if trace is not None:
        trace.append((time.time() - t0, J))
    size, iters, accepted, stall = 4, 0, 0, 0
    n_mov = sum(1 for b in blocks if not b["fixed"])
    while time.time() - t0 < total - 0.3:
        free = neighborhood(rng, blocks, nets, sol, size)
        rb, pmap = restrict(blocks, sol, free)
        hint = [{"x": sol[i]["x"], "y": sol[i]["y"], "pose": pmap[i].index(sol[i]["pose"])}
                for i in range(len(blocks))]
        remain = total - (time.time() - t0)
        r = fast.solve(rb, nets, P, weights, min(0.5 * len(free), remain), hint=hint, J_ub=J + 1e-9, seed=seed + iters)
        iters += 1
        improved = False
        if r:
            cand = [{"x": s_["x"], "y": s_["y"], "pose": pmap[i][s_["pose"]]} for i, s_ in enumerate(r["solution"])]
            Jc = base.validate(blocks, nets, P, cand, weights)["J_full_weights"]
            if Jc < J - 1e-6:
                sol, J = cand, Jc
                accepted += 1
                improved = True
                if trace is not None:
                    trace.append((time.time() - t0, J))
        # 自适应邻域：连续 8 次无改进则扩大，有改进则缩回
        stall = 0 if improved else stall + 1
        if improved:
            size = 4
        elif stall >= 8:
            size, stall = min(size + 2, n_mov), 0
    return sol, J, iters, accepted


def main():
    blocks, nets, P = base.load(base.HERE / "算例.json")
    trace = []
    sol, J, iters, acc = run_lns(blocks, nets, P, P["w"], TOTAL, SEED, trace)
    v = base.validate(blocks, nets, P, sol, P["w"])
    print(f"LNS 用时 {TOTAL} s  子问题 {iters} 次  接受 {acc} 次  J={J:.4f}")
    print(f"占地 {v['area_m2']} m²  管长下界 {v['L_lb_m']} m  弯头下界 {v['bends_lb']}")
    print("时间—目标：", [(round(t, 1), round(o, 4)) for t, o in trace])


if __name__ == "__main__":
    main()
