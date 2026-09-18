"""统一接口：一份任务书进，一份方案出。

    from layout_kernel import solve
    result = solve(case)            # case 为 dict 或 JSON 文件路径

两种模式（task.mode）：
  place_and_route  给拓扑与设备清单，求设备位置 + 管道。搜索若干候选摆放，
                   对每个候选**完整布管**，按独立校验器算出的真实 J 选最好的。
  route_only       设备位置已给定（devices[].placed），只布管。

结果里的 J、指标与违规全部来自独立校验器（routing.check_routes / placement_cpsat.validate），
不依赖求解器内部数据。

注意：place_and_route 用多进程（spawn），调用方须放在 if __name__ == "__main__": 之内。
"""
import json
import tempfile
import time
from pathlib import Path

from . import build
from .contract import CaseError, validate_case

bk, base, rt = build.bk, build.base, build.rt
import placement_sp as sp  # noqa: E402


def _dump_routes(routes):
    return {nid: [{"start": list(b["start"]),
                   "end": [b["end"][0], list(b["end"][1]) if b["end"][1] else None],
                   "points_mm": [[float(c) for c in p] for p in b["points"]]} for b in brs]
            for nid, brs in routes.items()}


def _dump_devices(inst, sc, placed, grid):
    out = {}
    for did, (x, y, r) in placed.items():
        d = sc.dev[did]
        out[did] = {"x_mm": x * grid, "y_mm": y * grid, "rot_deg": r,
                    "box_mm": [float(c) for c in d["box"]],
                    "ports_mm": {pn: [float(c) for c in v[:3]] for pn, v in d["ports"].items()}}
    return out


def _route(inst, placed, keepout, log):
    """给定设备位置 → 布管 → 独立校验。返回一份结果片段。"""
    blocks, nets, P, sol = build.device_level(inst, placed)
    lb = base.validate(blocks, nets, P, sol, P["w"], "v2")
    sc = build.scene(inst, blocks, nets, P, sol, keepout)
    t0 = time.time()
    routes, history, _G = rt.negotiate(sc, log=log)
    route_s = time.time() - t0
    viol, met = rt.check_routes(sc, routes)
    failed = history[-1]["failed"]
    ok = not viol and len(routes) == len(sc.nets)
    return {"ok": ok, "violations": viol, "unrouted": failed,
            "metrics": {k: v for k, v in met.items() if k != "per_net"},
            "per_net": met["per_net"],
            "lower_bound": {k: lb[k] for k in ("area_m2", "W_m", "H_m", "L_lb_m", "bends_lb", "J_full_weights")},
            "route_s": round(route_s, 1), "route_iters": len(history),
            "devices": _dump_devices(inst, sc, placed, P["grid"]), "routes": _dump_routes(routes)}


def solve(case, log=None, work_dir=None):
    """求解一份任务书，返回结果 dict（见 layout_kernel/契约.md 第 3 节）。"""
    if isinstance(case, (str, Path)):
        case = json.loads(Path(case).read_text(encoding="utf-8"))
    log = log or (lambda _line: None)
    t_all = time.time()
    inst, task, keepout = validate_case(case)
    grid = inst["P"]["grid_mm"]
    if task["mode"] == "route_only":
        placed = {d["id"]: (d["placed"][0] // grid, d["placed"][1] // grid, d["placed"][2])
                  for d in case["devices"]}
        log(f"只布管：{len(placed)} 台设备、{len(inst['nets'])} 个管网")
        out = _route(inst, placed, keepout, log)
        out.update(mode="route_only", candidates=[], total_s=round(time.time() - t_all, 1))
        return out

    modules, auto_report = bk.find_modules(inst)
    bdata, _owner, _stats = bk.build_block_instance(inst, modules)
    tmp = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="layout_kernel_"))
    tmp.mkdir(parents=True, exist_ok=True)
    bpath = tmp / "_块级算例.json"
    bpath.write_text(json.dumps(bdata, ensure_ascii=False), encoding="utf-8")
    margins = build.placement_margins(inst["P"]["routing"])
    log(f"摆放 + 布管：{len(inst['devs'])} 台设备 → {len(bdata['blocks'])} 个块"
        f"（模块 {len(modules)} 个，自动候选 {len(auto_report)} 个），"
        f"{task['candidates']} 个候选 × {task['time_budget_s']} s")
    cands, best = [], None
    for k in range(task["candidates"]):
        seed = task["seed"] + k
        t0 = time.time()
        r = sp.run(bpath, task["time_budget_s"], task["workers"], seed, margins=margins)
        if r["solution"] is None:
            log(f"  候选 {k}（种子 {seed}）：摆放没有找到可行解")
            cands.append({"seed": seed, "ok": False, "reason": "摆放无可行解"})
            continue
        placed = build.expand(inst, modules, bdata, base.load(bpath)[0], r["solution"])
        one = _route(inst, placed, keepout, lambda _l: None)
        one["placement_s"] = round(time.time() - t0 - one["route_s"], 1)
        m = one["metrics"]
        state = "无违规" if one["ok"] else f"违规 {len(one['violations'])}"
        log(f"  候选 {k}（种子 {seed}）：{state}，J={m['J']}，占地 {m['area_m2']} m²，"
            f"管长 {m['L_m']} m，布管 {one['route_s']} s")
        cands.append({"seed": seed, "ok": one["ok"], "J": m["J"], "area_m2": m["area_m2"],
                      "L_m": m["L_m"], "bends": m["bends"], "height_changes": m["height_changes"],
                      "violations": len(one["violations"]), "route_s": one["route_s"],
                      "placement_s": one["placement_s"]})
        key = (0 if one["ok"] else 1, m["J"])
        if best is None or key < best[0]:
            best = (key, one, seed)
    if best is None:
        why = "所有候选摆放都没有可行解"
        n_fixed = sum(1 for d in case["devices"] if "fixed" in d)
        if n_fixed:
            why += (f"（有 {n_fixed} 台设备给了 fixed 坐标：摆放会在设备四周为出管预留空间，"
                    "固定坐标太靠近原点或彼此太近时无解，可把坐标整体挪开或放宽）")
        return {"ok": False, "mode": "place_and_route", "violations": [why], "candidates": cands,
                "metrics": {}, "devices": {}, "routes": {}, "unrouted": {},
                "total_s": round(time.time() - t_all, 1)}
    out = best[1]
    out.update(mode="place_and_route", chosen_seed=best[2], candidates=cands,
               total_s=round(time.time() - t_all, 1))
    log(f"选中种子 {best[2]}：J={out['metrics']['J']}，{'无违规' if out['ok'] else '仍有违规'}")
    return out
