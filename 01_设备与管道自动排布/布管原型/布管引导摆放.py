"""摆放搜索中加入粗网格布管（placement_sp.run(..., guide=...)），与只做出管留空（B）对照。

D = B（按侧出管留空）+ 布管引导：退火中每隔 guide_every_s 秒对当前解做粗网格布管，
  按粗网格管长更新各管网长度权重，全局最好解按“粗网格可布 + 粗网格管长”的目标选择。
两种方法的摆放都接同一个精细布管流程（协商 + 清理），以独立校验器 check_routes 的结果比较。
B 的精细布管结果取自 协商加清理结果.json（同一流程）。

运行：.venv/Scripts/python 布管引导摆放.py [摆放秒数，默认 60] [种子数，默认 3] [fine|coarse，默认 fine]
  coarse：只做粗网格对比（B 与 D 都算 J_route），不做精细布管
输出：布管引导摆放结果.json；日志重定向到 布管引导摆放_log.txt
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
import 布管实验 as ex  # noqa: E402
import coarse_route as cr  # noqa: E402

rt, sp, base, bk = ex.rt, ex.sp, ex.base, ex.bk
CASE = bk.HERE / "设备算例_小.json"
BPATH = HERE / "_设备级块算例.json"
GUIDE_KEYS = ["guide_every_s", "guide_ema", "guide_w_max"]
_CACHE = {}


def load_inst():
    d = json.loads(CASE.read_text(encoding="utf-8"))
    d["params"]["blocking"]["auto_module_policy"] = "report_only"          # 设备级：不合并模块
    return bk.load_devices(d)


def _ctx():
    if not _CACHE:
        inst = load_inst()
        bdata, _, _ = bk.build_block_instance(inst, [])
        blocks, _, _ = base.load(BPATH)
        _CACHE.update(inst=inst, bdata=bdata, blocks=blocks)
    return _CACHE


def to_scene(sol):
    c = _ctx()
    placed = ex.expand(c["inst"], [], c["bdata"], c["blocks"], sol)
    blocks, nets, P, dsol = ex.device_level(c["inst"], placed)
    return placed, blocks, nets, P, dsol, ex.scene(c["inst"], blocks, nets, P, dsol)


def coarse_eval(sol):
    """guide["fn"]：块级解 → 粗网格布管结果（在退火工作进程中调用）。"""
    *_, sc = to_scene(sol)
    r = cr.coarse_route(sc)
    cell_m = sc.rp["coarse_cell_mm"] / 1000
    L_net = {nid: len(cells) * cell_m for nid, cells in r["paths"].items()}
    return {"ok": r["ok"], "L_m": sum(L_net.values()), "L_net_m": L_net, "time_s": r["time_s"]}


def fine(sc):
    t0 = time.time()
    routes, hist, _ = rt.negotiate(sc, log=lambda x: print(x, flush=True))
    el = round(time.time() - t0, 1)
    viol, met = rt.check_routes(sc, routes)
    ok = not viol and len(routes) == len(sc.nets)
    print(f"精细布管 {el} s（{len(hist)} 轮，清理 {hist[-1]['cleanup']['time_s']} s）：布通 {len(routes)}/{len(sc.nets)}，"
          f"违规 {len(viol)}；占地 {met['area_m2']} m²，管长 {met['L_m']} m，弯头 {met['bends']}，"
          f"高度变化 {met['height_changes']}，J={met['J']}" + ("  【全部无违规】" if ok else ""), flush=True)
    for v in viol:
        print(f"    {v}")
    return {"route_s": el, "iterations": len(hist), "routed": len(routes), "violations": viol, "all_ok": ok,
            "metrics": {k: v for k, v in met.items() if k != "per_net"}}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    seeds = range(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    do_fine = (sys.argv[3] if len(sys.argv) > 3 else "fine") == "fine"
    inst = load_inst()
    rp = inst["P"]["routing"]
    miss = [k for k in GUIDE_KEYS if k not in rp]
    if miss:
        raise KeyError(f"缺少布管引导参数：{miss}")
    bdata, _, _ = bk.build_block_instance(inst, [])
    BPATH.write_text(json.dumps(bdata, ensure_ascii=False), encoding="utf-8")
    margins = {"D_mm": rp["D_default_mm"], "delta_pp_mm": rp["delta_pp_mm"], "delta_ep_mm": rp["delta_ep_mm"],
               "c_rho": rp["c_rho"]}
    guide = {"fn": coarse_eval, "every_s": rp["guide_every_s"], "ema": rp["guide_ema"],
             "w_max": rp["guide_w_max"]}
    b_rows = {r["seed"]: r for r in json.loads((HERE / "协商加清理结果.json").read_text(encoding="utf-8"))
              if r["method"].startswith("B")}
    b_place = {c["seed"]: c for c in json.loads((HERE / "联合验证结果.json").read_text(encoding="utf-8"))
               if c["method"].startswith("B")}
    rows = []
    for seed in seeds:
        print(f"\n===== D出管留空+布管引导，种子 {seed} =====", flush=True)
        r = sp.run(BPATH, budget, 12, seed, margins=margins, guide=guide)
        n_ticks = sum(len(g["log"]) for g in r["guide"])
        n_ok = sum(1 for g in r["guide"] for x in g["log"] if x["ok"])
        wmax = max(max(g["net_w"]) for g in r["guide"])
        print(f"摆放 {r['elapsed']:.0f} s：粗网格评估 {n_ticks} 次（可布 {n_ok}），最好 J_route={r['J']:.4f}，"
              f"管网权重最大 {wmax}", flush=True)
        if r["solution"] is None:
            print("没有找到粗网格可布的摆放", flush=True)
            rows.append({"seed": seed, "placement_s": round(r["elapsed"], 1), "guide_ticks": n_ticks, "found": False})
            continue
        placed, blocks, nets, P, dsol, sc = to_scene(r["solution"])
        lbv = base.validate(blocks, nets, P, dsol, P["w"], "v2")
        co = coarse_eval(r["solution"])
        print(f"设备占地 {lbv['area_m2']} m²，管长下界 {lbv['L_lb_m']} m，粗网格管长 {co['L_m']:.1f} m", flush=True)
        f = fine(sc) if do_fine else None
        row = {"seed": seed, "found": True, "placement_s": round(r["elapsed"], 1), "J_route": r["J"],
               "guide_ticks": n_ticks, "guide_ok": n_ok, "net_w": r["guide"],
               "device_area_m2": lbv["area_m2"], "L_lb_m": lbv["L_lb_m"], "L_coarse_m": round(co["L_m"], 1),
               "fine": f, "placement": {k: list(v) for k, v in placed.items()}}
        if seed in b_place:
            bb = b_rows[seed]["new"] if seed in b_rows else None
            pb = {k: tuple(v) for k, v in b_place[seed]["placement"].items()}
            bl, bn, bP, bs = ex.device_level(inst, pb)
            bblocks, bnets, bP = base.load(BPATH)
            bsol = [{"x": pb[b["id"]][0], "y": pb[b["id"]][1],
                     "pose": next(k for k, p_ in enumerate(b["poses"]) if p_["rot"] == pb[b["id"]][2])} for b in bblocks]
            prb = sp.Problem(bblocks, bnets, bP, bP["w"])
            Jr_b, co_b = sp.route_score(prb, bsol, guide)
            row["B"] = {"device_area_m2": b_place[seed]["device_area_m2"], "J_route": Jr_b,
                        "L_coarse_m": round(co_b["L_m"], 1), "coarse_ok": co_b["ok"]}
            if bb:
                row["B"].update(all_ok=bb["all_ok"], metrics=bb["metrics"], route_s=bb["route_s"])
            print(f"对照 B 种子 {seed}：设备占地 {row['B']['device_area_m2']} m²，粗网格管长 {co_b['L_m']:.1f} m，"
                  f"J_route={Jr_b:.4f}" + (f"；精细 {'无违规' if bb['all_ok'] else '有违规'}，占地 {bb['metrics']['area_m2']} m²，"
                  f"管长 {bb['metrics']['L_m']} m，J={bb['metrics']['J']}" if bb else ""), flush=True)
        rows.append(row)
        (HERE / "布管引导摆放结果.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n汇总（精细布管 + 清理后，check_routes）：")
    for x in rows:
        if not x["found"]:
            print(f"  种子 {x['seed']}：D 未找到可布摆放")
            continue
        d, b = x["fine"], x.get("B", {})
        line = (f"  种子 {x['seed']}：粗网格 D J_route={x['J_route']:.4f} 管长 {x['L_coarse_m']} m 设备占地 {x['device_area_m2']} m²"
                f" ｜ B J_route={b.get('J_route', float('nan')):.4f} 管长 {b.get('L_coarse_m')} m 设备占地 {b.get('device_area_m2')} m²")
        if d:
            line += f"\n    精细 D {'无违规' if d['all_ok'] else '有违规'} J={d['metrics']['J']} 管长 {d['metrics']['L_m']} m"
        if b.get("metrics"):
            line += f" ｜ 精细 B {'无违规' if b['all_ok'] else '有违规'} J={b['metrics']['J']} 管长 {b['metrics']['L_m']} m"
        print(line)


if __name__ == "__main__":
    main()
