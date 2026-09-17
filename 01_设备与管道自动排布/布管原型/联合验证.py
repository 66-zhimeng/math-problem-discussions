"""小实验：摆放时是否按侧预留出管空间，对“能否布通”的影响。

算例：小冷站设备级（23 台设备各自独立摆放，不合并模块），25 个管网。
方法（摆放同为 placement_sp 序列对 + LP，12 进程，预算相同；布管参数完全相同）：
  A 接入区：端口前方 ℓ_min 不压其他设备与检修区（上一版做法）
  B 出管留空：每台设备每侧按端口数预留 ℓ_min + ρ + (k−1)(D+δ_pp) + D/2 + δ_ep，缝隙为两侧之和
评价：布管后用独立校验器 check_routes 统计是否全部布通、冲突、含管道占地、实际管长、弯头、高度变化、J。

运行：.venv/Scripts/python 联合验证.py [摆放秒数，默认 60] [种子数，默认 3]
输出：联合验证结果.json、联合验证_{方法}_种子{s}.png、日志重定向到 联合验证_log.txt
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
import 布管实验 as ex  # noqa: E402

bk, base, sp, rt = ex.bk, ex.base, ex.sp, ex.rt
sys.stdout.reconfigure(encoding="utf-8")


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    seeds = range(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    d = json.loads((bk.HERE / "设备算例_小.json").read_text(encoding="utf-8"))
    d["params"]["blocking"]["auto_module_policy"] = "report_only"          # 设备级：不合并模块
    inst = bk.load_devices(d)
    bdata, _, _ = bk.build_block_instance(inst, [])
    bpath = HERE / "_设备级块算例.json"
    bpath.write_text(json.dumps(bdata, ensure_ascii=False), encoding="utf-8")
    rp = inst["P"]["routing"]
    access = {"D_mm": rp["D_default_mm"], "delta_ep_mm": rp["delta_ep_mm"]}
    margins = {"D_mm": rp["D_default_mm"], "delta_pp_mm": rp["delta_pp_mm"], "delta_ep_mm": rp["delta_ep_mm"],
               "c_rho": rp["c_rho"]}
    methods = {"A接入区": {"access": access}, "B出管留空": {"margins": margins}}
    results = []
    for seed in seeds:
        for name, kw in methods.items():
            print(f"\n===== {name}，种子 {seed} =====", flush=True)
            t0 = time.time()
            r = sp.run(bpath, budget, 12, seed, **kw)
            bl, bn, bP = base.load(bpath)
            vb = base.validate(bl, bn, bP, r["solution"], bP["w"], "v2")
            placed = ex.expand(inst, [], bdata, bl, r["solution"])
            blocks, nets, P, sol = ex.device_level(inst, placed)
            lbv = base.validate(blocks, nets, P, sol, P["w"], "v2")
            sc = ex.scene(inst, blocks, nets, P, sol)
            print(f"摆放 {time.time() - t0:.0f} s：设备占地 {lbv['area_m2']} m²，管长下界 {lbv['L_lb_m']} m，"
                  f"弯头下界 {lbv['bends_lb']}，J_lb={lbv['J_full_weights']}", flush=True)
            t1 = time.time()
            routes, hist, G = rt.negotiate(sc, log=lambda x: print(x, flush=True))
            route_s = time.time() - t1
            viol, met = rt.check_routes(sc, routes)
            failed = hist[-1]["failed"]
            clash = [v for v in viol if "净距不足" in v or "穿过" in v]
            other = [v for v in viol if v not in clash and not v.endswith("未布通")]
            ok = not failed and not viol
            print(f"布管 {route_s:.0f} s，{len(hist)} 轮：布通 {len(routes)}/{len(sc.nets)}，冲突对 {len(clash)}，"
                  f"其他违规 {len(other)}，最后冲突边 {hist[-1]['conflict_edges']}；"
                  f"{'【全部布通且无违规】' if ok else '【未达成】'}")
            print(f"  含管道占地 {met['area_m2']} m²，管长 {met['L_m']} m，弯头 {met['bends']}，高度变化 "
                  f"{met['height_changes']}，J={met['J']}" + ("" if ok else "（未完整，仅供参考）"))
            if failed:
                print("  未布通：" + "；".join(f"{k}（{v}）" for k, v in failed.items()))
            ex.plot(sc, routes, met, lbv, f"{name} 种子 {seed}：布通 {len(routes)}/{len(sc.nets)}，违规 {len(viol)}",
                    HERE / f"联合验证_{name}_种子{seed}.png")
            results.append({"method": name, "seed": seed, "placement_s": budget, "route_s": round(route_s, 1),
                            "iterations": len(hist), "routed": len(routes), "nets": len(sc.nets),
                            "failed": failed, "clash_pairs": len(clash), "other_violations": other,
                            "final_conflict_edges": hist[-1]["conflict_edges"], "all_ok": ok,
                            "device_area_m2": lbv["area_m2"], "J_lb": lbv["J_full_weights"],
                            "routed_metrics": {k: v for k, v in met.items() if k != "per_net"},
                            "history": [{k: v for k, v in h.items() if k != "net_times"} for h in hist],
                            "placement": {k: list(v) for k, v in placed.items()}})
            (HERE / "联合验证结果.json").write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    print("\n汇总：")
    for x in results:
        m = x["routed_metrics"]
        print(f"  {x['method']} 种子 {x['seed']}：{'成功' if x['all_ok'] else '未达成'}，布通 {x['routed']}/{x['nets']}，"
              f"冲突对 {x['clash_pairs']}，含管道占地 {m['area_m2']} m²，管长 {m['L_m']} m，弯头 {m['bends']}，"
              f"高度变化 {m['height_changes']}，J={m['J']}，布管 {x['route_s']} s / {x['iterations']} 轮")


if __name__ == "__main__":
    main()
