"""布管实验：小冷站。

流程：设备级算例（有标注）→ 分块 → placement_sp 摆放（30 s）→ 展开成设备坐标
      → 设备级下界（placement_cpsat.validate v2）→ 协商布线 → 独立校验 → 与下界对比
摆放两种条件：原摆放（不考虑端口）；预留端口接入区（端口前方 ℓ_min 不压其他设备与检修区）

运行：.venv/Scripts/python 布管实验.py [摆放种子，可多个，默认 0 1]
输出：布管结果.json、布管结果_{条件}_种子{s}.png、布管实验_log.txt（重定向）
"""
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "分块原型"))
sys.path.insert(0, str(HERE.parent / "摆放原型"))
import blocking as bk  # noqa: E402
import placement_cpsat as base  # noqa: E402
import placement_sp as sp  # noqa: E402

import routing as rt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
PLACE_S = 30


# ============================================================ 摆放 → 设备坐标
# 转换逻辑在内核里（layout_kernel/build.py），这里只做转发，供本目录的实验脚本沿用 ex.expand 等写法
sys.path.insert(0, str(HERE.parent))
from layout_kernel import build as _kb  # noqa: E402

expand, device_level, scene = _kb.expand, _kb.device_level, _kb.scene


# ============================================================ 画图
def plot(sc, routes, metrics, lbv, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    fig = plt.figure(figsize=(18, 9))
    ax = fig.add_subplot(1, 2, 1)
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    colors = plt.cm.tab20.colors
    for did, d in sc.dev.items():
        x0, y0, _, x1, y1, z1 = d["box"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=True, alpha=0.15, color="gray"))
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, did, ha="center", va="center", fontsize=6)
        for zx0, zy0, zx1, zy1 in d["zones"]:
            ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, alpha=0.12, color="orange", lw=0))
        for xs, ys, zs in _box_edges(d["box"]):
            ax3.plot(xs, ys, zs, color="gray", lw=0.5)
    for c, (nid, brs) in enumerate(routes.items()):
        col = colors[c % 20]
        for br in brs:
            pts = br["points"]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=2.2, alpha=0.9)
            ax.scatter([p[0] for p in pts[1:-1]], [p[1] for p in pts[1:-1]], s=6, color=col)
            ax3.plot([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts], color=col, lw=1.8)
    ax.set_aspect("equal"); ax.autoscale()
    ax.set_title(f"{title}\n布管后：占地 {metrics['area_m2']} m²  管长 {metrics['L_m']} m  弯头 {metrics['bends']}  "
                 f"高度变化 {metrics['height_changes']}  J={metrics['J']}\n"
                 f"摆放阶段下界：占地 {lbv['area_m2']} m²  管长 {lbv['L_lb_m']} m  弯头 {lbv['bends_lb']}",
                 fontsize=9)
    ax.set_xlabel("mm（俯视；彩线为管道中心线，点为弯头）")
    ax3.set_title("三维视图", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _box_edges(b):
    x0, y0, z0, x1, y1, z1 = b
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    yield [p[0] for p in pts], [p[1] for p in pts], [z1] * 5
    for x, y in pts[:4]:
        yield [x, x], [y, y], [z0, z1]


# ============================================================ 主程序
def one_case(inst, modules, bdata, bpath, cond, access, seed):
    print(f"\n===== {cond}，摆放种子 {seed} =====", flush=True)
    r = sp.run(bpath, PLACE_S, 12, seed, access=access)
    bblocks, bnets, bP = base.load(bpath)
    vb = base.validate(bblocks, bnets, bP, r["solution"], bP["w"], "v2")
    placed = expand(inst, modules, bdata, bblocks, r["solution"])
    blocks, nets, P, sol = device_level(inst, placed)
    lbv = base.validate(blocks, nets, P, sol, P["w"], "v2")             # 设备级几何合法 + 全部管网下界
    sc = scene(inst, blocks, nets, P, sol)
    grid = P["grid"]
    members = {ins["id"]: ins["members"] for m in modules for ins in m["instances"]}
    for b, s_ in zip(bblocks, r["solution"]):                           # 展开正确性：模块对外端口坐标一致
        for pn, v in b["poses"][s_["pose"]]["ports"].items():
            if "__" in pn:
                role, port = pn.split("__")
                q = sc.port(members[b["id"]][role], port)
                assert (q[0], q[1]) == ((s_["x"] + v[0]) * grid, (s_["y"] + v[1]) * grid), (b["id"], pn)
    print(f"摆放（块级 J={vb['J_full_weights']}）→ 设备级下界：占地 {lbv['area_m2']} m²，管长 {lbv['L_lb_m']} m，"
          f"弯头 {lbv['bends_lb']}，J_lb={lbv['J_full_weights']}（不含高度变化项）", flush=True)
    t0 = time.time()
    routes, history, G = rt.negotiate(sc, log=lambda x: print(x, flush=True))
    route_s = time.time() - t0
    viol, met = rt.check_routes(sc, routes)
    failed = history[-1]["failed"]
    print(f"布管用时 {route_s:.1f} s，网格 {G.n[0]}×{G.n[1]}×{G.n[2]}={G.size} 个节点，轮数 {len(history)}")
    print(f"布通 {len(routes)}/{len(sc.nets)}；未布通：" + ("无" if not failed else "".join(f"\n    {k}：{v}" for k, v in failed.items())))
    other = [v for v in viol if not v.endswith("未布通")]
    print(f"校验（除未布通外）违规 {len(other)} 条" + ("" if not other else "\n    " + "\n    ".join(other[:30])))
    print(f"布管后（仅已布通管网）：占地 {met['area_m2']} m²，管长 {met['L_m']} m，弯头 {met['bends']}，"
          f"高度变化 {met['height_changes']}，J={met['J']}")
    per = []
    lb_net = {x["net"]: x for x in lbv["per_net"]}
    for n in inst["nets"]:
        nid, lb = n["id"], lb_net[n["id"]]
        v = met["per_net"].get(nid)
        per.append({"net": nid, "terminals": len(n["terms"]), "L_lb_m": lb["L_lb_m"], "bends_lb": lb["bends_lb"],
                    "L_m": v["L_mm"] / 1000 if v else None, "bends": v["bends"] if v else None,
                    "height_changes": v["height_changes"] if v else None,
                    "L_ratio": round(v["L_mm"] / 1000 / lb["L_lb_m"], 2) if v and lb["L_lb_m"] else None})
    print(f"{'管网':<8}{'端点':>4}{'管长 m':>9}{'下界 m':>9}{'比值':>6}{'弯头':>5}{'下界':>5}{'高变':>5}")
    for x in per:
        f = lambda v, w, fmt="": f"{v:>{w}{fmt}}" if v is not None else f"{'—':>{w}}"
        print(f"{x['net']:<8}{x['terminals']:>4}{f(x['L_m'], 9, '.1f')}{x['L_lb_m']:>9.1f}{f(x['L_ratio'], 6)}"
              f"{f(x['bends'], 5)}{x['bends_lb']:>5}{f(x['height_changes'], 5)}")
    both = [x for x in per if x["L_m"] is not None]
    routed_lb_L = sum(x["L_lb_m"] for x in both)
    routed_lb_B = sum(x["bends_lb"] for x in both)
    print(f"已布通管网合计：管长 {met['L_m']} m / 下界 {routed_lb_L:.1f} m"
          + (f" = {met['L_m'] / routed_lb_L:.2f} 倍" if routed_lb_L else "") + f"；弯头 {met['bends']} / 下界 {routed_lb_B}")
    plot(sc, routes, met, lbv, f"小冷站 {cond} 种子 {seed}（布通 {len(routes)}/{len(sc.nets)}，其余违规 {len(other)} 条）",
         HERE / f"布管结果_{cond}_种子{seed}.png")
    return {"condition": cond, "seed": seed, "placement_block_J": vb["J_full_weights"],
            "lower_bound": {k: lbv[k] for k in ("area_m2", "W_m", "H_m", "L_lb_m", "bends_lb", "J_full_weights")},
            "routed_nets": len(routes), "total_nets": len(sc.nets), "failed": failed,
            "routed": {k: v for k, v in met.items() if k != "per_net"},
            "routed_subset_lb": {"L_lb_m": round(routed_lb_L, 1), "bends_lb": routed_lb_B},
            "violations_other": other, "route_time_s": round(route_s, 1), "grid_nodes": G.size,
            "iterations": history, "per_net": per,
            "placement": {d: list(v) for d, v in placed.items()},
            "routes": {nid: [{"start": list(b["start"]), "end": [b["end"][0], list(b["end"][1]) if b["end"][1] else None],
                              "points": [list(pt) for pt in b["points"]]} for b in brs] for nid, brs in routes.items()}}


def main():
    inst = bk.load_devices(bk.HERE / "设备算例_小_标注.json")
    modules, _ = bk.find_modules(inst)
    bdata, _, _ = bk.build_block_instance(inst, modules)
    bpath = HERE / "小冷站_块级.json"
    bpath.write_text(json.dumps(bdata, ensure_ascii=False, indent=1), encoding="utf-8")
    rp = inst["P"]["routing"]
    access = {"D_mm": rp["D_default_mm"], "delta_ep_mm": rp["delta_ep_mm"]}
    seeds = [int(x) for x in sys.argv[1:]] or [0, 1]
    results = []
    for cond, acc in (("原摆放", None), ("预留接入区", access)):
        for seed in seeds:
            results.append(one_case(inst, modules, bdata, bpath, cond, acc, seed))
            (HERE / "布管结果.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n汇总：")
    for x in results:
        print(f"  {x['condition']} 种子 {x['seed']}：布通 {x['routed_nets']}/{x['total_nets']}，其余违规 "
              f"{len(x['violations_other'])}，管长 {x['routed']['L_m']} m（对应下界 {x['routed_subset_lb']['L_lb_m']} m），"
              f"弯头 {x['routed']['bends']}（下界 {x['routed_subset_lb']['bends_lb']}），高度变化 {x['routed']['height_changes']}，"
              f"占地 {x['routed']['area_m2']} m²（摆放下界 {x['lower_bound']['area_m2']}），布管 {x['route_time_s']} s")


if __name__ == "__main__":
    main()
