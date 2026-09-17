"""分块实验。

E1 小冷站（23 台）：有标注 vs 无标注 → 模块是否一致 → 转块级算例 → placement_sp 摆放（各 30 s）
E2 大算例（548 台）：自动识别的准确率（对照生成器的真实模块）、转块耗时、聚簇质量（对照随机划分、系统标签）
   以及取两个簇（一个冷站簇、一个区域簇）作为子问题用 placement_sp 摆放（各 20 s）

输出：输出/*.json（块级算例、簇子问题）、分块实验结果.json、分块实验_log.txt（重定向）、分块示例.png
运行：.venv/Scripts/python 分块实验.py
"""
import json
import statistics
import sys
import time
from collections import Counter

import blocking as bk

base = bk.base
sys.path.insert(0, str(bk.HERE.parent / "摆放原型"))
import placement_sp as sp  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = bk.HERE / "输出"
PROCS = 12


def to_block_instance(name):
    inst = bk.load_devices(bk.HERE / name)
    t0 = time.time()
    used, auto = bk.find_modules(inst)
    detect_s = time.time() - t0
    bdata, owner, st = bk.build_block_instance(inst, used)
    st["detect_s"] = round(detect_s, 4)
    return inst, used, auto, bdata, owner, st


def place(bdata, fname, budget, seed=0):
    path = OUT / fname
    path.write_text(json.dumps(bdata, ensure_ascii=False, indent=1), encoding="utf-8")
    r = sp.run(path, budget, PROCS, seed)
    blocks, nets, P = base.load(path)
    v = base.validate(blocks, nets, P, r["solution"], P["w"], "v2")
    return {"J": v["J_full_weights"], "area_m2": v["area_m2"], "W_m": v["W_m"], "H_m": v["H_m"],
            "L_lb_m": v["L_lb_m"], "bends_lb": v["bends_lb"], "blocks": len(blocks), "nets": len(nets),
            "evals": r["stats"]["evals"]}, (blocks, nets, P, r["solution"], v)


def detection_score(inst, used, auto):
    truth = {frozenset(m["members"].values()): m["template"] for m in inst["data"]["_truth"]["modules"]}
    found = {frozenset(i["members"].values()): m["template"] for m in auto for i in m["instances"]}
    tp = set(truth) & set(found)
    return {"truth_instances": len(truth), "auto_instances": len(found), "matched": len(tp),
            "recall_pct": round(100 * len(tp) / len(truth), 1), "precision_pct": round(100 * len(tp) / len(found), 1),
            "missed": sorted(truth[k] for k in set(truth) - tp),
            "extra": sorted(Counter(found[k] for k in set(found) - tp).items())}


def purity(inst, owner, clusters):
    system = inst["data"]["_truth"]["system"]
    devs_of = {}
    for d, (b, _) in owner.items():
        devs_of.setdefault(b, []).append(d)
    good = tot = 0
    for c in clusters:
        sy = Counter(system[d] for b in c for d in devs_of[b])
        good += sy.most_common(1)[0][1]; tot += sum(sy.values())
    return round(100 * good / tot, 1)


def main():
    OUT.mkdir(exist_ok=True)
    res = {}

    # ---------------- E1
    e1 = {}
    plots = []
    for name, label in (("设备算例_小_标注.json", "标注"), ("设备算例_小.json", "自动识别")):
        inst, used, auto, bdata, owner, st = to_block_instance(name)
        groups = sorted(sorted(sorted(i["members"].values()) for i in m["instances"]) for m in used)
        pr, (blocks, nets, P, sol, v) = place(bdata, f"小_块级_{label}.json", 30)
        e1[label] = {"modules": [(m["template"], m["source"], len(m["instances"])) for m in used],
                     "groups": groups, "blocking": st, "placement_30s": pr}
        plots.append({"label": f"小冷站（{label}）J={pr['J']}", "solution": sol, "validated": v, "_b": (blocks, nets, P)})
        print(f"[E1 {label}] 模块 {e1[label]['modules']}")
        print(f"  设备 {st['devices']} → 块 {st['blocks']}；管网 {st['nets_in']} → {st['nets_out']}（模块内部 {st['internal_nets']}）")
        for lay in st["layouts"]:
            print(f"  {lay['template']}：{lay['layout_from']}，{lay['time_s']} s，{lay['variants']}")
        print(f"  摆放 30 s：{pr}", flush=True)
    e1["same_grouping"] = e1["标注"]["groups"] == e1["自动识别"]["groups"]
    print(f"  标注与自动识别的分组一致：{e1['same_grouping']}")
    res["E1"] = {k: v for k, v in e1.items()}

    # ---------------- E2
    inst, used, auto, bdata, owner, st = to_block_instance("设备算例_大.json")
    det = detection_score(inst, used, auto)
    print(f"\n[E2] 识别 {st['detect_s']} s：{det}")
    print(f"  设备 {st['devices']} → 块 {st['blocks']}（模块块 {st['module_blocks']}）；管网 {st['nets_in']} → "
          f"{st['nets_out']}；转块（含求内部排法）{st['time_s']} s")
    for lay in st["layouts"]:
        print(f"  {lay['template']}（{lay['source']}，{lay['instances']} 个实例）：{lay['time_s']} s，{lay['variants']}")
    (OUT / "大_块级.json").write_text(json.dumps(bdata, ensure_ascii=False, indent=1), encoding="utf-8")
    bp = inst["bp"]
    clus = {}
    for mx in (bp["cluster_max_units"], 10, 5):
        t0 = time.time()
        cl = bk.cluster(bdata, mx, bp["cluster_resolution"], bp["seed"])
        el = time.time() - t0
        met = bk.cluster_metrics(bdata, cl)
        rnd = [bk.cluster_metrics(bdata, bk.random_partition(cl, s))["cut_nets_pct"] for s in range(20)]
        rnd_pur = [purity(inst, owner, bk.random_partition(cl, s)) for s in range(20)]
        clus[mx] = {**met, "time_s": round(el, 3), "purity_pct": purity(inst, owner, cl),
                    "random_cut_nets_pct_median": statistics.median(rnd),
                    "random_purity_pct_median": statistics.median(rnd_pur)}
        print(f"  聚簇 上限 {mx}：{clus[mx]}")
        if mx == bp["cluster_max_units"]:
            chosen = cl
    sub = {}
    plant = next(c for c in chosen if any("冷" in u or "链_冷冻泵" in u for u in c))
    zone = next(c for c in chosen if any(u.startswith("Z") for u in c))
    for tag, c in (("冷站簇", plant), ("区域簇", zone)):
        sdata = bk.cluster_instance(bdata, c)
        pr, _ = place(sdata, f"大_{tag}.json", 20)
        sub[tag] = {"units": c, **pr}
        print(f"  {tag}（{len(c)} 块）摆放 20 s：{pr}", flush=True)
    res["E2"] = {"detection": det, "blocking": st, "clusters": clus, "subproblems": sub}
    (bk.HERE / "分块实验结果.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                          encoding="utf-8")

    # 图：小冷站两种分组的摆放结果（块不同，分两张图）
    for p_, fname in zip(plots, ("分块示例_标注.png", "分块示例_自动.png")):
        blocks, nets, P = p_["_b"]
        base.plot(blocks, nets, P, [{k: v for k, v in p_.items() if k != "_b"}], bk.HERE / fname)
    print("\n已保存 分块实验结果.json、分块示例_*.png、输出/")


if __name__ == "__main__":
    main()
