"""分块原型测试。运行：.venv/Scripts/python -m pytest 分块原型 -q"""
import copy
import json

import pytest

import blocking as bk

base = bk.base


def data(name):
    d = json.loads((bk.HERE / name).read_text(encoding="utf-8"))
    d["params"]["blocking"]["module_solve_s"] = 0.5
    return d


def groups(mods):
    return sorted(sorted(sorted(i["members"].values()) for i in m["instances"]) for m in mods)


def truth_groups(d):
    return sorted(sorted(m["members"].values()) for m in d["_truth"]["modules"])


def test_missing_required_param_raises():
    d = data("设备算例_小.json")
    del d["params"]["blocking"]["min_copies"]
    with pytest.raises(KeyError, match="blocking.min_copies"):
        bk.load_devices(d)


def test_port_on_two_nets_raises():
    d = data("设备算例_小.json")
    d["nets"].append({"id": "重复", "terminals": ["CT1.out", "TK.out"]})
    with pytest.raises(ValueError, match="同时接在"):
        bk.load_devices(d)


def test_auto_detection_matches_truth_on_small_plant():
    d = data("设备算例_小.json")
    used, auto = bk.find_modules(bk.load_devices(d))
    assert sorted(g for m in used for g in groups([m])[0]) == truth_groups(d)


def test_annotation_and_auto_give_same_grouping():
    used_a, _ = bk.find_modules(bk.load_devices(data("设备算例_小_标注.json")))
    used_b, _ = bk.find_modules(bk.load_devices(data("设备算例_小.json")))
    assert sorted(g for m in used_a for g in groups([m])[0]) == sorted(g for m in used_b for g in groups([m])[0])


def test_report_only_does_not_merge_auto_candidates():
    d = data("设备算例_小.json")
    d["params"]["blocking"]["auto_module_policy"] = "report_only"
    inst = bk.load_devices(d)
    used, auto = bk.find_modules(inst)
    assert used == [] and len(auto) == 2
    bdata, _, st = bk.build_block_instance(inst, used)
    assert st["blocks"] == 23


def test_chain_growth_stops_at_shared_header():
    used, _ = bk.find_modules(bk.load_devices(data("设备算例_小.json")))
    members = {dev for m in used for i in m["instances"] for dev in i["members"].values()}
    assert "FS" not in members and "JS" not in members


def test_array_requires_same_nets():
    d = data("设备算例_小.json")
    for n in d["nets"]:                                  # 把 CT4 的出口挪到单独管网上
        if n["id"] == "冷却供水":
            n["terminals"].remove("CT4.out")
    d["nets"].append({"id": "单独", "terminals": ["CT4.out", "TK.out"]})
    d["nets"] = [n for n in d["nets"] if n["id"] != "软水"]
    used, _ = bk.find_modules(bk.load_devices(d))
    arr = [m for m in used if m["source"] == "自动-并联阵列"]
    assert groups(arr) == [[["CT1", "CT2", "CT3"]]]


def test_invalid_given_layout_raises():
    d = data("设备算例_小_标注.json")
    d["modules"][0]["layouts"][0]["members"]["主机"]["pos"] = [0, 500]
    inst = bk.load_devices(d)
    used, _ = bk.find_modules(inst)
    with pytest.raises(ValueError, match="净距不足|检修区"):
        bk.build_block_instance(inst, used)


def test_block_instance_loads_and_remaps_nets():
    inst = bk.load_devices(data("设备算例_小.json"))
    used, _ = bk.find_modules(inst)
    bdata, owner, st = bk.build_block_instance(inst, used)
    blocks, nets, P = base.load_data(bdata)
    assert st["blocks"] == 12 and st["internal_nets"] == 8 and len(nets) == 25 - 8
    for net in nets:                                     # 每个端点都能在每个姿态下找到端口
        for i, pn in net["terms"]:
            assert all(pn in p["ports"] for p in blocks[i]["poses"])
    chain = next(b for b in blocks if b["copy_group"])
    assert {p["variant"] for p in chain["poses"]} and len(chain["poses"]) % 4 == 0


def test_placement_solution_of_block_instance_validates():
    import placement_sp as sp
    inst = bk.load_devices(data("设备算例_小.json"))
    used, _ = bk.find_modules(inst)
    bdata, _, _ = bk.build_block_instance(inst, used)
    blocks, nets, P = base.load_data(bdata)
    pr = sp.Problem(blocks, nets, P, P["w"])
    pr.stats = dict.fromkeys(("evals", "infeasible_sp", "pruned", "infeasible_lp", "lp", "invalid", "fractional"), 0)
    import random
    rng = random.Random(0)
    J = float("inf")
    for _ in range(50):
        J, sol = sp.evaluate(pr, sp.random_state(pr, rng))
        if sol:
            break
    assert sol is not None
    base.validate(blocks, nets, P, sol, P["w"], "v2")


def test_cluster_respects_max_units_and_partitions():
    d = data("设备算例_大.json")
    d["params"]["blocking"]["module_aspect_bands"] = [[1, 30]]
    inst = bk.load_devices(d)
    used, _ = bk.find_modules(inst)
    bdata, _, _ = bk.build_block_instance(inst, used)
    cl = bk.cluster(bdata, 3, 1.0, 0)
    units = [u for c in cl for u in c]
    assert all(len(c) <= 3 for c in cl)
    assert sorted(units) == sorted(b["id"] for b in bdata["blocks"])


def test_cluster_instance_drops_outside_terminals():
    inst = bk.load_devices(data("设备算例_小.json"))
    used, _ = bk.find_modules(inst)
    bdata, _, _ = bk.build_block_instance(inst, used)
    keep = ["FS", "JS", "HX"]
    sub = bk.cluster_instance(bdata, keep)
    for n in sub["nets"]:
        assert all(t.split(".")[0] in keep for t in n["terminals"])
    base.load_data(copy.deepcopy(sub))
