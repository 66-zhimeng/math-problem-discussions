"""布管原型测试（手工小场景）。运行：.venv/Scripts/python -m pytest -q（在本目录）"""
import copy
import random

import numpy as np

import pytest

import routing as rt

RP = {"D_default_mm": 200, "c_rho": 1.5, "delta_ep_mm": 100, "delta_pp_mm": 100, "K": 1, "eps_z_mm": 1,
      "z_max_mm": 3000, "service_zone_height_mm": 2000, "pitch_mm": 300, "margin_mm": 1500, "max_iters": 10,
      "pres_fac_init": 0.5, "pres_fac_mult": 1.6, "hist_fac": 0.5, "max_expansions": 300000, "astar_weight": 1.0, "route_workers": 1,
      "stall_iters": 3, "cleanup_max_expansions": 300000, "cleanup_trigger_nets": 2,
      "freeze_after_exhausted": 2}
W = {"area": 1.0, "length": 1.0, "bends": 0.3, "height_changes": 0.3}
SCALE = {"A0": 1e7, "L0": 1e4, "B0": 1, "C0": 1, "kappa": 2, "l_min_mm": 300}


def box(x0, y0, w, h, ht=1000, ports=None, zones=()):
    return {"box": (x0, y0, 0, x0 + w, y0 + h, ht), "zones": list(zones), "ports": ports or {}}


def scene(devices, nets, **over):
    rp = {**RP, **over}
    return rt.Scene(devices, nets, rp, W, SCALE)


def two_facing(gap=2000, dy=0):
    return {"A": box(0, 0, 1000, 1000, ports={"p": (1000, 500, 500, 1, 0, 0)}),
            "B": box(1000 + gap, dy, 1000, 1000, ports={"q": (1000 + gap, 500 + dy, 500, -1, 0, 0)})}


def route_one(sc):
    routes, hist, G = rt.negotiate(sc, log=lambda *_: None)
    return routes, hist


def test_missing_param_raises():
    rp = dict(RP); del rp["K"]
    with pytest.raises(KeyError, match="K"):
        rt.Scene(two_facing(), [], rp, W, SCALE)


def test_straight_connection_has_no_bends():
    sc = scene(two_facing(), [{"id": "n", "terms": [("A", "p"), ("B", "q")]}])
    routes, _ = route_one(sc)
    viol, m = rt.check_routes(sc, routes)
    assert viol == [] and m["bends"] == 0 and m["L_m"] == 2.0


def test_offset_connection_respects_min_straight():
    sc = scene(two_facing(gap=3000, dy=2000), [{"id": "n", "terms": [("A", "p"), ("B", "q")]}])
    routes, _ = route_one(sc)
    viol, m = rt.check_routes(sc, routes)
    assert viol == [] and m["bends"] == 2


def test_route_avoids_obstacle():
    dev = two_facing(gap=4000)
    dev["C"] = box(2000, -500, 1000, 2000, ht=3000)                  # 挡在中间、顶到层高附近
    sc = scene(dev, [{"id": "n", "terms": [("A", "p"), ("B", "q")]}])
    routes, _ = route_one(sc)
    viol, m = rt.check_routes(sc, routes)
    assert viol == [] and m["bends"] >= 2


def test_blocked_port_reported():
    dev = two_facing(gap=4000)
    dev["Z"] = box(3000, 0, 500, 1000, zones=[(1000, 0, 1200, 1000)])  # 检修区紧贴 A 的端口
    sc = scene(dev, [{"id": "n", "terms": [("A", "p"), ("B", "q")]}])
    routes, hist = route_one(sc)
    assert "n" not in routes and hist[-1]["failed"]["n"].startswith("端口正前方")


def test_two_nets_keep_clearance():
    dev = {"A": box(0, 0, 1000, 1000, ports={"p": (1000, 400, 500, 1, 0, 0), "p2": (1000, 700, 500, 1, 0, 0)}),
           "B": box(4000, 0, 1000, 1000, ports={"q": (4000, 700, 500, -1, 0, 0), "q2": (4000, 400, 500, -1, 0, 0)})}
    nets = [{"id": "n1", "terms": [("A", "p"), ("B", "q")]}, {"id": "n2", "terms": [("A", "p2"), ("B", "q2")]}]
    sc = scene(dev, nets, K=3)
    routes, hist = route_one(sc)
    viol, _ = rt.check_routes(sc, routes)
    assert set(routes) == {"n1", "n2"} and viol == []


def test_tee_net_routes_and_validates():
    dev = {"A": box(0, 0, 1000, 1000, ports={"p": (1000, 500, 500, 1, 0, 0)}),
           "B": box(5000, 0, 1000, 1000, ports={"q": (5000, 500, 500, -1, 0, 0)}),
           "C": box(2500, 3000, 1000, 1000, ports={"r": (3000, 3000, 500, 0, -1, 0)})}
    sc = scene(dev, [{"id": "t", "terms": [("A", "p"), ("B", "q"), ("C", "r")]}])
    routes, _ = route_one(sc)
    viol, m = rt.check_routes(sc, routes)
    assert viol == [] and len(routes["t"]) == 2 and routes["t"][1]["end"][0] == "tee"


# ---------------- 校验器能查出违规
def base_route():
    sc = scene(two_facing(gap=3000, dy=2000), [{"id": "n", "terms": [("A", "p"), ("B", "q")]}])
    routes, _ = route_one(sc)
    return sc, routes


def test_validator_detects_short_straight():
    sc, routes = base_route()
    bad = copy.deepcopy(routes)
    bad["n"][0]["points"] = [(1000, 500, 500), (1300, 500, 500), (1300, 2500, 500), (4000, 2500, 500)]
    viol, _ = rt.check_routes(sc, bad)
    assert any("直管段" in v for v in viol)


def test_validator_detects_device_collision():
    dev = two_facing(gap=3000)
    dev["C"] = box(2000, 0, 1000, 1000)
    sc = scene(dev, [{"id": "n", "terms": [("A", "p"), ("B", "q")]}])
    bad = {"n": [{"start": ("A", "p"), "end": ("port", ("B", "q")), "points": [(1000, 500, 500), (4000, 500, 500)]}]}
    viol, _ = rt.check_routes(sc, bad)
    assert any("设备 C" in v for v in viol)


def test_validator_detects_height_change_limit():
    sc, _ = base_route()
    bad = {"n": [{"start": ("A", "p"), "end": ("port", ("B", "q")), "points": [
        (1000, 500, 500), (2000, 500, 500), (2000, 500, 1500), (3000, 500, 1500), (3000, 500, 500),
        (3000, 2500, 500), (4000, 2500, 500)]}]}
    viol, _ = rt.check_routes(sc, bad)
    assert any("高度变化 2 > K=1" in v for v in viol)


def test_validator_detects_pipe_pipe_clearance():
    dev = {"A": box(0, 0, 1000, 1000, ports={"p": (1000, 400, 500, 1, 0, 0), "p2": (1000, 600, 500, 1, 0, 0)}),
           "B": box(4000, 0, 1000, 1000, ports={"q": (4000, 400, 500, -1, 0, 0), "q2": (4000, 600, 500, -1, 0, 0)})}
    nets = [{"id": "n1", "terms": [("A", "p"), ("B", "q")]}, {"id": "n2", "terms": [("A", "p2"), ("B", "q2")]}]
    sc = scene(dev, nets)
    bad = {"n1": [{"start": ("A", "p"), "end": ("port", ("B", "q")), "points": [(1000, 400, 500), (4000, 400, 500)]}],
           "n2": [{"start": ("A", "p2"), "end": ("port", ("B", "q2")), "points": [(1000, 600, 500), (4000, 600, 500)]}]}
    viol, _ = rt.check_routes(sc, bad)
    assert any("管–管净距不足" in v for v in viol)


def test_cleanup_reroutes_clashing_net_with_others_fixed():
    dev = {"A": box(0, 0, 1000, 1500, ports={"p": (1000, 300, 500, 1, 0, 0), "p2": (1000, 1100, 500, 1, 0, 0)}),
           "B": box(5000, 0, 1000, 1500, ports={"q": (5000, 300, 500, -1, 0, 0), "q2": (5000, 1100, 500, -1, 0, 0)})}
    nets = [{"id": "n1", "terms": [("A", "p"), ("B", "q")]}, {"id": "n2", "terms": [("A", "p2"), ("B", "q2")]}]
    sc = scene(dev, nets, K=3)
    detour = [(1000, 300, 500), (2000, 300, 500), (2000, 1100, 500), (4000, 1100, 500), (4000, 300, 500), (5000, 300, 500)]
    bad = {"n1": [{"start": ("A", "p"), "end": ("port", ("B", "q")), "points": detour}],
           "n2": [{"start": ("A", "p2"), "end": ("port", ("B", "q2")), "points": [(1000, 1100, 500), (5000, 1100, 500)]}]}
    assert rt.clashes(sc, bad)
    routes, records = rt.cleanup(rt.Grid(sc), sc, bad, log=lambda *_: None)
    viol, _ = rt.check_routes(sc, routes)
    assert viol == [] and records[0]["result"] == "已消除"
    assert routes["n2"] == bad["n2"]                                   # 只重布了 n1


def test_compiled_astar_matches_python():
    """编译实现与纯 Python 参考实现：同样的扩展数与同样的折线（含拥堵代价、含接三通）。"""
    dev = {"A": box(0, 0, 1000, 1500, ports={"p": (1000, 300, 500, 1, 0, 0), "p2": (1000, 1100, 500, 1, 0, 0)}),
           "B": box(5000, 0, 1000, 1500, ports={"q": (5000, 300, 500, -1, 0, 0), "q2": (5000, 1100, 500, -1, 0, 0)}),
           "C": box(2500, 3000, 1000, 1000, ports={"r": (3000, 3000, 500, 0, -1, 0)})}
    nets = [{"id": "n1", "terms": [("A", "p"), ("B", "q"), ("C", "r")]},
            {"id": "n2", "terms": [("A", "p2"), ("B", "q2")]}]
    sc = scene(dev, nets, K=3)
    G = rt.Grid(sc)
    rng = random.Random(0)
    halo = np.array([1 if rng.random() < 0.05 else 0 for _ in range(G.size * 3)], dtype=np.int32)
    hist = np.array([rng.random() * 0.2 for _ in range(G.size * 3)], dtype=np.float32)
    for pres in (0.0, 1.0):
        ctx = {"halo": halo, "hist": hist, "pres": pres}
        for net in sc.nets:
            s1 = {"expansions": 0, "exhausted": 0}
            s2 = {"expansions": 0, "exhausted": 0}
            fast, _ = rt.route_net(G, sc, net, {**ctx, "stats": s1})
            saved, rt.astar = rt.astar, rt.astar_py
            try:
                slow, _ = rt.route_net(G, sc, net, {**ctx, "halo": halo.tolist(), "hist": hist.tolist(), "stats": s2})
            finally:
                rt.astar = saved
            assert s1["expansions"] == s2["expansions"]
            assert [b["points"] for b in fast] == [b["points"] for b in slow]


def test_parallel_negotiation_matches_rules():
    dev = {"A": box(0, 0, 1000, 1000, ports={"p": (1000, 400, 500, 1, 0, 0), "p2": (1000, 700, 500, 1, 0, 0)}),
           "B": box(4000, 0, 1000, 1000, ports={"q": (4000, 700, 500, -1, 0, 0), "q2": (4000, 400, 500, -1, 0, 0)})}
    nets = [{"id": "n1", "terms": [("A", "p"), ("B", "q")]}, {"id": "n2", "terms": [("A", "p2"), ("B", "q2")]}]
    sc = scene(dev, nets, K=3, route_workers=2)
    routes, hist = route_one(sc)
    viol, _ = rt.check_routes(sc, routes)
    assert set(routes) == {"n1", "n2"} and viol == []


def test_validator_accepts_tee_at_elbow_and_discounts_bend():
    dev = {"A": box(0, 0, 1000, 1000, ports={"p": (1000, 500, 500, 1, 0, 0)}),
           "B": box(3000, 3000, 1000, 1000, ports={"q": (3500, 3000, 500, 0, -1, 0)}),
           "C": box(6000, 0, 1000, 1000, ports={"r": (6000, 500, 500, -1, 0, 0)})}
    sc = scene(dev, [{"id": "t", "terms": [("A", "p"), ("B", "q"), ("C", "r")]}])
    main = {"start": ("A", "p"), "end": ("port", ("B", "q")),
            "points": [(1000, 500, 500), (3500, 500, 500), (3500, 3000, 500)]}      # 弯头在 (3500, 500)
    branch = {"start": ("C", "r"), "end": ("tee", None),
              "points": [(6000, 500, 500), (3500, 500, 500)]}                        # 沿 A 侧腿的延长线接入
    viol, m = rt.check_routes(sc, {"t": [main, branch]})
    assert viol == [] and m["bends"] == 0


def test_validator_rejects_collinear_tee_inside_segment():
    dev = {"A": box(0, 0, 1000, 1000, ports={"p": (1000, 500, 500, 1, 0, 0)}),
           "B": box(5000, 0, 1000, 1000, ports={"q": (5000, 500, 500, -1, 0, 0)}),
           "C": box(2500, 3000, 1000, 1000, ports={"r": (3000, 3000, 500, 0, -1, 0)})}
    sc = scene(dev, [{"id": "t", "terms": [("A", "p"), ("B", "q"), ("C", "r")]}])
    main = {"start": ("A", "p"), "end": ("port", ("B", "q")), "points": [(1000, 500, 500), (5000, 500, 500)]}
    bad = {"start": ("C", "r"), "end": ("tee", None),
           "points": [(3000, 3000, 500), (3000, 1500, 500), (2000, 1500, 500), (2000, 500, 500), (3000, 500, 500)]}
    viol, _ = rt.check_routes(sc, {"t": [main, bad]})
    assert any("共线" in v for v in viol)


# ---------------- 粗网格快速布管
import coarse_route as cr  # noqa: E402

CRP = {"coarse_cell_mm": 300, "coarse_iters": 10, "coarse_vertical_penalty_mm": 1000}


def test_coarse_routes_simple_pair():
    sc = scene(two_facing(gap=3000, dy=2000), [{"id": "n", "terms": [("A", "p"), ("B", "q")]}], **CRP)
    r = cr.coarse_route(sc)
    assert r["ok"] and r["failed"] == {}


def test_coarse_reports_blocked_port():
    dev = two_facing(gap=4000)
    dev["Z"] = box(3000, 0, 500, 1000, zones=[(1000, 0, 1200, 1000)])
    sc = scene(dev, [{"id": "n", "terms": [("A", "p"), ("B", "q")]}], **CRP)
    r = cr.coarse_route(sc)
    assert not r["ok"] and r["failed"]["n"].startswith("端口被堵")


def test_coarse_missing_param_raises():
    sc = scene(two_facing(), [{"id": "n", "terms": [("A", "p"), ("B", "q")]}])
    with pytest.raises(KeyError, match="coarse_cell_mm"):
        cr.coarse_route(sc)
