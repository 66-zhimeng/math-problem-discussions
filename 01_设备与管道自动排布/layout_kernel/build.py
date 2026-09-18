"""算例 → 求解器输入的转换（块级解 → 设备坐标 → 布管场景）。

本文件是 摆放 / 分块 / 布管 三个原型之间唯一的粘合层，api.py 与实验脚本都用它，不要各写一份。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("分块原型", "摆放原型", "布管原型"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import blocking as bk  # noqa: E402
import placement_cpsat as base  # noqa: E402
import routing as rt  # noqa: E402


def expand(inst, modules, bdata, blocks, sol):
    """块级解 → 每台设备的 (x, y, 旋转)（网格单位）。"""
    grid = inst["P"]["grid_mm"]
    members_of = {ins["id"]: ins["members"] for m in modules for ins in m["instances"]}
    tpl_of = {b["id"]: b.get("template") for b in bdata["blocks"]}
    out = {}
    for b, s in zip(blocks, sol):
        pose = b["poses"][s["pose"]]
        R = pose["rot"]
        if tpl_of[b["id"]] is None:
            out[b["id"]] = (s["x"], s["y"], R)
            continue
        var = bdata["templates"][tpl_of[b["id"]]]["variants"][pose["variant"]]
        w, h = base.g(var["size"][0], grid), base.g(var["size"][1], grid)
        for role, dev in members_of[b["id"]].items():
            spec = var["members"][role]
            t = inst["types"][inst["devs"][dev]["type"]]
            mp = base.make_pose(t, spec["rot"], grid, "默认")
            px, py = base.g(spec["pos"][0], grid), base.g(spec["pos"][1], grid)
            ax, ay = base.rot_point(px, py, w, h, R)
            bx, by = base.rot_point(px + mp["W"], py + mp["H"], w, h, R)
            out[dev] = (s["x"] + min(ax, bx), s["y"] + min(ay, by), (spec["rot"] + R) % 360)
    return out


def device_level(inst, placed):
    """设备级块算例（每台设备一个块、一个姿态）+ 解，用于下界与几何。"""
    blocks_json, sol = [], []
    for did, d in inst["devs"].items():
        t = inst["types"][d["type"]]
        x, y, r = placed[did]
        blocks_json.append({"id": did, "name": did, "rotations": [r],
                            "variants": {"默认": {k: t[k] for k in ("size", "ports", "service_zones")}}})
        sol.append({"x": x, "y": y, "pose": 0})
    data = {"params": {k: inst["P"][k] for k in bk.REQUIRED_PARAMS},
            "blocks": blocks_json,
            "nets": [{"id": n["id"], "terminals": [f"{d}.{p}" for d, p in n["terms"]]} for n in inst["nets"]]}
    blocks, nets, P = base.load_data(data)
    return blocks, nets, P, sol


def scene(inst, blocks, nets, P, sol, keepout=()):
    """布管场景。keepout = 额外的管道禁区盒 [(x0,y0,z0,x1,y1,z1), ...]（mm）。"""
    grid = P["grid"]
    devices = {}
    for b, s in zip(blocks, sol):
        p = b["poses"][0]
        t = inst["types"][inst["devs"][b["id"]]["type"]]
        x0, y0 = s["x"] * grid, s["y"] * grid
        devices[b["id"]] = {
            "box": (x0, y0, 0, x0 + p["W"] * grid, y0 + p["H"] * grid, t["height"]),
            "zones": [(x0 + zx * grid, y0 + zy * grid, x0 + (zx + zw) * grid, y0 + (zy + zh) * grid)
                      for zx, zy, zw, zh in p["zones"]],
            "ports": {pn: (x0 + v[0] * grid, y0 + v[1] * grid, v[4] * grid, v[2], v[3], 0)
                      for pn, v in p["ports"].items()}}
    nets_r = [{"id": e["id"], "terms": [(blocks[i]["id"], pn) for i, pn in e["terms"]]} for e in nets]
    scale = {"A0": P["A0"] * grid ** 2, "L0": P["L0"] * grid, "B0": P["B0"], "C0": len(nets),
             "kappa": P["kappa"], "l_min_mm": P["lmin"] * grid}
    sc = rt.Scene(devices, nets_r, inst["P"]["routing"], inst["P"]["weights"], scale)
    for box in keepout:
        sc.boxes.append((*box, 0.0, None))                     # 禁区按给定尺寸，不再膨胀
    return sc


def placement_margins(rp):
    """摆放时按侧预留出管空间所需的配置（方法 B；见 布管原型/README.md 第 5 节）。"""
    return {"D_mm": rp["D_default_mm"], "delta_pp_mm": rp["delta_pp_mm"],
            "delta_ep_mm": rp["delta_ep_mm"], "c_rho": rp["c_rho"]}
