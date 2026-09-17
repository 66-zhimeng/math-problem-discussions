"""生成设备级合成算例（原型自拟格式，之后对接正式拓扑 JSON）。

输出（同目录）：
  设备算例_小_标注.json   小冷站 23 台设备；标注了模块，制冷链给出内部布局
  设备算例_小.json        同上，去掉模块标注（测试自动识别）
  设备算例_大.json        2 个冷站 + 28 个区域，约 550 台设备、约 230 个管网；无模块标注

“_truth” 字段是生成器已知的真实模块与所属系统，只用于评估识别效果，求解程序不读取。
单位 mm；尺寸、坐标为 100 的整数倍；端口方向为外法向。
"""
import copy
import sys
import json
from pathlib import Path

HERE = Path(__file__).parent
sys.stdout.reconfigure(encoding="utf-8")


def port(x, y, dx, dy, z):
    return {"pos": [x, y], "dir": [dx, dy], "z": z}


ROT4 = [0, 90, 180, 270]

TYPES = {
    "冷水主机": {"size": [3000, 1600], "rotations": ROT4, "service_zones": [[0, -1000, 3000, 1000]],
             "ports": {"CHW_in": port(0, 400, -1, 0, 800), "CHW_out": port(0, 1200, -1, 0, 800),
                       "CW_in": port(3000, 400, 1, 0, 800), "CW_out": port(3000, 1200, 1, 0, 800)}},
    "冷冻泵": {"size": [1000, 800], "rotations": ROT4, "service_zones": [[0, 800, 1000, 600]],
            "ports": {"in": port(0, 400, -1, 0, 500), "out": port(1000, 400, 1, 0, 500)}},
    "冷却泵": {"size": [1000, 800], "rotations": ROT4, "service_zones": [[0, 800, 1000, 600]],
            "ports": {"in": port(0, 400, -1, 0, 500), "out": port(1000, 400, 1, 0, 500)}},
    "冷却塔": {"size": [5000, 4000], "rotations": ROT4, "service_zones": [],
            "ports": {"out": port(2500, 0, 0, -1, 500), "in": port(2500, 4000, 0, 1, 500)}},
    "分水器": {"size": [3500, 800], "rotations": ROT4, "service_zones": [[0, 800, 3500, 800]],
            "ports": {**{f"in_{k}": port(400 + 500 * (k - 1), 0, 0, -1, 1000) for k in range(1, 6)},
                      "out_bldg": port(3500, 400, 1, 0, 1000), "bypass": port(0, 400, -1, 0, 1000)}},
    "集水器": {"size": [3500, 800], "rotations": ROT4, "service_zones": [],
            "ports": {**{f"out_{k}": port(400 + 500 * (k - 1), 0, 0, -1, 1000) for k in range(1, 6)},
                      "in_bldg": port(3500, 400, 1, 0, 1000), "bypass": port(0, 400, -1, 0, 1000),
                      "makeup": port(1700, 800, 0, 1, 1000)}},
    "板换": {"size": [3000, 2000], "rotations": ROT4, "service_zones": [[0, -1000, 3000, 1000]],
           "ports": {"p1": port(0, 500, -1, 0, 800), "p2": port(0, 1500, -1, 0, 800),
                     "p3": port(3000, 500, 1, 0, 800), "p4": port(3000, 1500, 1, 0, 800)}},
    "定压补水": {"size": [1500, 1200], "rotations": ROT4, "service_zones": [[0, -800, 1500, 800]],
             "ports": {"in": port(0, 600, -1, 0, 600), "out": port(1500, 600, 1, 0, 600)}},
    "水处理": {"size": [2000, 1500], "rotations": ROT4, "service_zones": [[0, -800, 2000, 800]],
            "ports": {"in": port(0, 700, -1, 0, 600), "out": port(2000, 700, 1, 0, 600)}},
    "软化水箱": {"size": [2000, 2000], "rotations": ROT4, "service_zones": [],
             "ports": {"out": port(2000, 1000, 1, 0, 300)}},
    "建筑立管": {"size": [1000, 1000], "rotations": [0], "service_zones": [],
             "ports": {"supply_in": port(500, 1000, 0, 1, 1000), "return_out": port(1000, 500, 1, 0, 1000)}},
    # 区域
    "区域分水器": {"size": [4000, 800], "rotations": ROT4, "service_zones": [],
              "ports": {"main_in": port(0, 400, -1, 0, 1000), "arr": port(4000, 400, 1, 0, 1000),
                        **{f"o_{k}": port(1000 * k, 0, 0, -1, 1000) for k in range(1, 3)}}},
    "区域集水器": {"size": [4000, 800], "rotations": ROT4, "service_zones": [],
              "ports": {"main_out": port(0, 400, -1, 0, 1000), "arr": port(4000, 400, 1, 0, 1000),
                        **{f"i_{k}": port(1000 * k, 0, 0, -1, 1000) for k in range(1, 3)}}},
    "区域换热器": {"size": [2000, 1200], "rotations": ROT4, "service_zones": [[0, -800, 2000, 800]],
              "ports": {"in": port(0, 600, -1, 0, 800), "out": port(2000, 600, 1, 0, 800)}},
    "区域循环泵": {"size": [1000, 800], "rotations": ROT4, "service_zones": [[0, 800, 1000, 500]],
              "ports": {"in": port(0, 400, -1, 0, 500), "out": port(1000, 400, 1, 0, 500)}},
    "空调箱": {"size": [1500, 1000], "rotations": ROT4, "service_zones": [[0, -700, 1500, 700]],
            "ports": {"in": port(0, 500, -1, 0, 1200), "out": port(1500, 500, 1, 0, 1200)}},
}

PARAMS = {
    "grid_mm": 100, "delta_ee_mm": 800, "l_min_mm": 300, "kappa": 2,
    "service_zones_inside_footprint": False,
    "weights": {"area": 1.0, "length": 1.0, "bends": 0.3},
    "blocking": {
        "auto_module_policy": "accept",          # accept：自动候选直接使用；report_only：只报告，不合并
        "min_copies": 2,                         # 至少重复几次才算模块候选
        "min_members": 2,                        # 链式模块至少几台设备
        "module_rotations": ROT4,                # 模块整体允许的旋转
        "module_aspect_bands": [[1, 1.5], [1.5, 3], [3, 30]],   # 求内部排法时的长宽比区间（每区间一个候选）
        "module_solve_s": 3,                     # 每个候选的 CP-SAT 时间
        "cluster_max_units": 25,                 # 每簇最多几个块
        "cluster_resolution": 1.0,               # Louvain 分辨率
        "seed": 0,
    },
}


class Builder:
    def __init__(self):
        self.devices, self.nets, self.truth_modules, self.system = [], [], [], {}

    def dev(self, did, typ, system, fixed=None):
        d = {"id": did, "type": typ}
        if fixed is not None:
            d["fixed"] = fixed
        self.devices.append(d)
        self.system[did] = system
        return did

    def net(self, nid, *terms):
        self.nets.append({"id": nid, "terminals": list(terms)})


def plant(bd, pre, n_chain, with_riser):
    sysname = f"{pre}冷站"
    for k in range(1, n_chain + 1):
        ch = bd.dev(f"{pre}CH{k}_主机", "冷水主机", sysname)
        pc = bd.dev(f"{pre}CH{k}_冷冻泵", "冷冻泵", sysname)
        pw = bd.dev(f"{pre}CH{k}_冷却泵", "冷却泵", sysname)
        bd.truth_modules.append({"template": "制冷链", "members": {"主机": ch, "冷冻泵": pc, "冷却泵": pw}})
    cts = [bd.dev(f"{pre}CT{k}", "冷却塔", sysname) for k in range(1, n_chain + 1)]
    bd.truth_modules.append({"template": "冷却塔组", "members": {f"冷却塔#{k}": c for k, c in enumerate(cts)}})
    for t, name in (("分水器", "FS"), ("集水器", "JS"), ("板换", "HX"), ("定压补水", "MU"), ("水处理", "WT"),
                    ("软化水箱", "TK")):
        bd.dev(f"{pre}{name}", t, sysname)
    for k in range(1, n_chain + 1):
        c = f"{pre}CH{k}"
        bd.net(f"{pre}冷冻回水{k}", f"{pre}JS.out_{k}", f"{c}_冷冻泵.in")
        bd.net(f"{pre}冷冻泵出{k}", f"{c}_冷冻泵.out", f"{c}_主机.CHW_in")
        bd.net(f"{pre}冷冻供水{k}", f"{c}_主机.CHW_out", f"{pre}FS.in_{k}")
        bd.net(f"{pre}冷却泵出{k}", f"{c}_冷却泵.out", f"{c}_主机.CW_in")
    bd.net(f"{pre}冷却供水", *[f"{c}.out" for c in cts], *[f"{pre}CH{k}_冷却泵.in" for k in range(1, n_chain + 1)],
           f"{pre}HX.p3", f"{pre}WT.in")
    bd.net(f"{pre}冷却回水", *[f"{pre}CH{k}_主机.CW_out" for k in range(1, n_chain + 1)], *[f"{c}.in" for c in cts],
           f"{pre}HX.p4", f"{pre}WT.out")
    bd.net(f"{pre}板换供水", f"{pre}HX.p1", f"{pre}FS.in_5")
    bd.net(f"{pre}板换回水", f"{pre}JS.out_5", f"{pre}HX.p2")
    bd.net(f"{pre}旁通", f"{pre}FS.bypass", f"{pre}JS.bypass")
    bd.net(f"{pre}软水", f"{pre}TK.out", f"{pre}MU.in")
    bd.net(f"{pre}补水", f"{pre}MU.out", f"{pre}JS.makeup")
    if with_riser:
        bd.dev(f"{pre}RS", "建筑立管", sysname, fixed=[0, 0])
        bd.net(f"{pre}供楼", f"{pre}FS.out_bldg", f"{pre}RS.supply_in")
        bd.net(f"{pre}回楼", f"{pre}RS.return_out", f"{pre}JS.in_bldg")


def zone(bd, z, m, r):
    sysname = f"区域{z}"
    fs = bd.dev(f"Z{z}_分水器", "区域分水器", sysname)
    js = bd.dev(f"Z{z}_集水器", "区域集水器", sysname)
    for k in range(1, m + 1):
        hx = bd.dev(f"Z{z}_HX{k}", "区域换热器", sysname)
        pu = bd.dev(f"Z{z}_HX{k}_泵", "区域循环泵", sysname)
        bd.truth_modules.append({"template": "换热单元", "members": {"换热器": hx, "循环泵": pu}})
        bd.net(f"Z{z}_换热供{k}", f"{fs}.o_{k}", f"{hx}.in")
        bd.net(f"Z{z}_换热内{k}", f"{hx}.out", f"{pu}.in")
        bd.net(f"Z{z}_换热回{k}", f"{pu}.out", f"{js}.i_{k}")
    ahus = [bd.dev(f"Z{z}_AHU{k}", "空调箱", sysname) for k in range(1, r + 1)]
    bd.truth_modules.append({"template": f"空调箱组×{r}", "members": {f"空调箱#{k}": a for k, a in enumerate(ahus)}})
    bd.net(f"Z{z}_末端供", f"{fs}.arr", *[f"{a}.in" for a in ahus])
    bd.net(f"Z{z}_末端回", f"{js}.arr", *[f"{a}.out" for a in ahus])
    return fs, js


def dump(name, bd, modules=None, note=""):
    used = {d["type"] for d in bd.devices}
    data = {"说明": note, "params": copy.deepcopy(PARAMS),
            "device_types": {t: TYPES[t] for t in TYPES if t in used},
            "devices": bd.devices, "nets": bd.nets}
    if modules is not None:
        data["modules"] = modules
    data["_truth"] = {"modules": bd.truth_modules, "system": bd.system}
    (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{name}: 设备 {len(bd.devices)}，管网 {len(bd.nets)}")


def main():
    small = Builder()
    plant(small, "", 4, with_riser=True)
    note = "小冷站：4 条制冷链（主机+冷冻泵+冷却泵）、4 台冷却塔、分集水器等 23 台设备。"
    dump("设备算例_小.json", small, None, note + "无模块标注。")
    chain_layout = {"name": "给定排法", "members": {
        "冷冻泵": {"pos": [0, 0], "rot": 0}, "冷却泵": {"pos": [2000, 0], "rot": 0},
        "主机": {"pos": [0, 1800], "rot": 0}}}
    ann = [{"template": "制冷链", "layouts": [chain_layout],
            "instances": [{"id": f"CH{k}", "members": {"主机": f"CH{k}_主机", "冷冻泵": f"CH{k}_冷冻泵",
                                                       "冷却泵": f"CH{k}_冷却泵"}} for k in range(1, 5)]},
           {"template": "冷却塔组",
            "instances": [{"id": "CTG", "members": {f"冷却塔#{k}": f"CT{k + 1}" for k in range(4)}}]}]
    dump("设备算例_小_标注.json", small, ann, note + "制冷链、冷却塔组已标注；制冷链给出内部布局，冷却塔组未给出（由程序求）。")

    big = Builder()
    plant(big, "A", 4, with_riser=False)
    plant(big, "B", 4, with_riser=False)
    mains = {"A": ([], []), "B": ([], [])}
    for z in range(1, 29):
        fs, js = zone(big, z, m=1 + z % 2, r=(10, 12, 14, 16)[z % 4])
        p = "A" if z <= 14 else "B"
        mains[p][0].append(f"{fs}.main_in"); mains[p][1].append(f"{js}.main_out")
    for p, (sup, ret) in mains.items():
        big.net(f"{p}总供", f"{p}FS.out_bldg", *sup)
        big.net(f"{p}总回", f"{p}JS.in_bldg", *ret)
    dump("设备算例_大.json", big, None,
         "大算例：2 个冷站 + 28 个区域（每区 1–2 个换热单元、10–16 台并联空调箱）。无模块标注。")


if __name__ == "__main__":
    main()
