"""任务书（输入契约）的校验。契约说明见 layout_kernel/契约.md。

原则：所有参数必须显式给出，缺失即报错（不设默认值）；报错要指出是哪台设备、哪个端口、哪一项参数。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "分块原型") not in sys.path:
    sys.path.insert(0, str(ROOT / "分块原型"))
if str(ROOT / "布管原型") not in sys.path:
    sys.path.insert(0, str(ROOT / "布管原型"))

import blocking as bk  # noqa: E402
import routing as rt  # noqa: E402

MODES = ("place_and_route", "route_only")
TASK_REQUIRED = {"place_and_route": ["mode", "time_budget_s", "candidates", "seed", "workers"],
                 "route_only": ["mode"]}
ROTATIONS = (0, 90, 180, 270)


class CaseError(ValueError):
    """任务书不合法。"""


def _box(v, where):
    if not (isinstance(v, (list, tuple)) and len(v) == 6):
        raise CaseError(f"{where}：应为 6 个数 [x0,y0,z0,x1,y1,z1]（mm），实际为 {v!r}")
    if any(v[i] >= v[i + 3] for i in range(3)):
        raise CaseError(f"{where}：盒子的下界必须小于上界，实际为 {v!r}")
    return tuple(float(x) for x in v)


def _check_types(types, grid):
    """设备类型的尺寸、检修区、端口坐标都必须落在网格上（否则底层求解器才报错，信息不清楚）。"""
    for name, t in types.items():
        for k in ("height", "size", "rotations", "service_zones", "ports"):
            if k not in t:
                raise CaseError(f"设备类型 {name} 缺少 {k}")
        for i, v in enumerate(t["size"]):
            if v % grid:
                raise CaseError(f"设备类型 {name} 的 size[{i}] = {v} 不是网格 {grid} mm 的整数倍")
        for zi, z in enumerate(t["service_zones"]):
            for ci, c in enumerate(z):
                if c % grid:
                    raise CaseError(f"设备类型 {name} 的 service_zones[{zi}][{ci}] = {c} 不是网格 {grid} mm 的整数倍")
        for pn, p in t["ports"].items():
            for k in ("pos", "dir", "z"):
                if k not in p:
                    raise CaseError(f"设备类型 {name} 的端口 {pn} 缺少 {k}")
            if tuple(p["dir"]) not in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                raise CaseError(f"设备类型 {name} 的端口 {pn} 的 dir 只能是 ±X 或 ±Y 的单位向量，实际为 {p['dir']}")
            for ci, c in enumerate(list(p["pos"]) + [p["z"]]):
                if c % grid:
                    raise CaseError(f"设备类型 {name} 的端口 {pn} 坐标 {c} 不是网格 {grid} mm 的整数倍"
                                    f"（pos 与 z 都要对齐网格）")


def validate_case(case):
    """检查任务书并返回规范化后的 (inst, task, keepout)。inst 为 blocking.load_devices 的结果。"""
    if not isinstance(case, dict):
        raise CaseError("任务书应为 dict（或 JSON 文件的内容）")
    for k in ("params", "device_types", "devices", "nets", "task"):
        if k not in case:
            raise CaseError(f"任务书缺少 {k}")
    task = case["task"]
    mode = task.get("mode")
    if mode not in MODES:
        raise CaseError(f"task.mode 只能是 {MODES} 之一，实际为 {mode!r}")
    miss = [k for k in TASK_REQUIRED[mode] if k not in task]
    if miss:
        raise CaseError(f"task 缺少 {miss}（不设默认值）")
    if mode == "place_and_route":
        if task["candidates"] < 1:
            raise CaseError("task.candidates 至少为 1")
        if task["workers"] < 1:
            raise CaseError("task.workers 至少为 1")
        if task["time_budget_s"] <= 0:
            raise CaseError("task.time_budget_s 必须为正数")
    rp = case["params"].get("routing")
    if rp is None:
        raise CaseError("params 缺少 routing（布管参数）")
    rt.check_params(rp, case["params"].get("weights", {}))            # 布管参数与权重齐全性
    try:
        inst = bk.load_devices(case)                                   # 设备、端口、管网的合法性
    except (KeyError, ValueError) as ex:
        raise CaseError(str(ex)) from None
    grid = inst["P"]["grid_mm"]
    _check_types(case["device_types"], grid)
    for d in case["devices"]:
        for key in ("fixed", "placed"):
            if key not in d:
                continue
            v = d[key]
            n = 3 if key == "placed" else 2
            if not (isinstance(v, (list, tuple)) and len(v) == n):
                raise CaseError(f"设备 {d['id']} 的 {key} 应为 {n} 个数"
                                f"{'（x_mm, y_mm, 旋转角）' if n == 3 else '（x_mm, y_mm）'}，实际为 {v!r}")
            for c in v[:2]:
                if c % grid:
                    raise CaseError(f"设备 {d['id']} 的 {key} 坐标 {c} 不是网格 {grid} mm 的整数倍")
            if key == "placed" and v[2] not in ROTATIONS:
                raise CaseError(f"设备 {d['id']} 的 placed 旋转角只能是 {ROTATIONS}，实际为 {v[2]!r}")
    if mode == "route_only":
        missing = [d["id"] for d in case["devices"] if "placed" not in d]
        if missing:
            raise CaseError(f"route_only 模式下每台设备都要给 placed（缺：{missing}）")
    keepout = [_box(b, f"keepout[{i}]") for i, b in enumerate(case.get("keepout", []))]
    return inst, task, keepout
