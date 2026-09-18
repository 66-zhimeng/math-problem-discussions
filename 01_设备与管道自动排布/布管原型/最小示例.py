"""最小可运行示例：给定设备位置与连接关系 → 布管 → 独立校验。

运行（在本目录）：..\\..\\.venv\\Scripts\\python 最小示例.py
所有参数都必须显式给出，缺任何一项都会报 KeyError（项目约定：不设默认值）。
"""
import sys

import routing as rt

# ---------------------------------------------------------------- 参数（单位 mm）
ROUTING = {
    "D_default_mm": 200,            # 管径（本原型所有管网同径）
    "c_rho": 1.5,                   # 弯曲半径 ρ = c_rho × D
    "delta_ep_mm": 100,             # 设备—管最小净距
    "delta_pp_mm": 100,             # 管—管最小净距
    "K": 3,                         # 每条连接允许的高度变化次数上限
    "eps_z_mm": 1,                  # 高度容差（校验用）
    "z_max_mm": 4500,               # 管顶最高标高（层高）
    "service_zone_height_mm": 2000,  # 检修区高度：此高度以下不得走管
    "pitch_mm": 300,                # 布管网格的基础间距
    "margin_mm": 1500,              # 布管区域在设备外接矩形外的扩展
    # —— 协商布线
    "max_iters": 60,                # 协商最多轮数
    "pres_fac_init": 0.5,           # 拥堵代价初值
    "pres_fac_mult": 1.6,           # 每轮拥堵代价放大倍数
    "hist_fac": 0.5,                # 冲突边的历史代价增量
    "max_expansions": 500000,       # 单次 A* 最多扩展状态数
    "astar_weight": 1.5,            # 启发式放大系数（1 = 单管最优，> 1 更快）
    "route_workers": 1,             # 并行布管的进程数（> 1 时须在 __main__ 中调用）
    "stall_iters": 12,              # 连续多少轮无改进就停止协商，转入清理
    "cleanup_trigger_nets": 4,      # 协商中途冲突管网不多于此数时先试清理
    "cleanup_max_expansions": 2000000,  # 清理时单次 A* 的扩展上限
    "freeze_after_exhausted": 2,    # 连续几次搜到上限后，协商中不再重搜该管网
    "port_side_lines": True,        # 端口坐标两侧加密网格线（便于贴近端口走管；场景很大时关掉以控制网格规模）
    "self_skip_mm": 0,              # 同一根管上沿管长相隔超过此值的两段才检查自身净距（0 = 全部检查）
}
WEIGHTS = {"area": 1.0, "length": 1.0, "bends": 0.3, "height_changes": 0.3}   # 目标各项权重
SCALE = {"A0": 1e7, "L0": 1e4, "B0": 25, "C0": 25,   # 归一化尺度：占地 mm²、管长 mm、弯头数、高度变化数
         "kappa": 2,                                  # 占地矩形允许的最大长宽比
         "l_min_mm": 300}                             # 两个管件之间的最小直管长度 ℓ_min

# ---------------------------------------------------------------- 设备与管网
# box = (x0, y0, z0, x1, y1, z1)；zones = 检修区平面矩形 (x0, y0, x1, y1)，高度为 service_zone_height_mm
# ports = {端口名: (x, y, z, ux, uy, uz)}，(ux, uy, uz) 是管道必须先直出的方向（单位向量）
DEVICES = {
    "泵1": {"box": (0, 0, 0, 1200, 800, 1000),
            "zones": [(0, -800, 1200, 0)],                    # 泵前方 0.8 m 检修区
            "ports": {"out": (1200, 400, 500, 1, 0, 0)}},
    "机组": {"box": (5000, 0, 0, 7000, 2500, 2200),
             "zones": [],
             "ports": {"in": (5000, 1200, 800, -1, 0, 0),
                       "out": (7000, 1200, 800, 1, 0, 0)}},
    "冷却塔": {"box": (5000, 5000, 0, 7000, 7000, 3000),
               "zones": [],
               "ports": {"in": (6000, 5000, 1500, 0, -1, 0)}},
}
NETS = [                                   # 端点两个以上时自动接三通
    {"id": "供水", "terms": [("泵1", "out"), ("机组", "in")]},
    {"id": "回水", "terms": [("机组", "out"), ("冷却塔", "in")]},
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sc = rt.Scene(DEVICES, NETS, ROUTING, WEIGHTS, SCALE)
    routes, history, _G = rt.negotiate(sc, log=lambda line: print(line, flush=True))
    viol, met = rt.check_routes(sc, routes)                    # 独立校验：只看折线几何
    print(f"\n布通 {len(routes)}/{len(sc.nets)}，违规 {len(viol)}" + ("" if not viol else "：" + "；".join(viol)))
    print(f"含管道占地 {met['area_m2']} m²，管长 {met['L_m']} m，弯头 {met['bends']}，"
          f"高度变化 {met['height_changes']}，J={met['J']}")
    for nid, brs in routes.items():                            # 每个管网：若干条支路的折线点
        for br in brs:
            pts = "→".join(f"({p[0]:.0f},{p[1]:.0f},{p[2]:.0f})" for p in br["points"])
            print(f"  {nid} 从 {br['start'][0]}.{br['start'][1]}：{pts}")


if __name__ == "__main__":
    main()
