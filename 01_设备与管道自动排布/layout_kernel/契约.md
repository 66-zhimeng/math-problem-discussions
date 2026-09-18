# 计算内核契约

外部只通过这一份契约调用内核：一份**任务书**（JSON / dict）进，一份**方案**（JSON / dict）出。

```python
from layout_kernel import solve, validate_case, CaseError

result = solve("任务书.json")        # 或 solve(case_dict)
```

命令行：

```bash
.venv/Scripts/python -m layout_kernel 任务书.json -o 方案.json
```

> `place_and_route` 模式用多进程（spawn），调用方必须放在 `if __name__ == "__main__":` 里。

---

## 1. 原则

1. **参数不设默认值。** 缺任何一项都会抛 `CaseError`，并指出缺的是哪一项。默认值会悄悄改变结果，而这里每个数值都对应工艺规定。
2. **长度单位一律 mm，角度为度**（只允许 0 / 90 / 180 / 270）。坐标都必须是 `params.grid_mm` 的整数倍。
3. **结果里的 J、指标、违规全部来自独立校验器**（`routing.check_routes`、`placement_cpsat.validate`），不依赖求解器内部数据。求解器说自己做到了不算数。
4. **失败是正常返回，不是异常。** `result["ok"] = false` 并在 `violations` 里给出原因；只有任务书不合法才抛 `CaseError`。

## 2. 任务书

```jsonc
{
  "params": {
    "grid_mm": 100,                      // 摆放网格：所有坐标的最小单位
    "delta_ee_mm": 800,                  // 设备—设备最小间距
    "l_min_mm": 300,                     // 两个管件之间的最小直管长度 ℓ_min
    "kappa": 2,                          // 占地矩形允许的最大长宽比
    "service_zones_inside_footprint": false,  // 检修区是否必须落在占地矩形内
    "weights": {"area": 1.0, "length": 1.0, "bends": 0.3, "height_changes": 0.3},
    "routing": { /* 21 项布管参数，见第 5 节 */ },
    "blocking": { /* 9 项分块参数，见第 5 节 */ }
  },

  "device_types": {
    "泵": {
      "height": 1000,                    // 设备高度
      "size": [1200, 800],               // 平面尺寸（未旋转时 x, y）
      "rotations": [0, 90, 180, 270],    // 允许的旋转角
      "service_zones": [[0, -800, 1200, 0]],   // 检修区（相对设备原点的 x0,y0,x1,y1）
      "ports": {
        "in":  {"pos": [0, 400],    "dir": [-1, 0], "z": 500},   // 位置、必须先直出的方向、标高
        "out": {"pos": [1200, 400], "dir": [1, 0],  "z": 500}
      }
    }
  },

  "devices": [
    {"id": "泵1", "type": "泵"},
    {"id": "泵2", "type": "泵", "fixed":  [2000, 2000]},        // 可选：钉死平面位置（mm）
    {"id": "泵3", "type": "泵", "placed": [5000, 3000, 90]}     // route_only 模式必填：位置 + 旋转角
  ],

  "nets": [
    {"id": "供水", "terminals": ["泵1.out", "机组1.in"]}         // 端点 ≥ 3 时自动接三通
  ],

  "keepout": [[x0, y0, z0, x1, y1, z1]],   // 可选：管道禁区（设备可以占，管道不能进）

  "task": {
    "mode": "place_and_route",   // 或 "route_only"
    "time_budget_s": 60,         // 每个候选摆放的搜索秒数
    "candidates": 3,             // 候选摆放数（种子 seed, seed+1, …），逐个完整布管后按真实 J 选
    "seed": 0,
    "workers": 12                // 摆放搜索的进程数（布管的进程数是 params.routing.route_workers）
  }
}
```

`route_only` 模式下 `task` 只需要 `mode`，但每台设备都要给 `placed`。

**注意 `blocking.auto_module_policy`**：设为 `accept` 会把自动识别的重复结构（如冷却塔组、主机—泵链）合并成刚性模块再摆放。实测这样**布管明显更难**：小冷站上 6 个候选全部有违规（5–7 处）；设为 `report_only`（只报告、不合并）时 3 个候选中 2 个零违规，最终方案零违规（见 `分块策略对比_log.txt`）。原因推测是模块内部排法只按设备间距紧凑排列，没有为出管留空间；出管留空只作用在块的外边界。在修好之前建议用 `report_only`。

**注意 `fixed` 的坐标系**：摆放会在设备四周为出管预留空间，所以整个布局不会贴着原点。固定坐标太靠近原点（例如 `[0, 0]`）会与预留空间冲突而无解——这时 `violations` 会直接说明原因。

## 3. 方案（返回值）

```jsonc
{
  "ok": true,                       // 全部布通且独立校验无违规
  "mode": "place_and_route",
  "violations": [],                 // 校验器报出的每一条违规（原文）
  "unrouted": {},                   // 未布通的管网 → 原因
  "metrics": {"W_m": 19.8, "H_m": 16.7, "area_m2": 330.7, "L_m": 383.3,
              "bends": 146, "height_changes": 53, "J": 6.3521},   // 含管道的实际指标
  "per_net": [{"net": "供水", "L_mm": 12345, "bends": 6, "height_changes": 2, "branches": 1}],
  "lower_bound": {"area_m2": ..., "L_lb_m": ..., "bends_lb": ..., "J_full_weights": ...},
  "devices": {"泵1": {"x_mm": 2000, "y_mm": 2000, "rot_deg": 90,
                      "box_mm": [x0,y0,z0,x1,y1,z1],
                      "ports_mm": {"out": [x, y, z]}}},
  "routes": {"供水": [{"start": ["泵1", "out"],
                       "end": ["port", ["机组1", "in"]],      // 或 ["tee", null]：接在本管网的三通上
                       "points_mm": [[x,y,z], ...]}]},        // 折线，相邻点只沿一个轴变化
  "candidates": [{"seed": 0, "ok": true, "J": 6.35, "area_m2": ..., "L_m": ...,
                  "violations": 0, "route_s": 15.2, "placement_s": 60.3}],
  "chosen_seed": 0,
  "route_s": 15.2, "route_iters": 9, "total_s": 91.4
}
```

一个管网可能有多条支路：第一条连接两个端点，其余每条从一个端点接到已布管道上的三通（`end = ["tee", null]`）。

## 4. 目标函数

```
J = w_A · 占地/A₀ + w_L · 管长/L₀ + w_B · 弯头/B₀ + w_C · 高度变化/C₀
```

权重在 `params.weights` 里调：更在意占地就调大 `area`，不想多拐弯就调大 `bends`。当前算例下 1 个弯头相当于约 3.4 m 管长，调权重就是在改这个兑换率。

## 5. 参数清单

`params.routing`（全部必填）：

| 参数 | 含义 |
|---|---|
| `D_default_mm` | 管径（本原型所有管网同径） |
| `c_rho` | 弯曲半径 ρ = c_rho × D |
| `delta_ep_mm` / `delta_pp_mm` | 设备—管 / 管—管最小净距 |
| `K` | 每条连接允许的高度变化次数上限 |
| `eps_z_mm` | 高度容差（校验用） |
| `z_max_mm` | 管顶最高标高（层高） |
| `service_zone_height_mm` | 检修区高度，此高度以下不得走管 |
| `pitch_mm` | 布管网格基础间距 |
| `margin_mm` | 布管区域在设备外接矩形外的扩展 |
| `max_iters` / `stall_iters` | 协商最多轮数 / 连续多少轮无改进就转入清理 |
| `pres_fac_init` / `pres_fac_mult` / `hist_fac` | 协商的拥堵与历史代价系数 |
| `max_expansions` / `astar_weight` | 单次 A* 扩展上限 / 启发式放大系数（1 = 单管最优） |
| `cleanup_trigger_nets` / `cleanup_max_expansions` | 中途试清理的冲突管网上限 / 清理时的扩展上限 |
| `freeze_after_exhausted` | 连续几次搜到上限后，协商中不再重搜该管网 |
| `route_workers` | 并行布管进程数（与 `task.workers` 各管各的，不互相覆盖） |

`params.blocking`（全部必填）：`auto_module_policy`（`accept` / `report_only`）、`min_copies`、`min_members`、`module_rotations`、`module_aspect_bands`、`module_solve_s`、`cluster_max_units`、`cluster_resolution`、`seed`。

## 6. 还没有的能力

以下都还**不在**契约里，按需再加：

| 需求 | 现状 |
|---|---|
| 指定某根管道的走法（锁定已有管路） | 未实现。清理步骤里已有“其他管作硬障碍”的机制，反过来用即可，工作量小 |
| 禁止设备放在某区域 | 未实现（`keepout` 只挡管道），要在摆放的序列对约束里加区域排除，工作量中等 |
| 按单根管设权重 | 未实现，工作量小 |
| 设备之间的相对关系约束（必须相邻、必须靠墙） | 未实现 |
| 变径、斜三通、不同管径 | 未实现，模型本身也还没定义 |
