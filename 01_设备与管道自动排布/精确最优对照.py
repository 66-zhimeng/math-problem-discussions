"""精确对照：对《解析摆放示例.py》的 8 块算例求真正的最小占地。

形式化：析取约束的混合整数线性规划（MILP，HiGHS 求解）。
- 每对设备 4 选 1：左、右、下、上（大 M 编码）
- 固定宽度上限 W̄，最小化 H；扫描 W̄，取 W·H 最小者
只考虑占地，0° 朝向，与示例的比较口径一致。
"""
import sys
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

sys.stdout.reconfigure(encoding="utf-8")

blocks = [("冷却塔组", 12000, 3000)] + [(f"制冷链{k}", 6000, 4000) for k in range(1, 5)] + [
    ("分水器", 2500, 800), ("集水器", 2500, 800), ("定压补水", 1500, 1200)]
w = np.array([b[1] for b in blocks], float)
h = np.array([b[2] for b in blocks], float)
n, delta = len(blocks), 800.0
pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
CHAINS = [1, 2, 3, 4]


def solve_min_H(Wbar, symmetry_breaking):
    # 变量：x(n), y(n), H, z(4·|pairs|)
    nv = 2 * n + 1 + 4 * len(pairs)
    X = lambda i: i
    Y = lambda i: n + i
    HV = 2 * n
    Z = lambda p, k: 2 * n + 1 + 4 * p + k
    M = Wbar + 40000.0
    rows, lo, hi = [], [], []

    def add(coef, l, u):
        r = np.zeros(nv)
        for idx, val in coef:
            r[idx] += val
        rows.append(r); lo.append(l); hi.append(u)

    for p, (i, j) in enumerate(pairs):
        # k=0: i 在 j 左  x_i + w_i + δ ≤ x_j + M(1−z)
        add([(X(i), 1), (X(j), -1), (Z(p, 0), M)], -np.inf, M - w[i] - delta)
        add([(X(j), 1), (X(i), -1), (Z(p, 1), M)], -np.inf, M - w[j] - delta)
        add([(Y(i), 1), (Y(j), -1), (Z(p, 2), M)], -np.inf, M - h[i] - delta)
        add([(Y(j), 1), (Y(i), -1), (Z(p, 3), M)], -np.inf, M - h[j] - delta)
        add([(Z(p, k), 1) for k in range(4)], 1, np.inf)
    for i in range(n):
        add([(Y(i), 1), (HV, -1)], -np.inf, -h[i])
    if symmetry_breaking:
        # 4 条链完全相同、可互换：规定字典序 x_k + y_k/10⁵ 递增，去掉 4! 个等价解
        for a, b in zip(CHAINS, CHAINS[1:]):
            add([(X(a), 1), (Y(a), 1e-5), (X(b), -1), (Y(b), -1e-5)], -np.inf, 0)

    c = np.zeros(nv); c[HV] = 1
    ub = np.full(nv, np.inf)
    ub[:n] = Wbar - w
    ub[2 * n + 1:] = 1
    integrality = np.zeros(nv); integrality[2 * n + 1:] = 1
    if np.any(ub[:n] < 0):
        return None
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi),
               bounds=Bounds(np.zeros(nv), ub), integrality=integrality,
               options={"time_limit": 60})
    if res.status != 0:
        return None
    x, y = res.x[:n], res.x[n:2 * n]
    return (x + w).max(), res.x[HV], x, y


def check(x, y):
    for i, j in pairs:
        gx = max(x[j] - (x[i] + w[i]), x[i] - (x[j] + w[j]))
        gy = max(y[j] - (y[i] + h[i]), y[i] - (y[j] + h[j]))
        assert max(gx, gy) >= delta - 1e-3


for sb in (False, True):
    t0 = time.time()
    best = None
    for Wbar in np.arange(12000, 30001, 100):
        r = solve_min_H(Wbar, sb)
        if r is None:
            continue
        W_, H_, x, y = r
        if best is None or W_ * H_ < best[0] * best[1] - 1e-6:
            best = (W_, H_, x, y)
    W_, H_, x, y = best
    check(x, y)
    print(f"对称性破除={sb!s:<5}  最优 W={W_:.0f}  H={H_:.0f}  面积={W_*H_/1e6:.1f} m²  "
          f"用时 {time.time()-t0:.1f} s")

print("\n最优布局左下角坐标 (mm)：")
for k in range(n):
    print(f"  {blocks[k][0]:<6} x={x[k]:>7.0f}  y={y[k]:>7.0f}")
