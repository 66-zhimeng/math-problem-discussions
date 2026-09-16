"""解析摆放示例：谱排序 → max-plus 压紧 → 线性规划微调。

对应《解析求解-可行性与矩阵形式.md》。玩具算例，只摆设备块，不布管。
坐标单位 mm，设备只取 0° 朝向。
"""
import sys

import numpy as np
from scipy.optimize import linprog

sys.stdout.reconfigure(encoding="utf-8")
NEG = -np.inf

# ---------------------------------------------------------------- 算例
# 块：(名称, 宽 W, 深 H)。制冷链模块 = 冷水主机 + 冷冻泵 + 冷却泵，刚性复制 4 份。
blocks = [
    ("冷却塔组", 12000, 3000),
    ("制冷链1", 6000, 4000),
    ("制冷链2", 6000, 4000),
    ("制冷链3", 6000, 4000),
    ("制冷链4", 6000, 4000),
    ("分水器", 2500, 800),
    ("集水器", 2500, 800),
    ("定压补水", 1500, 1200),
]
# 连接：(i, j, 管道根数)
links = [(0, k, 2) for k in range(1, 5)]            # 冷却塔组 ↔ 各链：供、回
links += [(k, 5, 1) for k in range(1, 5)]           # 各链 → 分水器
links += [(k, 6, 1) for k in range(1, 5)]           # 集水器 → 各链
links += [(5, 6, 1), (7, 6, 1)]                     # 旁通、补水
delta = 800                                          # 设备–设备净距 δ_ee

names = [b[0] for b in blocks]
w = np.array([b[1] for b in blocks], float)
h = np.array([b[2] for b in blocks], float)
n = len(blocks)

# ---------------------------------------------------------------- 1. 谱排序
# 拉普拉斯矩阵 L = D − A；min xᵀLx, s.t. xᵀx=1, x⊥1 的解是第 2 小特征向量。
A = np.zeros((n, n))
for i, j, c in links:
    A[i, j] += c
    A[j, i] += c
L = np.diag(A.sum(1)) - A
vals, vecs = np.linalg.eigh(L)
print("拉普拉斯特征值:", np.round(vals, 3))

# 取第 2、3 个特征向量作为相对位置骨架，缩放到总设备面积的量级
S = np.sqrt((w * h).sum()) * 1.5
u = vecs[:, 1]
v = vecs[:, 2]
ux = S * (u - u.min()) / (np.ptp(u) or 1)
vy = S * (v - v.min()) / (np.ptp(v) or 1)

# ---------------------------------------------------------------- 2. 离散骨架：每对设备选一个方向分开
def skeleton_from_coords(px, py, use_size=True):
    """连续坐标：横、纵分离量取较大者；网格坐标：同行左右分、不同行上下分。坐标相同时按编号定先后。"""
    He = np.full((n, n), NEG)   # He[i,j] = w_i + δ 表示 i 在 j 左侧
    Ve = np.full((n, n), NEG)   # Ve[i,j] = h_i + δ 表示 i 在 j 下方
    for i in range(n):
        for j in range(i + 1, n):
            sx = abs(px[j] - px[i]) - (use_size and (w[i] + w[j]) / 2)
            sy = abs(py[j] - py[i]) - (use_size and (h[i] + h[j]) / 2)
            horizontal = sx >= sy if use_size else py[i] == py[j]   # 网格骨架：同一行才左右分开
            if horizontal:
                a, b = (i, j) if (px[i], i) <= (px[j], j) else (j, i)
                He[a, b] = w[a] + delta
            else:
                a, b = (i, j) if (py[i], i) <= (py[j], j) else (j, i)
                Ve[a, b] = h[a] + delta
    return He, Ve


H_edge, V_edge = skeleton_from_coords(ux, vy)


# ---------------------------------------------------------------- 3. max-plus 压紧（解析）
def maxplus_mul(P, Q):
    return np.max(P[:, :, None] + Q[None, :, :], axis=1)


def kleene_star(M):
    """A* = I ⊕ A ⊕ A² ⊕ …，用反复平方计算。I 为对角 0、其余 −∞。"""
    m = M.shape[0]
    R = np.full((m, m), NEG)
    np.fill_diagonal(R, 0.0)
    R = np.maximum(R, M)
    for _ in range(int(np.ceil(np.log2(max(m, 2)))) + 1):
        R = np.maximum(R, maxplus_mul(R, R))
    return R


def longest_path_dp(M):
    """对照：按拓扑序动态规划求最长路。"""
    m = M.shape[0]
    indeg = (M > NEG).sum(0)
    order, stack = [], [k for k in range(m) if indeg[k] == 0]
    indeg = indeg.copy()
    while stack:
        k = stack.pop()
        order.append(k)
        for j in np.where(M[k] > NEG)[0]:
            indeg[j] -= 1
            if indeg[j] == 0:
                stack.append(j)
    assert len(order) == m, "约束图有环"
    x = np.zeros(m)
    for k in order:
        for j in np.where(M[k] > NEG)[0]:
            x[j] = max(x[j], x[k] + M[k, j])
    return x


x0 = np.zeros(n)
x_mp = np.max(x0[:, None] + kleene_star(H_edge), axis=0)   # x = x0 ⊗ A*
y_mp = np.max(x0[:, None] + kleene_star(V_edge), axis=0)
assert np.allclose(x_mp, longest_path_dp(H_edge))
assert np.allclose(y_mp, longest_path_dp(V_edge))

W_mp = (x_mp + w).max()
H_mp = (y_mp + h).max()


def pipe_len(x, y):
    """中心间曼哈顿距离 × 管道根数，仅作管长估计。"""
    cx, cy = x + w / 2, y + h / 2
    return sum(c * (abs(cx[i] - cx[j]) + abs(cy[i] - cy[j])) for i, j, c in links)


def check(x, y):
    for i in range(n):
        for j in range(i + 1, n):
            gap_x = max(x[j] - (x[i] + w[i]), x[i] - (x[j] + w[j]))
            gap_y = max(y[j] - (y[i] + h[i]), y[i] - (y[j] + h[j]))
            assert max(gap_x, gap_y) >= delta - 1e-6, (names[i], names[j])


check(x_mp, y_mp)
print(f"\n[max-plus 压紧] W={W_mp:.0f}  H={H_mp:.0f}  面积={W_mp*H_mp/1e6:.1f} m²  "
      f"管长估计={pipe_len(x_mp, y_mp)/1000:.1f} m")


# ---------------------------------------------------------------- 4. 固定骨架下的线性规划
# 在 W、H 不超过压紧结果的前提下，最小化管长估计。每个轴独立。
def lp_axis(E, size, bound):
    # 变量：坐标 p_0..p_{n-1}，以及每条连接的 |Δ| 辅助变量 s_k
    m = len(links)
    c = np.concatenate([np.zeros(n), [cnt for _, _, cnt in links]])
    rows, rhs = [], []
    for i in range(n):
        for j in range(n):
            if E[i, j] > NEG:                      # p_i + E_ij − p_j ≤ 0
                r = np.zeros(n + m)
                r[i], r[j] = 1, -1
                rows.append(r)
                rhs.append(-E[i, j])
    for k, (i, j, _) in enumerate(links):          # ±(中心差) − s_k ≤ 0
        for sgn in (1, -1):
            r = np.zeros(n + m)
            r[i], r[j], r[n + k] = sgn, -sgn, -1
            rows.append(r)
            rhs.append(-sgn * (size[i] - size[j]) / 2)
    bounds = [(0, bound - size[i]) for i in range(n)] + [(0, None)] * m
    res = linprog(c, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bounds, method="highs")
    assert res.status == 0, res.message
    return res.x[:n]


x_lp = lp_axis(H_edge, w, W_mp)
y_lp = lp_axis(V_edge, h, H_mp)
check(x_lp, y_lp)
W_lp, H_lp = (x_lp + w).max(), (y_lp + h).max()
print(f"[骨架固定 + LP]  W={W_lp:.0f}  H={H_lp:.0f}  面积={W_lp*H_lp/1e6:.1f} m²  "
      f"管长估计={pipe_len(x_lp, y_lp)/1000:.1f} m")

# ---------------------------------------------------------------- 5. 对比：同样的解析步骤，换一个骨架
# 人工骨架（网格列、行）：四条链 2×2；集水器、补水、分水器在中间一排；冷却塔组在最上排。
grid = {0: (1, 3), 1: (0, 0), 2: (2, 0), 3: (0, 2), 4: (2, 2),
        5: (2, 1), 6: (0, 1), 7: (1, 1)}
gx = np.array([grid[k][0] for k in range(n)], float)
gy = np.array([grid[k][1] for k in range(n)], float)
He2, Ve2 = skeleton_from_coords(gx, gy, use_size=False)
x2, y2 = longest_path_dp(He2), longest_path_dp(Ve2)
check(x2, y2)
W2, H2 = (x2 + w).max(), (y2 + h).max()
x2, y2 = lp_axis(He2, w, W2), lp_axis(Ve2, h, H2)
check(x2, y2)
print(f"[人工骨架 + 压紧 + LP] W={W2:.0f}  H={H2:.0f}  面积={W2*H2/1e6:.1f} m²  "
      f"管长估计={pipe_len(x2, y2)/1000:.1f} m")

print("\n设备左下角坐标 (mm)：")
for k in range(n):
    print(f"  {names[k]:<6} x={x_lp[k]:>7.0f}  y={y_lp[k]:>7.0f}")

# ---------------------------------------------------------------- 图
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (x, y, title) in zip(axes, [(x_mp, y_mp, "谱骨架 + max-plus 压紧"), (x_lp, y_lp, "谱骨架 + LP"), (x2, y2, "人工骨架 + 压紧 + LP")]):
        for k in range(n):
            ax.add_patch(Rectangle((x[k], y[k]), w[k], h[k], fill=False, lw=1.5))
            ax.text(x[k] + w[k] / 2, y[k] + h[k] / 2, names[k], ha="center", va="center", fontsize=8)
        for i, j, c in links:
            ax.plot([x[i] + w[i] / 2, x[j] + w[j] / 2], [y[i] + h[i] / 2, y[j] + h[j] / 2],
                    lw=0.6 * c, alpha=0.4)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.autoscale()
    fig.tight_layout()
    fig.savefig("解析摆放示例.png", dpi=120)
    print("\n图已保存：解析摆放示例.png")
except ImportError:
    pass
