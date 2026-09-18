"""单根管 A* 的编译实现（numba）。规则与 routing.astar_py 完全相同，用来提速；
两者在测试中逐例比对（同样的扩展数与同样的折线）。

数据全部为 numpy 数组：
  网格    cx, cy, cz（各轴坐标）、n0, n1, n2、blocked（节点）、eb0/eb1/eb2（边，按下端节点）
  拥堵    halo（int32）、hist（float32）、pres
  豁免    ex_nodes（排序后的节点号）、ex_edges（排序后的边号 = 下端节点×3 + 轴）
  目标    mode=0 端口：t、t_dir、tpt、ta、ts；mode=1 接树：hl（每个节点的启发式）与树上可接点的数组
状态    (节点, 方向, 上个管件是端口, 高度变化次数, 已有水平段, 自上个管件起直管长度, g)
"""
import numpy as np
from numba import njit
from numba.core import types
from numba.typed import Dict

DIR_AXIS = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
DIR_SIGN = np.array([1, -1, 1, -1, 1, -1], dtype=np.int64)


@njit(cache=True)
def _find(arr, v):
    i = np.searchsorted(arr, v)
    return i < arr.size and arr[i] == v


@njit(cache=True)
def _lt(hf, hs, i, j):
    """堆序：先比 f，再比状态号（与 Python 版 heapq 的插入序 tie-break 一致）。"""
    return hf[i] < hf[j] or (hf[i] == hf[j] and hs[i] < hs[j])


@njit(cache=True)
def _sift_up(hf, hs, i):
    while i > 0:
        p = (i - 1) // 2
        if not _lt(hf, hs, i, p):
            break
        hf[p], hf[i] = hf[i], hf[p]
        hs[p], hs[i] = hs[i], hs[p]
        i = p


@njit(cache=True)
def _sift_down(hf, hs, size):
    i = 0
    while True:
        l, r, m = 2 * i + 1, 2 * i + 2, i
        if l < size and _lt(hf, hs, l, m):
            m = l
        if r < size and _lt(hf, hs, r, m):
            m = r
        if m == i:
            return
        hf[m], hf[i] = hf[i], hf[m]
        hs[m], hs[i] = hs[i], hs[m]
        i = m


@njit(cache=True)
def search(cx, cy, cz, n0, n1, n2, blocked, eb0, eb1, eb2, halo, hist, pres,
           ex_nodes, ex_edges, rho, lmin, K, cL, cB, cC, hw, max_exp,
           s, d0, mode, t, t_dir, tpt, hl, tnode, ttype, tseg_axis, tdA, tdB, tkA, tkB, tcorner):
    """返回 (状态节点, 状态方向, 父状态, 目标状态号, 扩展数, 是否到上限)；目标状态号 < 0 表示没找到。"""
    cap = 2.0 * rho + lmin
    stride = np.array([n1 * n2, n2, 1], dtype=np.int64)
    nn = np.array([n0, n1, n2], dtype=np.int64)
    ta = DIR_AXIS[t_dir]
    ts = DIR_SIGN[t_dir]

    cap_st = 1 << 16
    st_node = np.empty(cap_st, dtype=np.int64)
    st_dir = np.empty(cap_st, dtype=np.int64)
    st_lp = np.empty(cap_st, dtype=np.int64)
    st_vr = np.empty(cap_st, dtype=np.int64)
    st_hh = np.empty(cap_st, dtype=np.int64)
    st_run = np.empty(cap_st, dtype=np.float64)
    st_g = np.empty(cap_st, dtype=np.float64)
    st_par = np.empty(cap_st, dtype=np.int64)
    st_goal = np.empty(cap_st, dtype=np.int64)
    ns = 0

    cap_h = 1 << 16
    hf = np.empty(cap_h, dtype=np.float64)
    hs = np.empty(cap_h, dtype=np.int64)
    nh = 0

    # 支配检查：key → 链表头；链表存 (run, g)
    head = Dict.empty(types.int64, types.int64)
    ln_run = np.empty(cap_st, dtype=np.float64)
    ln_g = np.empty(cap_st, dtype=np.float64)
    ln_next = np.empty(cap_st, dtype=np.int64)
    ln_dead = np.zeros(cap_st, dtype=np.uint8)
    nl = 0

    exp = 0
    exhausted = False
    goal_sid = -1

    # ---- 初始状态
    hh0 = 1 if DIR_AXIS[d0] != 2 else 0
    key = (((s * 6 + d0) * 2 + 1) * (K + 1) + 0) * 2 + hh0
    head[key] = nl
    ln_run[nl] = 0.0; ln_g[nl] = 0.0; ln_next[nl] = -1; ln_dead[nl] = 0; nl += 1
    st_node[ns] = s; st_dir[ns] = d0; st_lp[ns] = 1; st_vr[ns] = 0; st_hh[ns] = hh0
    st_run[ns] = 0.0; st_g[ns] = 0.0; st_par[ns] = -1; st_goal[ns] = 0; ns += 1
    if mode == 0:
        p0 = np.array([cx[s // stride[0]], cy[(s // n2) % n1], cz[s % n2]])
        hv = cL * (abs(p0[0] - tpt[0]) + abs(p0[1] - tpt[1]) + abs(p0[2] - tpt[2]))
        a0 = DIR_AXIS[d0]
        if a0 == ta:
            if DIR_SIGN[d0] != ts:
                hv += cB * 2
            else:
                al = True
                for b in range(3):
                    if b != a0 and abs(p0[b] - tpt[b]) > 1e-9:
                        al = False
                if not (al and (tpt[a0] - p0[a0]) * ts >= 0):
                    hv += cB * 2
        else:
            hv += cB
    else:
        hv = hl[s]
    hf[0] = hw * hv; hs[0] = 0; nh = 1

    while nh > 0:
        sid = hs[0]
        nh -= 1
        hf[0] = hf[nh]; hs[0] = hs[nh]
        _sift_down(hf, hs, nh)
        if st_goal[sid] > 0:
            goal_sid = sid
            break
        exp += 1
        if exp > max_exp:
            exhausted = True
            break
        n = st_node[sid]; d = st_dir[sid]; lp = st_lp[sid]; vr = st_vr[sid]
        hh = st_hh[sid]; run = st_run[sid]; g = st_g[sid]
        i0 = n // stride[0]; i1 = (n // n2) % n1; i2 = n % n2
        ijk = np.array([i0, i1, i2], dtype=np.int64)
        need_turn = (0.0 if lp == 1 else rho) + rho + lmin
        for nd in range(6):
            if nd != d and DIR_AXIS[nd] == DIR_AXIS[d]:
                continue                                           # 不允许掉头
            turning = nd != d
            if turning and run < need_turn - 1e-9:
                continue
            a = DIR_AXIS[nd]; sg = DIR_SIGN[nd]
            m = ijk[a] + sg
            if m < 0 or m >= nn[a]:
                continue
            nb = n + sg * stride[a]
            lower = n if sg > 0 else nb
            if a == 0:
                ln = abs(cx[m] - cx[ijk[0]]); ebv = eb0[lower]
            elif a == 1:
                ln = abs(cy[m] - cy[ijk[1]]); ebv = eb1[lower]
            else:
                ln = abs(cz[m] - cz[ijk[2]]); ebv = eb2[lower]
            if ebv == 1 and not _find(ex_edges, lower * 3 + a):
                continue
            if blocked[nb] == 1 and not _find(ex_nodes, nb):
                continue
            ng = g
            nlp = lp; nvr = vr; nhh = hh
            if turning:
                ng += cB
                nlp = 0
                nrun = ln
                if DIR_AXIS[d] == 2 and a != 2 and hh == 1:
                    nvr += 1
                    ng += cC
                    if nvr > K:
                        continue
            else:
                nrun = run + ln
                if nrun > cap:
                    nrun = cap
            if a != 2:
                nhh = 1
            ei = lower * 3 + a
            ng += cL * ln * (1.0 + hist[ei]) * (1.0 + pres * halo[ei])
            goal = 0
            if mode == 0:
                if nb == t:
                    if nd == t_dir and nrun >= (0.0 if nlp == 1 else rho) + lmin - 1e-9:
                        goal = 1
                    else:
                        continue
            else:
                ti = np.searchsorted(tnode, nb)
                if ti < tnode.size and tnode[ti] == nb:
                    run_ok = nrun >= (0.0 if nlp == 1 else rho) + rho + lmin - 1e-9
                    ok = False
                    if ttype[ti] == 0:
                        dA = tdA[ti]; dB = tdB[ti]
                        ok = (run_ok and a != tseg_axis[ti]
                              and dA >= (0.0 if tkA[ti] == 0 else rho) + rho + lmin - 1e-9
                              and dB >= (0.0 if tkB[ti] == 0 else rho) + rho + lmin - 1e-9)
                    elif ttype[ti] == 1:
                        ok = run_ok and (tcorner[ti] >> nd) & 1 == 1
                    if not ok:
                        continue
                    goal = 1
            # ---- 支配检查后入堆
            key = (((nb * 6 + nd) * 2 + nlp) * (K + 1) + nvr) * 2 + nhh
            skip = False
            if key in head:
                idx = head[key]
                prev = -1
                while idx >= 0:
                    nxt = ln_next[idx]
                    if ln_dead[idx] == 0:
                        if ln_run[idx] >= nrun and ln_g[idx] <= ng + 1e-15:
                            skip = True
                            break
                        if nrun >= ln_run[idx] and ng <= ln_g[idx]:
                            ln_dead[idx] = 1                        # 被新状态支配，剔除
                            if prev < 0:
                                head[key] = nxt
                            else:
                                ln_next[prev] = nxt
                            idx = nxt
                            continue
                    prev = idx
                    idx = nxt
            if skip:
                continue
            if nl >= ln_run.size:
                ln_run = np.concatenate((ln_run, np.empty(nl, dtype=np.float64)))
                ln_g = np.concatenate((ln_g, np.empty(nl, dtype=np.float64)))
                ln_next = np.concatenate((ln_next, np.empty(nl, dtype=np.int64)))
                ln_dead = np.concatenate((ln_dead, np.zeros(nl, dtype=np.uint8)))
            ln_run[nl] = nrun; ln_g[nl] = ng; ln_dead[nl] = 0
            ln_next[nl] = head[key] if key in head else -1
            head[key] = nl
            nl += 1
            if ns >= st_node.size:
                st_node = np.concatenate((st_node, np.empty(ns, dtype=np.int64)))
                st_dir = np.concatenate((st_dir, np.empty(ns, dtype=np.int64)))
                st_lp = np.concatenate((st_lp, np.empty(ns, dtype=np.int64)))
                st_vr = np.concatenate((st_vr, np.empty(ns, dtype=np.int64)))
                st_hh = np.concatenate((st_hh, np.empty(ns, dtype=np.int64)))
                st_run = np.concatenate((st_run, np.empty(ns, dtype=np.float64)))
                st_g = np.concatenate((st_g, np.empty(ns, dtype=np.float64)))
                st_par = np.concatenate((st_par, np.empty(ns, dtype=np.int64)))
                st_goal = np.concatenate((st_goal, np.empty(ns, dtype=np.int64)))
            st_node[ns] = nb; st_dir[ns] = nd; st_lp[ns] = nlp; st_vr[ns] = nvr; st_hh[ns] = nhh
            st_run[ns] = nrun; st_g[ns] = ng; st_par[ns] = sid; st_goal[ns] = goal
            if goal == 1:
                f = ng
            else:
                if mode == 0:
                    j0 = nb // stride[0]; j1 = (nb // n2) % n1; j2 = nb % n2
                    px = cx[j0]; py = cy[j1]; pz = cz[j2]
                    hv = cL * (abs(px - tpt[0]) + abs(py - tpt[1]) + abs(pz - tpt[2]))
                    if a == ta:
                        if sg != ts:
                            hv += cB * 2
                        else:
                            pa = px if a == 0 else (py if a == 1 else pz)
                            al = True
                            if a != 0 and abs(px - tpt[0]) > 1e-9:
                                al = False
                            if a != 1 and abs(py - tpt[1]) > 1e-9:
                                al = False
                            if a != 2 and abs(pz - tpt[2]) > 1e-9:
                                al = False
                            if not (al and (tpt[a] - pa) * ts >= 0):
                                hv += cB * 2
                    else:
                        hv += cB
                else:
                    hv = hl[nb]
                f = ng + hw * hv
            if nh >= hf.size:
                hf = np.concatenate((hf, np.empty(nh, dtype=np.float64)))
                hs = np.concatenate((hs, np.empty(nh, dtype=np.int64)))
            hf[nh] = f; hs[nh] = ns; nh += 1
            _sift_up(hf, hs, nh - 1)
            ns += 1
    return st_node[:ns], st_dir[:ns], st_par[:ns], goal_sid, exp, exhausted
