"""信道编码：规则 LDPC（Gallager 构造）+ GF(2) 消元系统编码 + 归一化最小和（NMS）译码。

用法：
>>> code = ldpc_make(1944, 3, 6)                 # 码率约 1/2
>>> c = ldpc_encode(u, code)                     # u (nb, code['k'])
>>> b, ok = ldpc_decode(llr.reshape(nb, -1), code)   # llr 来自 tx.llr（正值 → 比特 0）
>>> u_hat = b[:, code['free']]
"""
import numpy as np


def ldpc_make(n, dv=3, dc=6, rng=None):
    """随机规则 (dv, dc) LDPC（Gallager 构造，n 须为 dc 的整数倍；码率 ≥ 1 − dv/dc）。

    返回 dict: H (m, n) 校验矩阵; cn (m, dc) 各校验节点相连的变量下标; ve (n, dv) 各变量节点的边下标;
              piv/free 消元后的校验位/信息位列; P 校验位生成矩阵; k 信息位数
    """
    rng = np.random.default_rng(rng)
    base = np.arange(n).reshape(-1, dc)
    cn = np.vstack([base] + [rng.permutation(n)[base] for _ in range(dv - 1)])
    H = np.zeros((len(cn), n), bool)
    H[np.arange(len(cn))[:, None], cn] = True
    R, piv, r = H.copy(), [], 0                             # GF(2) 高斯消元 -> 行最简形
    for c in range(n):
        rows = np.flatnonzero(R[r:, c]) + r
        if len(rows) == 0:
            continue
        R[[r, rows[0]]] = R[[rows[0], r]]
        R[np.flatnonzero(R[:, c] & (np.arange(len(R)) != r))] ^= R[r]
        piv.append(c)
        r += 1
        if r == len(R):
            break
    free = np.setdiff1d(np.arange(n), piv)
    ve = np.argsort(cn.ravel(), kind='stable').reshape(n, dv)
    return dict(H=H, cn=cn, ve=ve, piv=np.array(piv), free=free, P=R[:r][:, free].astype(int), k=len(free))


def ldpc_encode(u, code):
    """系统编码：码字信息位 = u，校验位 = P·u (mod 2)。u (nb, k) -> 码字 (nb, n)。"""
    u = np.atleast_2d(u)
    c = np.zeros((len(u), code['H'].shape[1]), np.uint8)
    c[:, code['free']] = u
    c[:, code['piv']] = (u @ code['P'].T) % 2
    return c


def ldpc_decode(llr, code, iters=30, alpha=0.75):
    """归一化最小和译码（flooding 调度，满足全部校验即提前停止）。

    参数: llr (nb, n) 信道 LLR = log P(b=0)/P(b=1); alpha 归一化因子
    返回: (硬判决码字 (nb, n), 每个码字是否满足全部校验 (nb,))
    """
    cn, ve = code['cn'], code['ve']
    L = np.atleast_2d(llr).astype(float)
    c2v = np.zeros(L.shape[:1] + cn.shape)
    for _ in range(iters):
        tot = L + c2v.reshape(len(L), -1)[:, ve].sum(-1)
        b = tot < 0
        ok = ~np.any(np.sum(b[:, cn], -1) % 2, -1)
        if ok.all():
            break
        v2c = tot[:, cn] - c2v
        a = np.abs(v2c)
        i1 = np.argmin(a, -1)[..., None]
        m1 = np.take_along_axis(a, i1, -1)
        np.put_along_axis(a, i1, np.inf, -1)
        m2 = a.min(-1, keepdims=True)
        sg = np.where(v2c < 0, -1.0, 1.0)
        c2v = alpha * np.prod(sg, -1, keepdims=True) * sg * np.where(np.arange(cn.shape[1]) == i1, m2, m1)
    return b.astype(np.uint8), ok
