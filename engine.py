# -*- coding: utf-8 -*-
"""
核心碰撞风险引擎 —— 从原始脚本重构，参数化 + 修正笔误。

原始脚本（向南遍历直飞/向北遍历直飞/向北复飞/向南尾流）均为硬编码：
  参数写死在代码顶部、输出路径写死 D:\\、无函数封装。
本模块把算法抽成可传参函数，供 Streamlit UI 调用。

公式（与《参赛作品说明书》一致）：
    P = Px · Py · Pz · N
    Px = Φ(λx; |Δx|+μ, sqrt(σ1²+σ2²)) − Φ(−λx; |Δx|+μ, sqrt(σ1²+σ2²))
    Py、Pz 同理（Pz 用 sqrt(σ3²+σ3²)）
    N 为单位时间航班架次（默认 1）

已修正原始脚本中的两处笔误：
    1) 进近避让风险_final.py:283  pz.append(P3-P4)  → 应为 P5-P6
    2) 向北复飞.py:220           Ldz.append(ldy)   → 应为 ldz
另注：原“直飞遍历”脚本中总风险有 /100 归一化，本模块用显式 N 参数替代（默认 1）。
"""
import math
import numpy as np
import pandas as pd
from scipy.stats import norm

PI = math.pi
TLS = 5e-9          # ICAO 安全目标水平（次/飞行小时）
TURN_CONST = 562.0  # 转弯率常数（坡度 15°）
TAS_MAX = 200 * 1.852 / 3.6   # 速度上限 200 节 → m/s


def degch(x):
    return PI * x / 180


# ---------------- 跑道几何（成都天府） ----------------
FAF_LEN = 14155.0    # FAF 点到跑道头距离 (m)
RUNWAY_ELEV = 441.7  # 跑道入口标高 (m)
ELEV_OFF = 16.5      # 下滑道过跑道头高差 (m)
DEP_Y_OFF = 340.0    # 离场航迹侧向错开 (m)
DEP_Z0 = 437.0       # 离场起始高度 (m)


def getz(l):
    """距离跑道头 l 米处的下滑道高度 (m)"""
    return l * np.tan(degch(3)) + ELEV_OFF + RUNWAY_ELEV


def get_R(V1, a1, t1):
    """转弯率 gama(deg/s) 与转弯半径 R(m)，坡度 15°"""
    TAS = V1 - a1 * t1
    if TAS > TAS_MAX:
        TAS = TAS_MAX
    gama = TURN_CONST * np.tan(degch(15)) / TAS
    R = 180 * TAS / (PI * gama)
    return gama, R


# ---------------- 进近避让航空器轨迹（直飞，3 段） ----------------
def approach_avoidance_traj(V0, a1, a2, SITA1, SITA2, T1, ALFA, X0, Z0,
                            arrt, t_end, step):
    """进近机从 arrt 时刻开始避让，返回 (Lx, Ly, Lz)。
    第1段：继续下滑减速；第2段：以 15° 坡度转弯；第3段：以 ALFA 改出角直飞爬升。"""
    V1 = V0 - a1 * arrt
    x1 = X0 - (V0 * arrt - 0.5 * a1 * arrt * arrt)
    z1 = Z0 - (V0 * arrt - 0.5 * a1 * arrt * arrt) * np.tan(degch(SITA1))
    gama, R = get_R(V1, a1, T1)
    T2 = ALFA / gama + T1
    Lx, Ly, Lz = [], [], []
    for t in np.arange(arrt, t_end, step):
        dt = t - arrt
        if dt < T1:
            x = x1 - (V1 * dt - 0.5 * a1 * dt * dt)
            y = 0.0
            z = z1 - (V1 * dt - 0.5 * a1 * dt * dt) * np.tan(degch(SITA1))
        elif T1 <= dt < T2:
            x = x1 - (V1 * T1 - 0.5 * a1 * T1 * T1) - R * np.sin(degch(gama * (dt - T1)))
            y = R - R * np.cos(degch(gama * (dt - T1)))
            z = z1 - (V1 * T1 - 0.5 * a1 * T1 * T1) * np.tan(degch(SITA1))
        else:
            V2 = V1 - a1 * T1
            s = V2 * (dt - T2) + 0.5 * a2 * (dt - T2) ** 2
            x = (x1 - (V1 * T1 - 0.5 * a1 * T1 * T1)
                 - R * np.sin(degch(ALFA)) - s * np.cos(degch(ALFA)))
            y = R - R * np.cos(degch(ALFA)) + s * np.sin(degch(ALFA))
            z = (z1 - (V1 * T1 - 0.5 * a1 * T1 * T1) * np.tan(degch(SITA1))
                 + s * SITA2)
        Lx.append(x); Ly.append(y); Lz.append(z)
    return Lx, Ly, Lz


# ---------------- 离场航空器轨迹（直线，3 段） ----------------
def departure_straight_traj(ad1, Vlof, ad2, V2_10, ad3, tansitaD2, tansitaD3,
                            x_off, t_end, step, TD1=None, TD2=None):
    """侧向跑道离场，3 段：松刹车-离地 / 离地-收起落架 / 收起落架-离场程序点。"""
    if TD1 is None:
        TD1 = Vlof / ad1
    if TD2 is None:
        TD2 = (V2_10 - Vlof) / ad2 + TD1
    Ldx, Ldy, Ldz = [], [], []
    for t in np.arange(0, t_end, step):
        if t < TD1:
            x = x_off
            y = DEP_Y_OFF + 0.5 * ad1 * t * t
            z = DEP_Z0
        elif t < TD2:
            s1 = Vlof * (t - TD1) + 0.5 * ad2 * (t - TD1) ** 2
            x = x_off
            y = DEP_Y_OFF + Vlof * Vlof / (2 * ad1) + s1
            z = DEP_Z0 + s1 * tansitaD2
        else:
            s2 = V2_10 * (t - TD2) + 0.5 * ad3 * (t - TD2) ** 2
            x = x_off
            y = (DEP_Y_OFF + Vlof * Vlof / (2 * ad1)
                 + (V2_10 ** 2 - Vlof ** 2) / (2 * ad2) + s2)
            z = (DEP_Z0 + (V2_10 ** 2 - Vlof ** 2) / (2 * ad2) * tansitaD2
                 + s2 * tansitaD3)
        Ldx.append(x); Ldy.append(y); Ldz.append(z)
    return Ldx, Ldy, Ldz


# ---------------- 碰撞概率 ----------------
def collision_probs(Lx, Ly, Lz, Ldx, Ldy, Ldz,
                    lamdax, lamday, lamdaz, u1x, sigma1, sigma2, sigma3):
    """逐时刻计算三方向碰撞概率。

    返回 (Px_max, Py_max, Pz_max, Pt_max)，其中
    Pt_max = max_k { Px_k · Py_k · Pz_k }（逐时刻乘积取最大，同原始遍历脚本）。"""
    n = min(len(Lx), len(Ldx))
    s_xy = math.sqrt(sigma1 ** 2 + sigma2 ** 2)
    s_z = math.sqrt(sigma3 ** 2 + sigma3 ** 2)
    px = py = pz = pt = 0.0
    for k in range(n):
        dx = abs(Lx[k] - Ldx[k]); dy = abs(Ly[k] - Ldy[k]); dz = abs(Lz[k] - Ldz[k])
        Px = norm.cdf(lamdax, loc=dx + u1x, scale=s_xy) - norm.cdf(-lamdax, loc=dx + u1x, scale=s_xy)
        Py = norm.cdf(lamday, loc=dy + u1x, scale=s_xy) - norm.cdf(-lamday, loc=dy + u1x, scale=s_xy)
        Pz = norm.cdf(lamdaz, loc=dz + u1x, scale=s_z) - norm.cdf(-lamdaz, loc=dz + u1x, scale=s_z)
        px = max(px, Px); py = max(py, Py); pz = max(pz, Pz)
        pt = max(pt, Px * Py * Pz)
    return px, py, pz, pt


# ---------------- 默认参数（与原始向南直飞脚本一致） ----------------
DEFAULT_PARAMS = dict(
    V0=84.0, a1=0.085, a2=1.2, SITA1=3.0, SITA2=0.083,
    ad1=1.4, Vlof=80.0, ad2=0.8, V2_10=90.0, ad3=1.2,
    tansitaD2=0.08, tansitaD3=0.10,
    u1x=50.0, sigma1=50.0, sigma2=50.0, sigma3=60.0,
)

# 方向差异：离场 x 偏移、反应时间
DIRECTION_CONF = {
    '向南': dict(x_off=1430.0, T1_default=18),
    '向北': dict(x_off=-4630.0, T1_default=30),
}

COLLISION_BOX = {
    'A': (2800, 2800, 150),
    'B': (1000, 1000, 100),
    'C': (100, 100, 60),
}


def compute_avoidance(direction='向南', lamda='A', T1=18, degrees=(35,),
                      N=0.01, step=2, t_end=200, dep_t_end=180, params=None,
                      l1_min=1500.0, progress_cb=None):
    """遍历改出角度 × 避让位置(L1) × 离场已起飞时长(dept)，返回碰撞风险表。

    时间对齐（与原始 向南/向北遍历直飞 脚本一致）：
        进近机在 arrt 时刻开始避让，此刻离场机已起飞 dept 秒；
        进近机内部时间 τ 与离场机内部时间 dept+τ 逐点比较（索引对齐，joff=dept/step）。
    风险聚合：对每个 (L1,dept) 取 总风险=max_k{ Px·Py·Pz }，再对 dept 取最大 → 每 (ALFA,L1) 一行。

    返回 DataFrame 列：
        ALFA 避让距离L1(m) 最不利起飞时长(s) 纵向Px 侧向Py 垂直Pz 总风险Pt
    N 为架次参数：原始脚本写死 pt=px·py·pz/100，等价于 N=0.01（默认）。
    """
    p = dict(DEFAULT_PARAMS);
    if params: p.update(params)
    lamdax, lamday, lamdaz = COLLISION_BOX[lamda]
    conf = DIRECTION_CONF[direction]
    X0 = FAF_LEN; Z0 = getz(X0)
    s_xy = math.sqrt(p['sigma1'] ** 2 + p['sigma2'] ** 2)
    s_z = math.sqrt(p['sigma3'] ** 2 + p['sigma3'] ** 2)
    u = p['u1x']

    # 离场轨迹只与方向/参数有关，与 dept/arrt/ALFA 无关 → 预计算一次
    # 覆盖内部时间 [0, dept_max + dep_t_end]（原始 dept∈[0,TD3], indept∈[dept,dept+TD3]）
    Ldx0, Ldy0, Ldz0 = departure_straight_traj(
        p['ad1'], p['Vlof'], p['ad2'], p['V2_10'], p['ad3'],
        p['tansitaD2'], p['tansitaD3'], conf['x_off'],
        dep_t_end + dep_t_end, step)
    Ldx0 = np.asarray(Ldx0); Ldy0 = np.asarray(Ldy0); Ldz0 = np.asarray(Ldz0)
    ndep = len(Ldx0)

    rows = []
    arrts = np.arange(0, t_end, step)
    total = len(degrees) * len(arrts)
    idx = 0
    for ALFA in degrees:
        for arrt in arrts:
            idx += 1
            if progress_cb and (idx % 4 == 0 or idx == total):
                progress_cb(idx / total)
            V1 = p['V0'] - p['a1'] * arrt
            if V1 < 0:
                continue
            L1 = FAF_LEN - (p['V0'] * arrt - 0.5 * p['a1'] * arrt * arrt)
            if L1 < l1_min:
                continue
            Lx, Ly, Lz = approach_avoidance_traj(
                p['V0'], p['a1'], p['a2'], p['SITA1'], p['SITA2'],
                T1, ALFA, X0, Z0, arrt, t_end, step)
            if not Lx:
                continue
            Lx = np.asarray(Lx); Ly = np.asarray(Ly); Lz = np.asarray(Lz)
            napp = len(Lx)
            best = (0.0, 0.0, 0.0, 0.0, 0)   # px, py, pz, pt, dept
            for dept in np.arange(0, dep_t_end, step):
                joff = int(round(dept / step))
                n = min(napp, ndep - joff)
                if n <= 0:
                    continue
                # 逐时刻向量化碰撞概率
                dx = np.abs(Lx[:n] - Ldx0[joff:joff + n])
                dy = np.abs(Ly[:n] - Ldy0[joff:joff + n])
                dz = np.abs(Lz[:n] - Ldz0[joff:joff + n])
                Px = norm.cdf(lamdax, loc=dx + u, scale=s_xy) - norm.cdf(-lamdax, loc=dx + u, scale=s_xy)
                Py = norm.cdf(lamday, loc=dy + u, scale=s_xy) - norm.cdf(-lamday, loc=dy + u, scale=s_xy)
                Pz = norm.cdf(lamdaz, loc=dz + u, scale=s_z) - norm.cdf(-lamdaz, loc=dz + u, scale=s_z)
                pxm = float(Px.max()); pym = float(Py.max()); pzm = float(Pz.max())
                ptm = float((Px * Py * Pz).max()) * N
                if ptm > best[3]:
                    best = (pxm, pym, pzm, ptm, int(dept))
            px, py, pz, pt, dept = best
            rows.append([ALFA, round(L1, 1), dept, px, py, pz, pt])
    df = pd.DataFrame(rows, columns=['ALFA', 'L1_m', 'dept_s',
                                     'Px', 'Py', 'Pz', 'Pt'])
    return df


def adw_boundary(df, tls=TLS):
    """对每个改出角度，求总风险首次低于 TLS 的避让距离（ADW 下边界，m）。"""
    out = []
    for alfa, g in df.groupby('ALFA'):
        g = g.sort_values('L1_m')
        safe = g[g['Pt'] < tls]
        boundary = round(safe['L1_m'].min(), 0) if len(safe) else np.nan
        out.append({'ALFA': alfa, 'ADW下边界_m': boundary,
                    '最大风险': g['Pt'].max()})
    return pd.DataFrame(out)


def risk_vs_dept(direction='向南', lamda='A', T1=18, ALFA=35.0, L1=5000.0,
                 N=0.01, step=2, t_end=200, dep_t_end=180, params=None,
                 progress_cb=None):
    """固定改出角 ALFA 与避让距离 L1，扫描离场已起飞时长 dept，返回风险随 dept 的变化。

    用于展示「最不利起飞时刻」——峰值对应的 dept 即 compute_avoidance 里每行记录的 dept_s。
    返回 DataFrame：dept_s Px Py Pz Pt
    """
    p = dict(DEFAULT_PARAMS);
    if params: p.update(params)
    lamdax, lamday, lamdaz = COLLISION_BOX[lamda]
    conf = DIRECTION_CONF[direction]
    X0 = FAF_LEN; Z0 = getz(X0)
    s_xy = math.sqrt(p['sigma1'] ** 2 + p['sigma2'] ** 2)
    s_z = math.sqrt(p['sigma3'] ** 2 + p['sigma3'] ** 2)
    u = p['u1x']

    # 找最接近目标 L1 的 arrt（L1 由 arrt 唯一决定，与 ALFA 无关）
    best_arrt, best_dl = None, np.inf
    for arrt in np.arange(0, t_end, step):
        V1 = p['V0'] - p['a1'] * arrt
        if V1 < 0:
            continue
        l = FAF_LEN - (p['V0'] * arrt - 0.5 * p['a1'] * arrt * arrt)
        if abs(l - L1) < best_dl:
            best_dl, best_arrt = abs(l - L1), arrt
    if best_arrt is None:
        return pd.DataFrame(columns=['dept_s', 'Px', 'Py', 'Pz', 'Pt'])

    Lx, Ly, Lz = approach_avoidance_traj(
        p['V0'], p['a1'], p['a2'], p['SITA1'], p['SITA2'],
        T1, ALFA, X0, Z0, best_arrt, t_end, step)
    if not Lx:
        return pd.DataFrame(columns=['dept_s', 'Px', 'Py', 'Pz', 'Pt'])
    Lx = np.asarray(Lx); Ly = np.asarray(Ly); Lz = np.asarray(Lz)
    napp = len(Lx)

    Ldx0, Ldy0, Ldz0 = departure_straight_traj(
        p['ad1'], p['Vlof'], p['ad2'], p['V2_10'], p['ad3'],
        p['tansitaD2'], p['tansitaD3'], conf['x_off'],
        dep_t_end + dep_t_end, step)
    Ldx0 = np.asarray(Ldx0); Ldy0 = np.asarray(Ldy0); Ldz0 = np.asarray(Ldz0)
    ndep = len(Ldx0)

    rows = []
    depts = np.arange(0, dep_t_end, step)
    total = len(depts)
    for k, dept in enumerate(depts):
        if progress_cb:
            progress_cb((k + 1) / total)
        joff = int(round(dept / step))
        n = min(napp, ndep - joff)
        if n <= 0:
            continue
        dx = np.abs(Lx[:n] - Ldx0[joff:joff + n])
        dy = np.abs(Ly[:n] - Ldy0[joff:joff + n])
        dz = np.abs(Lz[:n] - Ldz0[joff:joff + n])
        Px = norm.cdf(lamdax, loc=dx + u, scale=s_xy) - norm.cdf(-lamdax, loc=dx + u, scale=s_xy)
        Py = norm.cdf(lamday, loc=dy + u, scale=s_xy) - norm.cdf(-lamday, loc=dy + u, scale=s_xy)
        Pz = norm.cdf(lamdaz, loc=dz + u, scale=s_z) - norm.cdf(-lamdaz, loc=dz + u, scale=s_z)
        rows.append([int(dept), float(Px.max()), float(Py.max()),
                     float(Pz.max()), float((Px * Py * Pz).max()) * N])
    return pd.DataFrame(rows, columns=['dept_s', 'Px', 'Py', 'Pz', 'Pt'])


# ---------------- 尾流 / 改出角（35° 来源） ----------------
def compute_wake_min_height(degrees=(10, 15, 20, 25, 30, 35, 40, 45),
                            T1=20, step=1, sep=350.0, params=None,
                            progress_cb=None):
    """计算各改出角度下，两机水平接近(≤sep m)且离场已离地时的最小垂直高度差。

    与原始「向南尾流7.23」一致：
        - 离场参数：ad1=1.5, TD1=17(硬编码), tansitaD2=0.13, tansitaD3=0.16, x偏移=0
        - 条件：进近机 |x|<sep 且 |Δy|<sep 且 |Δx|<sep 且 离场 z>437
        - 结果：满足条件的 |z进近 − z离场| 最小值
    返回 DataFrame：ALFA, 最小高度差_m
    """
    ap = dict(DEFAULT_PARAMS);
    if params: ap.update(params)
    X0 = FAF_LEN; Z0 = getz(X0)
    t_end = 300          # 原始 T12
    # 尾流场景离场轨迹（x偏移0，TD1=17 覆盖）
    Ldx0, Ldy0, Ldz0 = departure_straight_traj(
        1.5, ap['Vlof'], 0.8, ap['V2_10'], 1.2, 0.13, 0.16, 0.0,
        180, step, TD1=17)
    Ldx0 = np.asarray(Ldx0); Ldy0 = np.asarray(Ldy0); Ldz0 = np.asarray(Ldz0)
    lifted = Ldz0 > 437.0          # 离场已离地
    ndep = len(Ldx0)
    res = []
    total = len(degrees)
    for k, ALFA in enumerate(degrees):
        if progress_cb:
            progress_cb((k + 1) / total)
        min_dz = np.inf
        for arrt in np.arange(0, t_end, step):
            V1 = ap['V0'] - ap['a1'] * arrt
            if V1 < 0:
                continue
            l = FAF_LEN - (ap['V0'] * arrt - 0.5 * ap['a1'] * arrt * arrt)
            if l < 500:            # 原始 lastLen
                continue
            Lx, Ly, Lz = approach_avoidance_traj(
                ap['V0'], ap['a1'], ap['a2'], ap['SITA1'], ap['SITA2'],
                T1, ALFA, X0, Z0, arrt, t_end, step)
            Lx = np.asarray(Lx); Ly = np.asarray(Ly); Lz = np.asarray(Lz)
            for i in range(len(Lx)):
                if abs(Lx[i]) > sep:
                    continue
                m = lifted & (np.abs(Ly[i] - Ldy0) < sep) & (np.abs(Lx[i] - Ldx0) < sep)
                if m.any():
                    dz = float(np.abs(Lz[i] - Ldz0[m]).min())
                    if dz < min_dz:
                        min_dz = dz
        res.append({'ALFA': ALFA,
                    '最小高度差_m': None if np.isinf(min_dz) else round(min_dz, 1)})
    return pd.DataFrame(res)


# ================= 复飞（Go-Around）场景 =================
# 原始「向北复飞.py」存在多处笔误与符号混乱（a1 正负、dep2 的 +437、Ldz.append(ldy)、
# dep5 的 98°/95°、未调用的 DH 复飞死代码）。此处按《说明书》复飞模型**重构**：
#   进近机：下降 → 复飞直线加速爬升(梯度 PDGMA, 加速度 a2, 至 vM) → 转弯(15°坡度) turn_angle
#   定点转弯：到达固定 x=turn_x 时转；定高转弯：到达固定高度 z=turn_h 时转。
GOAROUND_DEFAULT = dict(
    v0=95.0, a_desc=0.1, a2=1.0, vM=130.6, PDGMA=0.08,
    turn_angle=33.0, turn_x=13330.0, turn_h=700.0,
    sigma1=50.0, sigma2=50.0, sigma3=60.0, u1x=50.0,
)
GOAROUND_X0 = 14154.0 + 3200.0 + 1430.0   # 复飞初点纵向坐标（原脚本 18784）
GOAROUND_Z0 = 1200.0                       # 复飞初点高度


def goaround_approach_traj(v0, a_desc, a2, vM, PDGMA, turn_angle, turn_mode,
                           X0, Z0, turn_x, turn_h, arrt, t_end, step):
    """进近复飞轨迹（重构，4 段）。返回 (Lx, Ly, Lz)。

    ① [0,arrt) 沿 3° 下滑道下降（速度 v0-a_desc·t）；
    ② 复飞直线加速爬升（梯度 PDGMA，加速度 a2，至 vM）；
    ③ 转弯（15° 坡度）turn_angle 度——定点在 x≤turn_x、定高在 z≥turn_h 时开始；
    ④ 转弯后沿新航向直飞爬升。
    """
    s_arrt = v0 * arrt - 0.5 * a_desc * arrt * arrt
    x_arrt = X0 - s_arrt
    z_arrt = Z0 - s_arrt * np.tan(degch(3))
    v_arrt = v0 - a_desc * arrt
    tb = max(0.0, (vM - v_arrt) / a2)      # 加速至 vM 用时

    Lx, Ly, Lz = [], [], []
    turn_t = None
    R = gama = 1.0
    x_turn = z_turn = x_end = y_end = z_end = 0.0
    for t in np.arange(0, t_end, step):
        if t < arrt:
            s = v0 * t - 0.5 * a_desc * t * t
            x = X0 - s
            y = 0.0
            z = Z0 - s * np.tan(degch(3))
        else:
            dt = t - arrt
            if dt < tb:
                v = v_arrt + a2 * dt
                s = v_arrt * dt + 0.5 * a2 * dt * dt
            else:
                v = vM
                s = v_arrt * tb + 0.5 * a2 * tb * tb + vM * (dt - tb)
            x = x_arrt - s
            y = 0.0
            z = z_arrt + s * PDGMA
            if turn_t is None:
                if (turn_mode == '定点' and x <= turn_x) or (turn_mode == '定高' and z >= turn_h):
                    turn_t = t
                    x_turn, z_turn = x, z
                    gama, R = get_R(v, 0.0, 0.0)      # 转弯率/半径（15° 坡度）
                    x_end = x_turn - R * np.sin(degch(turn_angle))
                    y_end = R - R * np.cos(degch(turn_angle))
                    z_end = z_turn + R * (turn_angle * PI / 180) * PDGMA
            if turn_t is not None:
                ang = gama * (t - turn_t)
                if ang < turn_angle:
                    x = x_turn - R * np.sin(degch(ang))
                    y = R - R * np.cos(degch(ang))
                    z = z_turn + R * (ang * PI / 180) * PDGMA
                else:
                    s2 = vM * ((t - turn_t) - turn_angle / gama)
                    x = x_end - s2 * np.cos(degch(turn_angle))
                    y = y_end + s2 * np.sin(degch(turn_angle))
                    z = z_end + s2 * PDGMA
        Lx.append(x); Ly.append(y); Lz.append(z)
    return Lx, Ly, Lz


def departure_turn_traj(step, t_end):
    """RWY11 离场 5 段（含 95° 转弯，TT465 后）。参数取自「向北复飞.py」dep1-5，
    已修正 dep2 的 +437 笔误与 dep5 的 98°→95° 笔误。返回 (Ldx, Ldy, Ldz)。"""
    aD1 = 1.4; Vlof = 80.0; aD2 = 0.8; vD3 = 90.0 + 5.144
    aD3 = 1.2; tansitaD2 = 0.08; tansitaD3 = 0.1
    zTT465 = 1350.0
    TD1 = Vlof / aD1
    TD2 = (vD3 - Vlof) / aD2 + TD1
    TD3 = TD2 + (-vD3 + np.sqrt(vD3 * vD3 + 2 * aD3 * (
        8800 - Vlof * Vlof / (2 * aD1) - (vD3 * vD3 - Vlof * Vlof) / (2 * aD2)))) / aD3
    vTT465 = (470 * 171233 * np.sqrt(288 + 15 - 0.006496 * zTT465)) / (
        3.6 * np.power(288 - 0.006496 * zTT465, 2.628))
    gama = 562 * np.tan(degch(15)) / vTT465
    r = 180 * vTT465 / (PI * gama)
    TD4 = TD3 + 95 / gama
    tmp = (-vD3 + np.sqrt(vD3 * vD3 + 2 * aD3 * (
        8800 - Vlof * Vlof / (2 * aD1) - (vD3 * vD3 - Vlof * Vlof) / (2 * aD2)))) / aD3

    Ldx, Ldy, Ldz = [], [], []
    for t in np.arange(0, t_end, step):
        if t < TD1:
            x = 0.0; y = 340 + 0.5 * aD1 * t * t; z = 437.0
        elif t < TD2:
            x = 0.0
            y = 340 + Vlof * Vlof / (2 * aD1) + (Vlof * (t - TD1) + 0.5 * aD2 * (t - TD1) ** 2)
            z = 437 + (Vlof * (t - TD1) + 0.5 * aD2 * (t - TD1) ** 2) * tansitaD2
        elif t < TD3:
            x = 0.0
            y = 340 + Vlof * Vlof / (2 * aD1) + (vD3 * vD3 - Vlof * Vlof) / (2 * aD2) \
                + (vD3 * (t - TD2) + 0.5 * aD3 * (t - TD2) ** 2)
            z = 437 + (vD3 * vD3 - Vlof * Vlof) / (2 * aD2) * tansitaD2 \
                + (vD3 * (t - TD2) + 0.5 * aD3 * (t - TD2) ** 2) * tansitaD3
        elif t < TD4:
            x = -r * (1 - np.cos(degch(gama * (t - TD3))))
            y = 340 + 8800 + r * np.sin(degch(gama * (t - TD3)))
            z = 437 + (vD3 * vD3 - Vlof * Vlof) / (2 * aD2) * tansitaD2 \
                + (vD3 * tmp + 0.5 * aD3 * tmp * tmp + r * gama * (t - TD3) * PI / 180) * tansitaD3
        else:
            x = -r * (1 - np.cos(degch(95))) \
                - (vTT465 * (t - TD4) + 0.5 * aD3 * (t - TD4) ** 2) * np.cos(degch(5))
            y = 340 + 8800 + r * np.sin(degch(95)) \
                + (vTT465 * (t - TD4) + 0.5 * aD3 * (t - TD4) ** 2) * np.sin(degch(5))
            z = 437 + (vD3 * vD3 - Vlof * Vlof) / (2 * aD2) * tansitaD2 \
                + (vD3 * tmp + 0.5 * aD3 * tmp * tmp) * tansitaD3 \
                + (r * PI * 95 / 180 + (vTT465 * (t - TD4) + 0.5 * aD3 * (t - TD4) ** 2)) * tansitaD3
        Ldx.append(x); Ldy.append(y); Ldz.append(z)
    return Ldx, Ldy, Ldz


def compute_goaround(turn_mode='定点', N=1.0, step=5, t_end=200, dep_t_end=600,
                     params=None, len_dh=830.0, progress_cb=None):
    """复飞碰撞风险遍历。碰撞核 12000×12000×300（复飞无 /100 归一化，N 默认 1）。

    返回 DataFrame：ALFA 避让距离L1(m) 最不利起飞时长(s) Px Py Pz Pt
    （ALFA 列在此场景恒为 NaN/0，仅保留结构一致；实际仅按 L1 遍历）
    """
    p = dict(GOAROUND_DEFAULT);
    if params: p.update(params)
    lamdax, lamday, lamdaz = 12000.0, 12000.0, 300.0
    u = p['u1x']
    s_xy = math.sqrt(p['sigma1'] ** 2 + p['sigma2'] ** 2)
    s_z = math.sqrt(p['sigma3'] ** 2 + p['sigma3'] ** 2)
    X0 = GOAROUND_X0; Z0 = GOAROUND_Z0

    Ldx0, Ldy0, Ldz0 = departure_turn_traj(step, dep_t_end)
    Ldx0 = np.asarray(Ldx0); Ldy0 = np.asarray(Ldy0); Ldz0 = np.asarray(Ldz0)
    ndep = len(Ldx0)

    rows = []
    arrts = np.arange(0, t_end, step)
    total = len(arrts)
    for k, arrt in enumerate(arrts):
        if progress_cb:
            progress_cb((k + 1) / total)
        L1 = FAF_LEN - (p['v0'] * arrt - 0.5 * p['a_desc'] * arrt * arrt)
        if L1 <= len_dh:
            continue
        Lx, Ly, Lz = goaround_approach_traj(
            p['v0'], p['a_desc'], p['a2'], p['vM'], p['PDGMA'],
            p['turn_angle'], turn_mode, X0, Z0, p['turn_x'], p['turn_h'],
            arrt, t_end, step)
        if not Lx:
            continue
        Lx = np.asarray(Lx); Ly = np.asarray(Ly); Lz = np.asarray(Lz)
        napp = len(Lx)
        best = (0.0, 0.0, 0.0, 0.0, 0)
        for dept in np.arange(0, dep_t_end - t_end, step):
            joff = int(round(dept / step))
            n = min(napp, ndep - joff)
            if n <= 0:
                continue
            dx = np.abs(Lx[:n] - Ldx0[joff:joff + n])
            dy = np.abs(Ly[:n] - Ldy0[joff:joff + n])
            dz = np.abs(Lz[:n] - Ldz0[joff:joff + n])
            Px = norm.cdf(lamdax, loc=dx + u, scale=s_xy) - norm.cdf(-lamdax, loc=dx + u, scale=s_xy)
            Py = norm.cdf(lamday, loc=dy + u, scale=s_xy) - norm.cdf(-lamday, loc=dy + u, scale=s_xy)
            Pz = norm.cdf(lamdaz, loc=dz + u, scale=s_z) - norm.cdf(-lamdaz, loc=dz + u, scale=s_z)
            pxm = float(Px.max()); pym = float(Py.max()); pzm = float(Pz.max())
            ptm = float((Px * Py * Pz).max()) * N
            if ptm > best[3]:
                best = (pxm, pym, pzm, ptm, int(dept))
        px, py, pz, pt, dept = best
        rows.append([float('nan'), round(L1, 1), dept, px, py, pz, pt])
    df = pd.DataFrame(rows, columns=['ALFA', 'L1_m', 'dept_s',
                                     'Px', 'Py', 'Pz', 'Pt'])
    return df


def goaround_min_sep(turn_mode='定点', step=5, t_end=200, dep_t_end=600,
                     params=None, len_dh=830.0, progress_cb=None):
    """复飞机与离场机的最小间距（横向/纵向/垂直/水平合成），用于对比定点 vs 定高。

    遍历 L1 × dept，取全局最小间距。返回 dict：{min_dx, min_dy, min_dz, min_dh}。"""
    p = dict(GOAROUND_DEFAULT);
    if params: p.update(params)
    X0 = GOAROUND_X0; Z0 = GOAROUND_Z0
    Ldx0, Ldy0, Ldz0 = departure_turn_traj(step, dep_t_end)
    Ldx0 = np.asarray(Ldx0); Ldy0 = np.asarray(Ldy0); Ldz0 = np.asarray(Ldz0)
    ndep = len(Ldx0)
    min_dx = min_dy = min_dz = min_dh = np.inf
    arrts = np.arange(0, t_end, step)
    total = len(arrts)
    for k, arrt in enumerate(arrts):
        if progress_cb:
            progress_cb((k + 1) / total)
        L1 = FAF_LEN - (p['v0'] * arrt - 0.5 * p['a_desc'] * arrt * arrt)
        if L1 <= len_dh:
            continue
        Lx, Ly, Lz = goaround_approach_traj(
            p['v0'], p['a_desc'], p['a2'], p['vM'], p['PDGMA'],
            p['turn_angle'], turn_mode, X0, Z0, p['turn_x'], p['turn_h'],
            arrt, t_end, step)
        Lx = np.asarray(Lx); Ly = np.asarray(Ly); Lz = np.asarray(Lz)
        napp = len(Lx)
        for dept in np.arange(0, dep_t_end - t_end, step):
            joff = int(round(dept / step))
            n = min(napp, ndep - joff)
            if n <= 0:
                continue
            dx = np.abs(Lx[:n] - Ldx0[joff:joff + n])
            dy = np.abs(Ly[:n] - Ldy0[joff:joff + n])
            dz = np.abs(Lz[:n] - Ldz0[joff:joff + n])
            dh = np.sqrt(dx ** 2 + dy ** 2)
            min_dx = min(min_dx, float(dx.min()))
            min_dy = min(min_dy, float(dy.min()))
            min_dz = min(min_dz, float(dz.min()))
            min_dh = min(min_dh, float(dh.min()))
    return {'min_dx_m': min_dx, 'min_dy_m': min_dy,
            'min_dz_m': min_dz, 'min_dh_m': min_dh}


if __name__ == '__main__':
    # 冒烟测试
    df = compute_avoidance(direction='向南', lamda='A', T1=18, degrees=[35], step=5)
    print(df.head())
    print(adw_boundary(df))
    w = compute_wake_min_height(degrees=[15, 30, 35, 40, 45], step=2)
    print(w)
