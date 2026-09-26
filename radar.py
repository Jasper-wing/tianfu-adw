# -*- coding: utf-8 -*-
"""
雷达航迹解析与误差统计 —— 从 tf_track.py / tf_track_sort.py 重构，参数化。

输入：
    - 雷达原始数据 SQL（每条 INSERT 内含 XML <SData><PositionReport>…</PositionReport></SData>）
    - 呼号表 ac_tp.txt（制表符分隔：离场呼号 \t 离场机型 \t 进近呼号 \t 进近机型）
输出：
    - 天府范围内进/离场航迹 DataFrame（11 字段）
    - 进近误差 / 一边误差 / TT465 高度 / 进近速度 / 8 个进近区域统计
"""
import re
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt

# 11 个字段（顺序与原始正则捕获组一致）
RADAR_FIELDS = ['TimeStamp', 'TRACKID', 'CallSign', 'RSPID',
                'GroudSpeed', 'Height', 'Altitude',
                'LONGITUDE', 'LATITUDE', 'Vector', 'VerticalRate']

# 原始 SQL 中 XML 以字面 \n（backslash-n）分隔；部分导出含真实换行/制表符，
# 或额外夹带 <DataSource>/<DataType> 标签，故用统一分隔符 + 可选标签做兼容。
_SEP = r'(?:\\n|\s)*'   # 字面 \n 或任意真实空白，可重复出现

RADAR_PATTERN = re.compile(
    r'<TimeStamp>(.*?)</TimeStamp>' + _SEP +
    r'(?:<DataSource>.*?</DataSource>' + _SEP + r')?' +
    r'(?:<DataType>.*?</DataType>' + _SEP + r')?' +
    r'<PositionReport>' + _SEP +
    r'<TRACKID>(.*?)</TRACKID>' + _SEP +
    r'<CallSign>(.*?)</CallSign>' + _SEP +
    r'<REGID>.*?</REGID>' + _SEP +
    r'<RSPID>(.*?)</RSPID>' + _SEP +
    r'<GroudSpeed>(.*?)</GroudSpeed>' + _SEP +
    r'<Height>(.*?)</Height>' + _SEP +
    r'<Altitude>(.*?)</Altitude>' + _SEP +
    r'<LONGITUDE>(.*?)</LONGITUDE>' + _SEP +
    r'<LATITUDE>(.*?)</LATITUDE>' + _SEP +
    r'<Vector>(.*?)</Vector>' + _SEP +
    r'<VerticalRate>(.*?)</VerticalRate>' + _SEP +
    r'</PositionReport>' + _SEP + r'</SData>'
)


def check_app(long, lat, alt):
    """天府范围筛选：高度 ≤3000 且 经度∈(1020000,1060000) 纬度∈(290000,310000)"""
    if int(alt) <= 3000:
        if 1020000 < int(long) < 1060000 and 290000 < int(lat) < 310000:
            return True
    return False


def load_callsigns(ac_path):
    """读取呼号表，返回 (dep_set, arr_set, dep_types, arr_types)"""
    dep_set, arr_set, dep_types, arr_types = set(), set(), [], []
    with open(ac_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            b = line.split('\t')
            if len(b) < 4:
                continue
            if b[0] != '' and b[0].strip():
                dep_set.add(b[0].strip())
                dep_types.append(b[1].strip())
            if b[2].strip():
                arr_set.add(b[2].strip())
                arr_types.append(b[3].strip())
    return dep_set, arr_set, dep_types, arr_types


def parse_radar_sql(sql_path, ac_path):
    """解析雷达 SQL，返回 (df_arr, df_dep)。"""
    dep_set, arr_set, _, _ = load_callsigns(ac_path)

    def classify(callsign):
        if callsign in dep_set:
            return 'dep'
        if callsign in arr_set:
            return 'arr'
        return None

    arr_rows, dep_rows = [], []
    with open(sql_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            for chunk in line.split('xml')[1:]:      # 去掉 "xml" 前缀，同原始逻辑
                for m in RADAR_PATTERN.findall(chunk):
                    if not check_app(m[7], m[8], m[6]):   # 经度/纬度/高度
                        continue
                    kind = classify(m[2])                  # CallSign
                    if kind == 'arr':
                        arr_rows.append(m)
                    elif kind == 'dep':
                        dep_rows.append(m)
    df_arr = pd.DataFrame(arr_rows, columns=RADAR_FIELDS)
    df_dep = pd.DataFrame(dep_rows, columns=RADAR_FIELDS)
    for c in ['GroudSpeed', 'Height', 'Altitude', 'LONGITUDE',
              'LATITUDE', 'VerticalRate']:
        if c in df_arr:
            df_arr[c] = pd.to_numeric(df_arr[c], errors='coerce')
        if c in df_dep:
            df_dep[c] = pd.to_numeric(df_dep[c], errors='coerce')
    return df_arr, df_dep


# ---------------- 几何工具 ----------------
def is_in_poly(p, poly):
    """射线法判断点是否在多边形内"""
    px, py = p
    is_in = False
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        x1, y1 = poly[i]
        x2, y2 = poly[j]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):
            return True
        if min(y1, y2) < py <= max(y1, y2):
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:
                return True
            elif x > px:
                is_in = not is_in
    return is_in


def haversine_km(lng1, lat1, lng2, lat2):
    """两点间大圆距离 (km)"""
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon, dlat = lng2 - lng1, lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return round(2 * asin(sqrt(a)) * 6371, 3)


def point_to_segment(PAx, PAy, PBx, PBy, PCx, PCy):
    """点 (PC) 到线段 (PA-PB) 的最小距离 (km)"""
    a = haversine_km(PAy, PAx, PBy, PBx)
    b = haversine_km(PBy, PBx, PCy, PCx)
    c = haversine_km(PAy, PAx, PCy, PCx)
    if b * b >= c * c + a * a:
        return c
    if c * c >= b * b + a * a:
        return b
    l = (a + b + c) / 2
    s = sqrt(abs(l * (l - a) * (l - b) * (l - c)))
    return 2 * s / a


def jwd_ch(deg):
    """DMS 压缩格式 → 十进制度（7 位：DDDMMSS；6 位：DDMMSS）"""
    deg = str(deg).strip()
    if len(deg) == 7:
        return float(deg[0:3]) + float(deg[3:5]) / 60 + float(deg[5:7]) / 3600
    if len(deg) == 6:
        return float(deg[0:2]) + float(deg[2:4]) / 60 + float(deg[4:6]) / 3600
    return float(deg)


# ---------------- 标称航迹 / 区域多边形（沿用原始取值） ----------------
FINAL_LOC = [104.3361990, 30.059712, 104.438518, 30.277731]      # RWY02 进近标称航迹
UPW_LOC = [104.4964917, 30.30216389, 104.5445583, 30.28508611]    # RWY11 一边标称航迹
FAF_POINT = [104.38286805555555, 30.159424999999995]

FINAL_RANGE = [[104260801, 30164318], [104262879, 30163582],
               [104202016, 30033091], [104195941, 30033826]]
UPW_RANGE = [[104294313, 30175878], [104295162, 30181681],
             [104324482, 30171541], [104323632, 30165738]]
TT465_RANGE = [[104323814, 30170548], [104323955, 30170849],
               [104324160, 30170425], [104324302, 30170726]]

APP_RANGES = {
    'app1': [[104202016, 30033091], [104195941, 30033826], [104204151, 30050810], [104210226, 30050075]],
    'app2': [[104212364, 30063794], [104214439, 30063059], [104210226, 30050075], [104204151, 30050810]],
    'app3': [[104220578, 30080778], [104222654, 30080042], [104212364, 30063794], [104214439, 30063059]],
    'app4': [[104224794, 30093761], [104230871, 30093025], [104222654, 30080042], [104220578, 30080778]],
    'app5': [[104233792, 30112401], [104235868, 30111665], [104230871, 30093025], [104224794, 30093761]],
    'app6': [[104242792, 30131041], [104244869, 30130304], [104235868, 30111665], [104233792, 30112401]],
    'app7': [[104251795, 30145680], [104253873, 30144943], [104244869, 30130304], [104242792, 30131041]],
    'app8': [[104260801, 30164318], [104262879, 30163582], [104253873, 30144943], [104251795, 30145680]],
}


def _stat(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return {'mean': None, 'max': None, 'std': None, 'n': 0}
    return {'mean': float(np.mean(vals)), 'max': float(np.max(vals)),
            'std': float(np.std(vals)), 'n': len(vals)}


def approach_error(df_arr):
    """进近误差：RWY02 五边范围内航班到标称航迹的最小距离 (km)"""
    wc = []
    for _, r in df_arr.iterrows():
        if int(r['Altitude']) <= 1500:
            p = [int(r['LONGITUDE']) * 100, int(r['LATITUDE']) * 100]
            if is_in_poly(p, FINAL_RANGE):
                d = point_to_segment(jwd_ch(r['LONGITUDE']), jwd_ch(r['LATITUDE']),
                                     FINAL_LOC[0], FINAL_LOC[1],
                                     FINAL_LOC[2], FINAL_LOC[3])
                wc.append(d)
    return _stat(wc)


def departure_error(df_dep):
    """一边误差：RWY11 一边范围内航班到标称航迹的最小距离 (km)"""
    wc = []
    for _, r in df_dep.iterrows():
        if int(r['Altitude']) <= 1500:
            p = [int(r['LONGITUDE']) * 100, int(r['LATITUDE']) * 100]
            if is_in_poly(p, UPW_RANGE):
                d = point_to_segment(jwd_ch(r['LONGITUDE']), jwd_ch(r['LATITUDE']),
                                     UPW_LOC[0], UPW_LOC[1],
                                     UPW_LOC[2], UPW_LOC[3])
                wc.append(d)
    return _stat(wc)


def tt465_height(df_dep):
    """RWY11 一边 TT465 点的高度统计 (m)"""
    hs = []
    for _, r in df_dep.iterrows():
        p = [int(r['LONGITUDE']) * 100, int(r['LATITUDE']) * 100]
        if is_in_poly(p, TT465_RANGE):
            hs.append(int(r['Height']))
    return _stat(hs)


def approach_speed(df_arr):
    """进近速度 vs 距 FAF 点距离：返回 [(距FAF km, 速度 m/s), ...]"""
    res = []
    for _, r in df_arr.iterrows():
        if int(r['Altitude']) <= 1300:
            p = [int(r['LONGITUDE']) * 100, int(r['LATITUDE']) * 100]
            if is_in_poly(p, FINAL_RANGE):
                lng1, lat1 = jwd_ch(r['LONGITUDE']), jwd_ch(r['LATITUDE'])
                d = haversine_km(lng1, lat1, FAF_POINT[0], FAF_POINT[1])
                res.append([round(d), int(r['GroudSpeed']) / 3.6])
    return res


def app_region_stats(df_arr):
    """8 个进近区域的高度/速度/垂直速度统计"""
    rows = []
    for name, poly in APP_RANGES.items():
        hs, vs, vv = [], [], []
        for _, r in df_arr.iterrows():
            if int(r['Altitude']) <= 1200:
                p = [int(r['LONGITUDE']) * 100, int(r['LATITUDE']) * 100]
                if is_in_poly(p, poly):
                    hs.append(int(r['Height']))
                    vs.append(int(r['GroudSpeed']))
                    vv.append(int(r['VerticalRate']))
        h = _stat(hs); v = _stat(vs); vrt = _stat(vv)
        rows.append({'区域': name,
                     '点数': h['n'],
                     '高度均值_m': h['mean'], '高度max_m': h['max'],
                     '速度均值_kmh': v['mean'], '速度max_kmh': v['max'],
                     '垂直速度均值': vrt['mean']})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import sys
    sql, ac = sys.argv[1], sys.argv[2]
    da, dd = parse_radar_sql(sql, ac)
    print('进近航班:', len(da), ' 离场航班:', len(dd))
    print('进近误差:', approach_error(da))
    print('一边误差:', departure_error(dd))
    print('TT465高度:', tt465_height(dd))
