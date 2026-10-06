# -*- coding: utf-8 -*-
"""
成都天府国际机场「平行＋开口V型」混合跑道独立进近
—— 碰撞风险 / 避让 / 尾流 / 雷达数据 全流程 UI（本地网页）

运行：  streamlit run app.py
"""
import io
import time
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import engine
from engine import (compute_avoidance, adw_boundary, compute_wake_min_height,
                    risk_vs_dept,
                    approach_avoidance_traj, departure_straight_traj,
                    compute_goaround, goaround_min_sep,
                    goaround_approach_traj, departure_turn_traj,
                    GOAROUND_DEFAULT, GOAROUND_X0, GOAROUND_Z0,
                    FAF_LEN, getz, DEFAULT_PARAMS, DIRECTION_CONF, TLS)

def _df_to_excel_bytes(df, sheet='结果'):
    """DataFrame → Excel 字节（openpyxl），供下载按钮使用。"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        df.to_excel(w, index=False, sheet_name=sheet)
    return buf.getvalue()


def _memo(key, fn):
    """按参数键缓存到 session_state；未命中则带进度条计算并写入缓存。"""
    if key not in st.session_state:
        bar = st.progress(0.0, text='计算中…')
        st.session_state[key] = fn(lambda f: bar.progress(min(float(f), 1.0), text='计算中…'))
        bar.empty()
    return st.session_state[key]


def _traj3d(x, y, z, dx, dy, dz, name1='进近航空器', name2='离场航空器',
            color1='#1a5fa8', color2='#e07b39', title=''):
    """Plotly 交互式三维航迹（可旋转 / 缩放 / 悬停）。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name=name1,
                               line=dict(color=color1, width=5)))
    fig.add_trace(go.Scatter3d(x=dx, y=dy, z=dz, mode='lines', name=name2,
                               line=dict(color=color2, width=5)))
    fig.update_layout(
        title=title, height=560,
        scene=dict(xaxis_title='纵向 x (m)', yaxis_title='侧向 y (m)',
                   zaxis_title='高度 z (m)', aspectmode='data'),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.7)'),
        margin=dict(l=0, r=0, t=44, b=0),
    )
    return fig


def _arrt_for_l1(L1, p):
    """求最接近目标避让距离 L1 (m) 的进近机避让起始时刻 arrt (s)。"""
    best_arrt, best_dl = None, np.inf
    for arrt in np.arange(0, 200, 1):
        V1 = p['V0'] - p['a1'] * arrt
        if V1 < 0:
            continue
        l = FAF_LEN - (p['V0'] * arrt - 0.5 * p['a1'] * arrt * arrt)
        if abs(l - L1) < best_dl:
            best_dl, best_arrt = abs(l - L1), arrt
    return best_arrt


def _adw_plan_view(direction, params, T1, alfa, l1_upper, l1_lower):
    """ADW 俯视示意图：进近/离场航迹 + ADW 窗 + 不同避让距离的改出轨迹（x-y 平面）。"""
    p = dict(DEFAULT_PARAMS); p.update(params)
    x_off = DIRECTION_CONF[direction]['x_off']
    X0, Z0 = FAF_LEN, getz(FAF_LEN)

    fig = go.Figure()

    # 进近标称航迹（跑道中线延长，y=0）
    fig.add_trace(go.Scatter(
        x=[0, FAF_LEN], y=[0, 0], mode='lines',
        line=dict(color='#9aa7b8', width=2, dash='dot'),
        name='RWY02/20 进近标称航迹', hoverinfo='skip'))

    # 离场航迹（x=x_off 竖直向上）
    Ldx, Ldy, _ = departure_straight_traj(
        p['ad1'], p['Vlof'], p['ad2'], p['V2_10'], p['ad3'],
        p['tansitaD2'], p['tansitaD3'], x_off, 180, 2)
    fig.add_trace(go.Scatter(
        x=Ldx, y=Ldy, mode='lines', line=dict(color='#e07b39', width=3),
        name='RWY11 离场航迹', hoverinfo='skip'))

    # 三条改出轨迹：窗内（危险）/ 下边界 / 窗外（安全）
    for name, L1, color in [('窗内 · 危险', l1_upper, '#d62728'),
                            ('ADW 下边界', l1_lower, '#f0a03c'),
                            ('窗外 · 安全', l1_lower + 2500, '#2ca02c')]:
        arrt = _arrt_for_l1(L1, p)
        if arrt is None:
            continue
        Lx, Ly, _ = approach_avoidance_traj(
            p['V0'], p['a1'], p['a2'], p['SITA1'], p['SITA2'],
            T1, alfa, X0, Z0, arrt, arrt + 250, 1)
        fig.add_trace(go.Scatter(
            x=Lx, y=Ly, mode='lines', line=dict(color=color, width=3),
            name=f'改出 L1≈{L1:.0f} m（{name}）', hoverinfo='skip'))

    # ADW 窗（半透明带，沿进近航迹）
    fig.add_shape(type='rect', x0=l1_upper, x1=l1_lower, y0=-500, y1=500,
                  fillcolor='rgba(214,39,40,0.14)',
                  line=dict(color='#d62728', width=1.5, dash='dot'), layer='below')

    # 跑道入口竖线 + 进近方向箭头
    fig.add_shape(type='line', x0=0, x1=0, y0=-500, y1=500,
                  line=dict(color='#333', width=2.5), layer='below')

    fig.update_layout(
        title=f'{direction}运行 · ADW 俯视示意图（改出角 {alfa:.0f}°）',
        xaxis_title='纵向 x (m) —— 0 为跑道入口，正值朝向 FAF',
        yaxis_title='侧向 y (m)',
        annotations=[
            dict(x=0, y=-720, text='跑道入口<br>(x=0)', showarrow=False,
                 font=dict(size=11, color='#333')),
            dict(x=FAF_LEN, y=-720, text='FAF<br>(14155 m)', showarrow=False,
                 font=dict(size=11, color='#1a5fa8')),
            dict(x=0, y=-250, ax=2800, ay=-250, showarrow=True, arrowhead=3,
                 arrowsize=1, arrowwidth=2, arrowcolor='#1a5fa8',
                 text='进近方向', font=dict(size=11, color='#1a5fa8')),
            dict(x=l1_upper, y=820, text=f'ADW 上边界<br>{l1_upper:.0f} m',
                 showarrow=True, arrowhead=2, ax=0, ay=-40,
                 font=dict(size=11, color='#d62728')),
            dict(x=l1_lower, y=820, text=f'ADW 下边界<br>{l1_lower:.0f} m',
                 showarrow=True, arrowhead=2, ax=0, ay=-40,
                 font=dict(size=11, color='#d62728')),
            dict(x=x_off, y=5600, text='RWY11 离场', showarrow=False,
                 font=dict(size=11, color='#e07b39')),
        ],
        height=560,
        xaxis=dict(range=[min(x_off, 0) - 800, FAF_LEN + 400]),
        yaxis=dict(range=[-1100, 6000]),
        margin=dict(l=0, r=0, t=48, b=0),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.7)',
                    font=dict(size=10)),
        hovermode='closest',
    )
    return fig


def _adw_stop_strip(l1_upper, l1_lower, arr_pos):
    """纵向 ADW 停航窗：进场机在 ADW 窗内 → 离场 STOP；窗外 → GO。"""
    inside = l1_upper <= arr_pos <= l1_lower
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, FAF_LEN], y=[0, 0], mode='lines',
        line=dict(color='#9aa7b8', width=4), hoverinfo='skip', showlegend=False))
    fig.add_shape(type='rect', x0=l1_upper, x1=l1_lower, y0=-0.35, y1=0.35,
                  fillcolor='rgba(214,39,40,0.20)',
                  line=dict(color='#d62728', width=1.5, dash='dot'))
    fig.add_shape(type='line', x0=0, x1=0, y0=-0.5, y1=0.5,
                  line=dict(color='#333', width=3))
    fig.add_trace(go.Scatter(
        x=[arr_pos], y=[0], mode='markers',
        marker=dict(symbol='diamond', size=14,
                    color='#d62728' if inside else '#2ca02c'),
        hoverinfo='skip', showlegend=False))
    fig.update_layout(
        title='ADW 停航窗（沿进近航迹）',
        xaxis=dict(range=[min(0, l1_upper) - 500, FAF_LEN + 200],
                   title='距跑道入口纵向距离 (m)'),
        yaxis=dict(range=[-1, 1], visible=False),
        annotations=[
            dict(x=0, y=-0.75, text='跑道入口', showarrow=False,
                 font=dict(size=11, color='#333')),
            dict(x=l1_upper, y=0.6, text=f'上边界 {l1_upper:.0f} m',
                 showarrow=True, arrowhead=2, ax=0, ay=-22,
                 font=dict(size=11, color='#d62728')),
            dict(x=l1_lower, y=0.6, text=f'下边界 {l1_lower:.0f} m',
                 showarrow=True, arrowhead=2, ax=0, ay=-22,
                 font=dict(size=11, color='#d62728')),
        ],
        height=240, margin=dict(l=0, r=0, t=44, b=0), showlegend=False)
    return fig


def _bundle_zip():
    """把当前已算出的全部结果打包为 zip（内含多个 Excel），供一键下载。"""
    key = 'zip_all'
    if key in st.session_state:
        return st.session_state[key]
    parts = []
    if 'risk_df' in st.session_state:
        parts.append(('碰撞风险.xlsx', _df_to_excel_bytes(st.session_state['risk_df'], '碰撞风险')))
    if 'risk_bd' in st.session_state:
        parts.append(('ADW下边界.xlsx', _df_to_excel_bytes(st.session_state['risk_bd'], 'ADW下边界')))
    if 'wake_df' in st.session_state:
        parts.append(('尾流最小高度差.xlsx', _df_to_excel_bytes(st.session_state['wake_df'], '尾流')))
    if 'ga_res' in st.session_state:
        res = st.session_state['ga_res']
        risk_all = pd.concat([r['risk'].assign(转弯方式=m) for m, r in res.items()],
                             ignore_index=True)
        parts.append(('复飞风险.xlsx', _df_to_excel_bytes(risk_all, '复飞风险')))
        rows = [{'转弯方式': m,
                 '最小纵向间距(m)': r['sep']['min_dx_m'],
                 '最小侧向间距(m)': r['sep']['min_dy_m'],
                 '最小垂直间距(m)': r['sep']['min_dz_m'],
                 '最小水平间距(m)': r['sep']['min_dh_m']} for m, r in res.items()]
        parts.append(('复飞最小间距.xlsx', _df_to_excel_bytes(pd.DataFrame(rows), '复飞最小间距')))
    if 'radar_arr' in st.session_state:
        parts.append(('雷达_进近航迹.xlsx',
                      _df_to_excel_bytes(st.session_state['radar_arr'].head(20000), '进近')))
        parts.append(('雷达_离场航迹.xlsx',
                      _df_to_excel_bytes(st.session_state['radar_dep'].head(20000), '离场')))
    if not parts:
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in parts:
            z.writestr(name, data)
    st.session_state[key] = buf.getvalue()
    return st.session_state[key]


st.set_page_config(page_title='天府机场 ADW 冲突避让动态调度 · 安全评估工具', page_icon='✈️',
                   layout='wide', initial_sidebar_state='expanded')

# 全局样式（航空蓝主题）
st.markdown("""
<style>
    /* 隐藏默认页脚与顶部装饰 */
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}

    /* 整体背景与字体 */
    .stApp {
        background: linear-gradient(180deg, #f4f7fb 0%, #edf2f8 100%);
    }
    html, body, [class*="css"] {font-family: "Segoe UI", "Microsoft YaHei", sans-serif;}
    .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1280px;}

    /* Hero 头图 */
    .hero {
        background: linear-gradient(120deg, #0d3a67 0%, #185c9e 52%, #2d7fc4 100%);
        border-radius: 14px;
        padding: 26px 32px 24px;
        color: #fff;
        margin-bottom: 6px;
        box-shadow: 0 8px 24px rgba(13,58,103,.28);
    }
    .hero h1 {color: #fff; font-size: 1.65rem; font-weight: 700; margin: 0; letter-spacing: .5px;}
    .hero p  {color: #cfe3f7; margin: 8px 0 0; font-size: .98rem;}
    .hero .chips {margin-top: 14px;}
    .hero .chip {
        display: inline-block; background: rgba(255,255,255,.14);
        border: 1px solid rgba(255,255,255,.22);
        color: #eef6ff; border-radius: 20px; padding: 3px 12px;
        font-size: .82rem; margin-right: 8px;
    }

    /* 侧边栏 */
    section[data-testid="stSidebar"] {
        background: #eaf1f9;
        border-right: 1px solid #d8e4f1;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {color: #0d3a67;}

    /* 页签 */
    .stTabs [data-baseweb="tab-list"] {gap: 6px; border-bottom: 2px solid #d8e4f1;}
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px 9px 0 0; padding: 9px 20px; font-weight: 600;
        background: #e7eef7; color: #3a4a5c;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #185c9e !important; color: #fff !important;
    }

    /* 按钮 */
    .stButton > button {
        border-radius: 9px; font-weight: 600; border: none;
        background: linear-gradient(120deg, #185c9e, #2d7fc4); color: #fff;
        padding: 8px 20px; box-shadow: 0 3px 10px rgba(24,92,158,.28);
    }
    .stButton > button:hover {background: linear-gradient(120deg, #0d3a67, #185c9e);}
    .stDownloadButton > button {border-radius: 9px; font-weight: 600;}

    /* 指标卡片 */
    [data-testid="stMetric"] {
        background: #fff; border-radius: 11px; padding: 14px 18px;
        border: 1px solid #e2eaf3; box-shadow: 0 2px 8px rgba(13,58,103,.06);
    }
    [data-testid="stMetricValue"] {color: #185c9e; font-weight: 700;}

    /* 数据表 */
    [data-testid="stDataFrame"] {border-radius: 10px; overflow: hidden; border: 1px solid #e2eaf3;}
    [data-testid="stExpander"] {border-radius: 10px;}

    /* 段落标题 */
    .sec {color: #0d3a67; font-weight: 700; margin-top: .4rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>成都天府国际机场 混合跑道进离场冲突避让 · ADW 动态调度 — 安全评估工具</h1>
    <p>基于到达离场窗（ADW）的混合跑道进离场冲突避让动态调度研究　·　碰撞风险 P=Px·Py·Pz　·　紧急避让　·　复飞（定点/定高）　·　尾流　·　雷达航迹 —— 全流程演示平台</p>
    <div class="chips">
        <span class="chip">TLS = 5×10⁻⁹ 次/飞行小时</span>
        <span class="chip">ICAO 安全目标水平</span>
        <span class="chip">平行 + 开口V型 混合跑道</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ================= 侧边栏参数 =================
st.sidebar.header('模型参数')
direction = st.sidebar.selectbox('运行方向', ['向南', '向北'])
lamda_key = st.sidebar.selectbox('碰撞核', ['A', 'B', 'C'], index=0,
                                 format_func=lambda k: {'A': 'A 2800×2800×150',
                                                        'B': 'B 1000×1000×100',
                                                        'C': 'C 100×100×60'}[k])
T1 = st.sidebar.slider('管制员反应时间 T1 (s)', 10, 40,
                       DIRECTION_CONF[direction]['T1_default'])
N = st.sidebar.number_input('架次参数 N（原始 /100 → 0.01）', 0.0001, 100.0, 0.01,
                            step=0.01, format='%.4f')

deg_min, deg_max = st.sidebar.slider('避让改出角范围 (°)', 10, 45, (15, 45))
deg_step = st.sidebar.slider('改出角步长 (°)', 1, 10, 5)
degrees = list(range(deg_min, deg_max + 1, deg_step))

step = st.sidebar.slider('时间步长 (s)', 1, 5, 2)
st.sidebar.markdown('---')
st.sidebar.caption('碰撞风险安全目标水平 TLS = 5×10⁻⁹ 次/飞行小时（ICAO）')

# 高级参数（可折叠）
with st.sidebar.expander('运动学 / 误差参数'):
    V0 = st.number_input('进近初始速度 V0 (m/s)', 60.0, 100.0, DEFAULT_PARAMS['V0'])
    a1 = st.number_input('进近减速 a1 (m/s²)', 0.0, 0.5, DEFAULT_PARAMS['a1'])
    sigma1 = st.number_input('纵向误差 σ1 (m)', 10.0, 200.0, DEFAULT_PARAMS['sigma1'])
    sigma2 = st.number_input('侧向误差 σ2 (m)', 10.0, 200.0, DEFAULT_PARAMS['sigma2'])
    sigma3 = st.number_input('高度误差 σ3 (m)', 10.0, 200.0, DEFAULT_PARAMS['sigma3'])
    params = dict(DEFAULT_PARAMS, V0=V0, a1=a1, sigma1=sigma1, sigma2=sigma2, sigma3=sigma3)


# ================= 页签 =================
tab_risk, tab_wake, tab_traj, tab_radar, tab_ga, tab_adw = st.tabs(
    ['🛡️ ① 碰撞风险与 ADW', '🌪️ ② 尾流 / 改出角', '🛫 ③ 三维航迹',
     '📡 ④ 雷达数据导入', '↩️ ⑤ 复飞（定点 vs 定高）', '🧭 ⑥ ADW 直观示意图'])

# ---------- ① 碰撞风险 ----------
with tab_risk:
    st.subheader('紧急避让碰撞风险 → ADW 下边界')
    st.caption('遍历避让改出角 × 避让距离 L1 × 起飞时刻，反推满足 TLS 的 ADW 下边界。')
    if st.button('计算碰撞风险', type='primary'):
        t0 = time.time()
        pkey = tuple(sorted(params.items()))
        key = ('avoid', direction, lamda_key, T1, tuple(degrees), N, step, pkey)
        df = _memo(key, lambda cb: compute_avoidance(
            direction=direction, lamda=lamda_key, T1=T1, degrees=degrees,
            N=N, step=step, params=params, progress_cb=cb))
        bd = adw_boundary(df)
        st.session_state['risk_df'] = df
        st.session_state['risk_bd'] = bd
        st.session_state['risk_elapsed'] = time.time() - t0
        st.session_state.pop('zip_all', None)
    if 'risk_df' in st.session_state:
        df = st.session_state['risk_df']
        bd = st.session_state['risk_bd']
        el = st.session_state.get('risk_elapsed')
        m1, m2, m3 = st.columns([1, 1, 2])
        m1.metric('计算耗时', f'{el:.1f} s' if el else '—')
        m2.metric('角度数', len(df['ALFA'].unique()))
        m3.metric('风险行数', len(df))
        dl1, dl2 = st.columns(2)
        dl1.download_button('⬇ 下载风险表 Excel', _df_to_excel_bytes(df, '碰撞风险'),
                            file_name='碰撞风险.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        dl2.download_button('⬇ 下载 ADW 下边界 Excel', _df_to_excel_bytes(bd, 'ADW下边界'),
                            file_name='ADW下边界.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('**ADW 下边界（m）**')
            st.dataframe(bd.style.format({'ADW下边界_m': '{:.0f}', '最大风险': '{:.3e}'}),
                         width='stretch')
        with c2:
            st.markdown('**风险 vs 避让距离 L1（可交互）**')
            all_alfa = sorted(df['ALFA'].dropna().unique().tolist())
            sel_alfa = st.multiselect('显示角度', all_alfa, default=all_alfa,
                                      key='risk_alfa_sel')
            fig = go.Figure()
            for alfa in sel_alfa:
                g = df[df['ALFA'] == alfa].sort_values('L1_m')
                fig.add_trace(go.Scatter(
                    x=g['L1_m'], y=g['Pt'], mode='lines+markers',
                    name=f'{alfa:.0f}°'))
            fig.add_hline(y=TLS, line_dash='dash', line_color='#d62728',
                          annotation_text='TLS=5e-9', annotation_position='bottom right')
            fig.update_yaxes(type='log', title='碰撞风险 P')
            fig.update_xaxes(title='避让距离 L1 (m)')
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=0),
                              legend=dict(font=dict(size=10)), hovermode='x unified')
            st.plotly_chart(fig, width='stretch')
        with st.expander('原始风险表（前 200 行）'):
            st.dataframe(df.head(200).style.format({'Px': '{:.3e}', 'Py': '{:.3e}',
                                                    'Pz': '{:.3e}', 'Pt': '{:.3e}'}),
                         width='stretch')

        # ---- 附加分析曲线 ----
        st.markdown('---')
        st.markdown('**📈 附加分析曲线**')
        all_alfa = sorted(df['ALFA'].dropna().unique().tolist())
        ac1, ac2 = st.columns(2)
        with ac1:
            st.markdown('**ADW 下边界 vs 改出角**')
            bd_s = bd.sort_values('ALFA')
            fig = go.Figure(go.Scatter(
                x=bd_s['ALFA'], y=bd_s['ADW下边界_m'], mode='lines+markers',
                line=dict(color='#185c9e', width=3), marker=dict(size=9)))
            fig.update_xaxes(title='改出角 (°)')
            fig.update_yaxes(title='ADW 下边界 (m)')
            fig.update_layout(height=360, margin=dict(l=0, r=0, t=20, b=0),
                              showlegend=False)
            st.plotly_chart(fig, width='stretch')
        with ac2:
            st.markdown('**风险 vs 改出角（选定避让距离）**')
            l1_list = sorted(df['L1_m'].unique().tolist())
            l1_sel = st.selectbox('避让距离 L1', l1_list,
                                  index=min(len(l1_list) // 2, len(l1_list) - 1),
                                  format_func=lambda v: f'{v:.0f} m', key='ana_l1')
            sub = df[df['L1_m'] == l1_sel].sort_values('ALFA')
            fig = go.Figure(go.Scatter(
                x=sub['ALFA'], y=sub['Pt'], mode='lines+markers',
                line=dict(color='#e07b39', width=3), marker=dict(size=9)))
            fig.add_hline(y=TLS, line_dash='dash', line_color='#d62728',
                          annotation_text='TLS', annotation_position='bottom right')
            fig.update_yaxes(type='log', title='碰撞风险 Pt')
            fig.update_xaxes(title='改出角 (°)')
            fig.update_layout(height=360, margin=dict(l=0, r=0, t=20, b=0),
                              showlegend=False)
            st.plotly_chart(fig, width='stretch')

        st.markdown('**风险分量分解 Px / Py / Pz vs 避让距离 L1（看哪个分量主导）**')
        comp_alfa = st.selectbox('分解角度', all_alfa,
                                 format_func=lambda v: f'{v:.0f}°', key='ana_alfa')
        g = df[df['ALFA'] == comp_alfa].sort_values('L1_m')
        fig = go.Figure()
        for col, nm, clr in [('Px', '纵向 Px', '#1a5fa8'), ('Py', '侧向 Py', '#2ca02c'),
                             ('Pz', '垂直 Pz', '#e07b39'), ('Pt', '总风险 Pt', '#d62728')]:
            fig.add_trace(go.Scatter(x=g['L1_m'], y=g[col], mode='lines+markers',
                                     name=nm, line=dict(color=clr, width=2.5)))
        fig.add_hline(y=TLS, line_dash='dash', line_color='#555',
                      annotation_text='TLS', annotation_position='bottom right')
        fig.update_yaxes(type='log', title='概率')
        fig.update_xaxes(title='避让距离 L1 (m)')
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=20, b=0),
                          legend=dict(font=dict(size=11)), hovermode='x unified')
        st.plotly_chart(fig, width='stretch')

        st.markdown('**风险 vs 起飞时刻 dept（最不利起飞时刻）**')
        l1_list = sorted(df['L1_m'].unique().tolist())
        d1, d2 = st.columns(2)
        dept_alfa = d1.selectbox('改出角', all_alfa, format_func=lambda v: f'{v:.0f}°',
                                 key='dept_alfa')
        dept_l1 = d2.selectbox('避让距离', l1_list,
                               index=min(len(l1_list) // 2, len(l1_list) - 1),
                               format_func=lambda v: f'{v:.0f} m', key='dept_l1')
        dkey = ('dept', direction, lamda_key, T1, dept_alfa, dept_l1, N, step,
                tuple(sorted(params.items())))
        ddf = _memo(dkey, lambda cb: risk_vs_dept(
            direction=direction, lamda=lamda_key, T1=T1, ALFA=dept_alfa, L1=dept_l1,
            N=N, step=step, params=params, progress_cb=cb))
        fig = go.Figure()
        for col, nm, clr in [('Pt', '总风险 Pt', '#d62728'), ('Px', '纵向 Px', '#1a5fa8'),
                             ('Py', '侧向 Py', '#2ca02c'), ('Pz', '垂直 Pz', '#e07b39')]:
            yy = np.clip(ddf[col].to_numpy(dtype=float), 1e-15, None)  # 远端 dept 风险→0，截断保对数轴
            fig.add_trace(go.Scatter(x=ddf['dept_s'], y=yy, mode='lines+markers',
                                     name=nm, line=dict(color=clr, width=2.5)))
        if len(ddf):
            peak = ddf.loc[ddf['Pt'].idxmax()]
            fig.add_vline(x=peak['dept_s'], line_dash='dot', line_color='#d62728',
                          annotation_text=f"最不利 {int(peak['dept_s'])}s",
                          annotation_position='top left')
        fig.add_hline(y=TLS, line_dash='dash', line_color='#555',
                      annotation_text='TLS', annotation_position='bottom right')
        fig.update_yaxes(type='log', title='概率')
        fig.update_xaxes(title='离场已起飞时长 dept (s)')
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=0),
                          legend=dict(font=dict(size=11)), hovermode='x unified')
        st.plotly_chart(fig, width='stretch')
        st.caption('曲线峰值即「最不利起飞时刻」——ADW 下边界反推时所取的最坏离场错开量。')

# ---------- ② 尾流 ----------
with tab_wake:
    st.subheader('尾流影响 → 最小高度差 → 改出角结论')
    st.caption('两机水平接近（≤350 m）时的最小垂直高度差；若小于 300 m（1000 ft）尾流阈值则需更大改出角。')
    wake_deg = st.multiselect('改出角 (°)', [10, 15, 20, 25, 30, 35, 40, 45],
                              default=[15, 20, 25, 30, 35, 40, 45])
    if st.button('计算尾流最小高度差', type='primary'):
        t0 = time.time()
        pkey = tuple(sorted(params.items()))
        key = ('wake', tuple(wake_deg), 20, 1, pkey)
        wdf = _memo(key, lambda cb: compute_wake_min_height(
            degrees=wake_deg, T1=20, step=1, params=params, progress_cb=cb))
        st.session_state['wake_df'] = wdf
        st.session_state['wake_elapsed'] = time.time() - t0
        st.session_state.pop('zip_all', None)
    if 'wake_df' in st.session_state:
        wdf = st.session_state['wake_df']
        el = st.session_state.get('wake_elapsed')
        if el:
            st.caption(f'计算耗时 {el:.2f} s')
        st.download_button('⬇ 下载尾流结果 Excel', _df_to_excel_bytes(wdf, '尾流'),
                           file_name='尾流最小高度差.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        st.dataframe(wdf, width='stretch')
        x = wdf['ALFA']; y = wdf['最小高度差_m'].astype(float)
        colors = ['#2ca02c' if (v and v >= 300) else '#d62728' for v in y]
        fig = go.Figure(go.Bar(x=x, y=y, marker_color=colors))
        fig.add_hline(y=300, line_dash='dash', line_color='#d62728',
                      annotation_text='300 m 阈值', annotation_position='right')
        fig.update_xaxes(title='改出角 (°)')
        fig.update_yaxes(title='最小高度差 (m)')
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
        st.plotly_chart(fig, width='stretch')
        ok = [a for a, v in zip(x, y) if v and v >= 300]
        if ok:
            st.success(f'满足 300 m 尾流阈值的最小改出角 ≈ {min(ok)}°')
        else:
            st.warning('所选改出角范围内均未满足 300 m 尾流阈值，需增大改出角或减小离场错开。')

# ---------- ③ 三维航迹 ----------
with tab_traj:
    st.subheader('进近避让 vs 离场 三维航迹（可旋转 / 缩放）')
    alfa = st.slider('避让改出角 (°)', 10, 45, 35)
    arrt = st.slider('避让起始时刻 arrt (s)', 0, 120, 40)
    x_off = DIRECTION_CONF[direction]['x_off']
    Lx, Ly, Lz = approach_avoidance_traj(
        params['V0'], params['a1'], params['a2'], params['SITA1'], params['SITA2'],
        T1, alfa, FAF_LEN, getz(FAF_LEN), arrt, 200, 1)
    Ldx, Ldy, Ldz = departure_straight_traj(
        params['ad1'], params['Vlof'], params['ad2'], params['V2_10'], params['ad3'],
        params['tansitaD2'], params['tansitaD3'], x_off, 180, 1)
    fig = _traj3d(Lx, Ly, Lz, Ldx, Ldy, Ldz,
                  name1='进近避让航空器', name2='离场航空器',
                  color1='#1a5fa8', color2='#e07b39',
                  title=f'{direction} · 改出角 {alfa}° · arrt={arrt}s')
    st.plotly_chart(fig, width='stretch')
    st.caption('离场 x 偏移：向南 +1430 m / 向北 −4630 m；侧向错开 340 m。')

# ---------- ④ 雷达数据 ----------
with tab_radar:
    st.subheader('雷达航迹数据导入与误差统计')
    st.caption('上传雷达原始 SQL（内含 <SData><PositionReport>…）与呼号表 ac_tp.txt，'
               '自动筛选天府范围、区分进/离场并计算误差统计。')
    import radar
    sql_file = st.file_uploader('雷达数据 SQL', type=['sql', 'txt'])
    ac_file = st.file_uploader('呼号表 ac_tp.txt', type=['txt'])
    if sql_file and ac_file and st.button('解析雷达数据', type='primary'):
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            sqlp = os.path.join(td, 'radar.sql'); acp = os.path.join(td, 'ac.txt')
            with open(sqlp, 'wb') as f: f.write(sql_file.getvalue())
            with open(acp, 'wb') as f: f.write(ac_file.getvalue())
            with st.spinner('解析中（大文件可能较慢）…'):
                df_arr, df_dep = radar.parse_radar_sql(sqlp, acp)
                st.session_state['radar_arr'] = df_arr
                st.session_state['radar_dep'] = df_dep
                st.session_state.pop('zip_all', None)
    if 'radar_arr' in st.session_state:
        da, dd = st.session_state['radar_arr'], st.session_state['radar_dep']
        c1, c2 = st.columns(2)
        c1.metric('进近航班航迹点', len(da))
        c2.metric('离场航班航迹点', len(dd))
        st.markdown('**进近误差 (km)**')
        st.json(radar.approach_error(da))
        st.markdown('**一边误差 (km)**')
        st.json(radar.departure_error(dd))
        st.markdown('**TT465 高度 (m)**')
        st.json(radar.tt465_height(dd))
        st.markdown('**8 个进近区域统计**')
        st.dataframe(radar.app_region_stats(da), width='stretch')
        st.markdown('**离场航迹样例（前 50 行）**')
        st.dataframe(dd.head(50), width='stretch')

# ---------- ⑤ 复飞 ----------
with tab_ga:
    st.subheader('复飞碰撞风险：定点转弯 vs 定高转弯')
    st.caption('复飞碰撞核 12000×12000×300 m（无 /100 归一化，N 默认 1）。'
               '碰撞核远大于两机间距，使 Px、Py≈1，风险退化为垂直接近概率 Pz；'
               '故以「最小间距」判别两种转弯方式的时空差异。')
    ga_mode = st.radio('转弯方式', ['对比 定点 vs 定高', '仅定点', '仅定高'],
                       index=0, horizontal=True)
    cga1, cga2 = st.columns(2)
    ga_step = cga1.slider('复飞时间步长 (s)', 2, 10, 5)
    ga_arrt = cga2.slider('三维航迹 · 复飞起始时刻 arrt (s)', 0, 180, 60)

    if st.button('计算复飞', type='primary'):
        t0 = time.time()
        modes = {'对比 定点 vs 定高': ['定点', '定高'],
                 '仅定点': ['定点'], '仅定高': ['定高']}[ga_mode]
        res = {}
        for m in modes:
            res[m] = {
                'risk': _memo(('ga_risk', m, ga_step),
                              lambda cb, _m=m: compute_goaround(turn_mode=_m, step=ga_step,
                                                                progress_cb=cb)),
                'sep': _memo(('ga_sep', m, ga_step),
                             lambda cb, _m=m: goaround_min_sep(turn_mode=_m, step=ga_step,
                                                               progress_cb=cb)),
            }
        st.session_state['ga_res'] = res
        st.session_state['ga_arrt'] = ga_arrt
        st.session_state['ga_elapsed'] = time.time() - t0
        st.session_state.pop('zip_all', None)

    if 'ga_res' in st.session_state:
        res = st.session_state['ga_res']
        el = st.session_state.get('ga_elapsed')
        rows = []
        for m, r in res.items():
            s = r['sep']
            rows.append({'转弯方式': m,
                         '最小纵向间距 (m)': float(s['min_dx_m']),
                         '最小侧向间距 (m)': float(s['min_dy_m']),
                         '最小垂直间距 (m)': float(s['min_dz_m']),
                         '最小水平间距 (m)': float(s['min_dh_m'])})
        sep_df = pd.DataFrame(rows)
        if el:
            st.caption(f'计算耗时 {el:.2f} s')
        st.download_button('⬇ 下载最小间距对比 Excel', _df_to_excel_bytes(sep_df, '复飞最小间距'),
                           file_name='复飞最小间距.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        st.markdown('**最小间距对比（遍历 L1 × 起飞时刻的全局最小值）**')
        st.dataframe(sep_df.style.format({c: '{:.0f}' for c in sep_df.columns[1:]}),
                     width='stretch')

        risk_all = pd.concat([r['risk'].assign(转弯方式=m) for m, r in res.items()],
                             ignore_index=True)
        st.download_button('⬇ 下载复飞风险表 Excel', _df_to_excel_bytes(risk_all, '复飞风险'),
                           file_name='复飞风险.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        st.markdown('**复飞风险 vs 避让距离 L1（垂直分量 Pz，因 Px、Py 饱和）**')
        fig = go.Figure()
        for m, r in res.items():
            g = r['risk'].sort_values('L1_m')
            fig.add_trace(go.Scatter(x=g['L1_m'], y=g['Pz'], mode='lines+markers',
                                     name=f'{m}转弯'))
        fig.add_hline(y=TLS, line_dash='dash', line_color='#d62728',
                      annotation_text='TLS=5e-9', annotation_position='bottom right')
        fig.update_yaxes(type='log', title='垂直接近概率 Pz')
        fig.update_xaxes(title='避让距离 L1 (m)')
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0),
                          legend=dict(font=dict(size=11)), hovermode='x unified')
        st.plotly_chart(fig, width='stretch')

        st.markdown('**三维航迹叠加（可旋转 / 缩放）**')
        arrt_v = st.session_state.get('ga_arrt', 60)
        p = GOAROUND_DEFAULT
        Ldx, Ldy, Ldz = departure_turn_traj(ga_step, 600)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter3d(x=Ldx, y=Ldy, z=Ldz, mode='lines',
                                    name='RWY11 离场机',
                                    line=dict(color='#e07b39', width=5)))
        colors = {'定点': '#1a5fa8', '定高': '#2ca02c'}
        for m, r in res.items():
            Lx, Ly, Lz = goaround_approach_traj(
                p['v0'], p['a_desc'], p['a2'], p['vM'], p['PDGMA'],
                p['turn_angle'], m, GOAROUND_X0, GOAROUND_Z0,
                p['turn_x'], p['turn_h'], arrt_v, 200, ga_step)
            fig3.add_trace(go.Scatter3d(x=Lx, y=Ly, z=Lz, mode='lines',
                                        name=f'复飞机（{m}转弯）',
                                        line=dict(color=colors[m], width=5)))
        fig3.update_layout(
            title=f'复飞三维航迹 · arrt={arrt_v}s', height=560,
            scene=dict(xaxis_title='纵向 x (m)', yaxis_title='侧向 y (m)',
                       zaxis_title='高度 z (m)', aspectmode='data'),
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.7)'),
            margin=dict(l=0, r=0, t=44, b=0),
        )
        st.plotly_chart(fig3, width='stretch')
        st.caption('说明：复飞模型为从「向北复飞.py」重构（修正其 +437、98°→95° 等笔误）；'
                   '12000 m 碰撞核下风险饱和，最小间距是更有判别力的指标。')


# ---------- ⑥ ADW 直观示意图 ----------
with tab_adw:
    st.subheader('到达离场窗 ADW —— 直观示意图')
    st.caption('依据《成都天府国际机场多跑道管制运行暨到达离场窗(ADW)专项研究报告》：'
               '当进场（或复飞）航空器处于 ADW 窗范围内时，管制员不得向侧向跑道离场航空器发布起飞许可；'
               '离场航空器须等待进场机离开 ADW 范围后方可起飞。')

    c1, c2, c3 = st.columns(3)
    adw_alfa = c1.slider('避让改出角 (°)', 10, 45, 35, key='adw_alfa')
    adw_upper = c2.number_input('ADW 上边界（距跑道入口, m）', 0.0, 3000.0, 900.0,
                                step=100.0, key='adw_upper')
    adw_lower = c3.number_input('ADW 下边界（最小避让距离, m）', 2000.0, 12000.0, 9000.0,
                                step=100.0, key='adw_lower')
    st.caption('终端方案取 ADW 上边界 −900 m、下边界 9000 m；下边界即本工具遍历计算的「最小避让距离 L1」。')

    if 'risk_bd' in st.session_state:
        bd = st.session_state['risk_bd']
        if len(bd):
            row = bd.loc[(bd['ALFA'] - adw_alfa).abs().idxmin()]
            val = float(row['ADW下边界_m'])
            if np.isfinite(val):
                st.info(f'① 页已算：改出角 {row["ALFA"]:.0f}° 的 ADW 下边界 ≈ **{val:.0f} m**'
                        f'（可据此回填上方「下边界」对照）。')

    st.plotly_chart(_adw_plan_view(direction, params, T1, adw_alfa, adw_upper, adw_lower),
                    width='stretch')

    st.markdown('---')
    arr_pos = st.slider('进场机当前距跑道入口距离 (m)', 0, 12000, 4000, key='adw_arr_pos')
    inside = adw_upper <= arr_pos <= adw_lower
    if inside:
        st.markdown(f'<div style="background:#fdecea;border:1px solid #d62728;border-radius:10px;'
                    f'padding:14px 18px;font-size:1.05rem;color:#a31616">'
                    f'🔴 进场机位于 ADW 窗内（{arr_pos:.0f} m）→ RWY11 离场航空器 '
                    f'<b>STOP · 停止起飞</b>，须等待进场机离开 ADW 范围</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background:#e9f7ef;border:1px solid #2ca02c;border-radius:10px;'
                    f'padding:14px 18px;font-size:1.05rem;color:#1e7a3a">'
                    f'🟢 进场机位于 ADW 窗外（{arr_pos:.0f} m）→ RWY11 离场航空器 '
                    f'<b>GO · 可以起飞</b></div>', unsafe_allow_html=True)
    st.plotly_chart(_adw_stop_strip(adw_upper, adw_lower, arr_pos), width='stretch')

    with st.expander('ADW 划设要点（摘自专项研究报告）'):
        st.markdown(
            '- **规则**：进场（或复飞）航空器处于 ADW 窗范围内时，不得向侧向跑道离场航空器发布起飞许可；'
            '离场机须等进场机离开 ADW 范围后方可起飞。\n'
            '- **ADW 下边界**（即最小避让距离，本工具在 ① 页遍历计算）：碰撞核越大、避让改出角越大，'
            '所需下边界越大；以航空器尺寸为碰撞核计算时，风险已满足 TLS，可不设 ADW。\n'
            '- **ADW 上边界**：一般取复飞决断高（DH）处；更保守取跑道入口后 900 m。'
            '验证运行后可将上边界取至跑道入口之上高 120 m（距入口约 1974 m）。')


# ================= 一键导出（放在脚本末尾，结果更新后打包最新数据） =================
zip_data = _bundle_zip()
st.sidebar.download_button(
    '📦 一键导出全部结果 (ZIP)',
    data=zip_data if zip_data else b'',
    file_name='天府机场安全评估结果.zip',
    mime='application/zip',
    disabled=zip_data is None,
    width='stretch',
)
if zip_data is None:
    st.sidebar.caption('计算并展示结果后即可打包下载全部 Excel。')
else:
    st.sidebar.caption('已打包：碰撞风险 / ADW / 尾流 / 复飞 / 雷达（如有）。')
