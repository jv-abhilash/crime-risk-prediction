"""
P25 — Streamlit Emergency Dispatch Intelligence App
Place this file in the project root and run:
    streamlit run app.py
"""

import os, sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── resolve project root so imports work regardless of CWD ──────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from scripts.p25_inference import P25Pipeline

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="P25 — Emergency Dispatch Intelligence",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — security / dispatch theme
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── page background ── */
.stApp { background-color: #0d1117; color: #e6edf3; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1a2744 100%);
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stMarkdown p { color: #8b949e; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #58a6ff; }

/* ── metric cards ── */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px;
}
[data-testid="metric-container"] label { color: #8b949e !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e6edf3 !important; font-size: 1.6rem !important; font-weight: 700;
}

/* ── section headers ── */
h1 { color: #58a6ff !important; font-weight: 700; }
h2 { color: #79c0ff !important; font-weight: 600; }
h3 { color: #d2a8ff !important; font-weight: 600; }

/* ── dividers ── */
hr { border-color: #30363d; }

/* ── dispatch decision card ── */
.dispatch-l1 {
    background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
    border: 2px solid #f87171; border-radius: 12px;
    padding: 28px 32px; text-align: center;
    box-shadow: 0 0 30px rgba(248,113,113,0.3);
}
.dispatch-l2 {
    background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
    border: 2px solid #fbbf24; border-radius: 12px;
    padding: 28px 32px; text-align: center;
    box-shadow: 0 0 30px rgba(251,191,36,0.25);
}
.dispatch-std {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
    border: 2px solid #34d399; border-radius: 12px;
    padding: 28px 32px; text-align: center;
    box-shadow: 0 0 20px rgba(52,211,153,0.2);
}
.dispatch-icon { font-size: 3.5rem; margin-bottom: 8px; }
.dispatch-title { font-size: 1.1rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    opacity: 0.85; margin-bottom: 6px; }
.dispatch-decision { font-size: 1.9rem; font-weight: 700; }
.dispatch-sub { font-size: 0.88rem; opacity: 0.75; margin-top: 8px; }

/* ── info cards ── */
.info-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 18px 22px; height: 100%;
}
.info-card-title { color: #8b949e; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 8px; }
.info-card-value { color: #e6edf3; font-size: 1.5rem; font-weight: 700; }
.info-card-sub { color: #58a6ff; font-size: 0.85rem; margin-top: 4px; }

/* ── feature badge ── */
.feat-badge {
    display: inline-block; background: #1f2937;
    border: 1px solid #374151; border-radius: 6px;
    padding: 4px 10px; margin: 3px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
    color: #93c5fd;
}

/* ── welcome banner ── */
.welcome-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 100%);
    border: 1px solid #1d4ed8; border-radius: 12px;
    padding: 36px 40px; text-align: center;
}

/* ── status pill ── */
.pill-online {
    display: inline-block; background: #14532d; color: #86efac;
    border: 1px solid #16a34a; border-radius: 20px;
    padding: 3px 12px; font-size: 0.78rem; font-weight: 600;
}
.pill-warn {
    display: inline-block; background: #78350f; color: #fde68a;
    border: 1px solid #d97706; border-radius: 20px;
    padding: 3px 12px; font-size: 0.78rem; font-weight: 600;
}

/* ── stSelectbox, stRadio ── */
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: #161b22; border-color: #30363d;
}
div[data-testid="stRadio"] label { color: #c9d1d9; }

/* ── buttons ── */
.stButton button {
    background: linear-gradient(90deg, #1d4ed8, #2563eb);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; letter-spacing: 0.04em;
    width: 100%; padding: 12px;
    transition: all 0.2s;
}
.stButton button:hover {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    box-shadow: 0 0 15px rgba(59,130,246,0.4);
}

/* ── progress bar ── */
.stProgress > div > div > div { background: #1d4ed8; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# LOAD PIPELINE (cached — loads models once at startup)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading P25 intelligence models...")
def load_pipeline(policy):
    return P25Pipeline(policy=policy)

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════
def dispatch_card(tier, p_emergency, threshold):
    if "Level-1" in tier:
        cls = "dispatch-l1"
        icon = "🚨"
        label = "LEVEL-1 ALERT"
        color = "#f87171"
    elif "Level-2" in tier:
        cls = "dispatch-l2"
        icon = "⚠️"
        label = "LEVEL-2 STANDBY"
        color = "#fbbf24"
    else:
        cls = "dispatch-std"
        icon = "✅"
        label = "STANDARD PATROL"
        color = "#34d399"

    return f"""
    <div class="{cls}">
        <div class="dispatch-icon">{icon}</div>
        <div class="dispatch-title">Dispatch Decision</div>
        <div class="dispatch-decision" style="color:{color}">{tier}</div>
        <div class="dispatch-sub">
            P(Emergency) = {p_emergency:.4f} &nbsp;|&nbsp;
            Threshold = {threshold}
        </div>
    </div>"""

def gauge_chart(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 28, "color": "#e6edf3"}},
        title={"text": title, "font": {"size": 13, "color": "#8b949e"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "#30363d", "tickfont": {"color": "#8b949e"}},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "#161b22",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 27],  "color": "#064e3b"},
                {"range": [27, 61], "color": "#1e3a5f"},
                {"range": [61, 80], "color": "#78350f"},
                {"range": [80, 100],"color": "#7f1d1d"},
            ],
            "threshold": {
                "line": {"color": "#f8f8f8", "width": 3},
                "thickness": 0.85,
                "value": value * 100,
            },
        }
    ))
    fig.update_layout(
        height=220, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#0d1117", font_color="#e6edf3",
    )
    return fig

def bar_chart_y1hat(result):
    names = ["LinearRegression","Ridge","RandomForest","GradientBoosting"]
    vals  = [result["Y1_hat_LinReg"], result["Y1_hat_Ridge"],
             result["Y1_hat_RF"],     result["Y1_hat_GBM"]]
    colors = ["#58a6ff","#79c0ff","#d2a8ff","#ffa657"]
    fig = go.Figure(go.Bar(
        x=names, y=vals, marker_color=colors,
        text=[f"{v:.4f}" for v in vals],
        textposition="outside", textfont={"color":"#e6edf3","size":11},
    ))
    fig.update_layout(
        title={"text":"Stage 1 — Crime Intensity Scores (Y1_hat)",
               "font":{"size":13,"color":"#8b949e"}},
        xaxis={"tickfont":{"color":"#c9d1d9","size":10},
               "gridcolor":"#21262d"},
        yaxis={"range":[0,1.1], "tickfont":{"color":"#8b949e"},
               "gridcolor":"#21262d", "title":"Crime Intensity Score"},
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        margin=dict(l=10, r=10, t=40, b=10), height=260,
        showlegend=False,
    )
    fig.add_hline(y=0.48, line_dash="dash", line_color="#f87171",
                  annotation_text="Emergency threshold",
                  annotation_font_color="#f87171")
    return fig

def bar_chart_d2(d2_features):
    labels = list(d2_features.keys())
    vals   = list(d2_features.values())
    short  = [l.replace("state_","").replace("_rate","").replace("_1995","")
               .replace("_"," ").title() for l in labels]
    colors = ["#58a6ff" if v < 0.3 else "#fbbf24" if v < 0.6 else "#f87171"
              for v in vals]
    fig = go.Figure(go.Bar(
        x=short, y=vals, marker_color=colors,
        text=[f"{v:.3f}" for v in vals],
        textposition="outside", textfont={"color":"#e6edf3","size":10},
    ))
    fig.update_layout(
        title={"text":"Stage 2 — State Crime Context (D2 features, normalised)",
               "font":{"size":13,"color":"#8b949e"}},
        xaxis={"tickfont":{"color":"#c9d1d9","size":10},"gridcolor":"#21262d"},
        yaxis={"range":[0,1.15],"tickfont":{"color":"#8b949e"},
               "gridcolor":"#21262d","title":"Normalised Rate [0,1]"},
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        margin=dict(l=10, r=10, t=40, b=10), height=250,
        showlegend=False,
    )
    return fig

def feature_radar(top_features):
    cats  = list(top_features.keys())
    vals  = list(top_features.values())
    vals += [vals[0]]
    cats += [cats[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        line_color="#58a6ff", fillcolor="rgba(88,166,255,0.15)",
        marker=dict(color="#58a6ff", size=6),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1],
                            tickfont={"color":"#8b949e","size":9},
                            gridcolor="#30363d"),
            angularaxis=dict(tickfont={"color":"#c9d1d9","size":10},
                             gridcolor="#30363d"),
            bgcolor="#161b22",
        ),
        paper_bgcolor="#161b22",
        margin=dict(l=30, r=30, t=30, b=30), height=280,
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-size:2.5rem;'>🚨</div>
        <div style='font-size:1.1rem; font-weight:700; color:#58a6ff;
                    letter-spacing:0.08em;'>P25 DISPATCH</div>
        <div style='font-size:0.75rem; color:#8b949e; letter-spacing:0.12em;'>
            INTELLIGENCE SYSTEM</div>
        <div style='margin-top:8px;'>
            <span class='pill-online'>● ONLINE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Select Community")

    # Policy selector first (needed to load pipeline)
    policy_map = {
        "Optimal F1 — Best balance":       "optimal_f1",
        "Recall ≥ 90% — High safety":      "recall_90",
        "Min Cost — FN×10 + FP×1":         "min_cost",
    }
    policy_label = st.radio(
        "Threshold Policy",
        options=list(policy_map.keys()),
        index=0,
        help="Controls the P(Emergency) cut-off for dispatch decisions",
    )
    policy = policy_map[policy_label]

    # Load pipeline with selected policy
    try:
        pipe = load_pipeline(policy)
        st.markdown(f"<span class='pill-online'>● Models loaded</span>",
                    unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Pipeline load failed: {e}")
        st.stop()

    st.markdown("")

    # Community search
    search_term = st.text_input("🔍 Search community name",
                                 placeholder="e.g. Selma, California")
    all_communities = pipe.get_community_list()

    if search_term:
        filtered = [c for c in all_communities
                    if search_term.lower() in c.lower()]
    else:
        filtered = all_communities

    if not filtered:
        st.warning("No communities match. Try a different search term.")
        selected_community = None
    else:
        selected_community = st.selectbox(
            f"Community ({len(filtered)} shown)",
            options=filtered,
            index=0,
        )

    st.markdown("---")
    predict_clicked = st.button("🚀 RUN INFERENCE", use_container_width=True)

    st.markdown("---")

    # Model info panel
    with st.expander("📊 Model Information"):
        try:
            s2_df = pd.read_csv(
                os.path.join(ROOT, "outputs/p25_outputs/stage2_test_metrics.csv"))
            s1_df = pd.read_csv(
                os.path.join(ROOT, "outputs/p25_outputs/stage1_test_metrics.csv"))
            thr_df = pd.read_csv(
                os.path.join(ROOT, "outputs/p25_outputs/thresholds.csv"))

            best_s2 = s2_df[s2_df["is_best"]==True].iloc[0]
            best_s1 = s1_df.loc[s1_df["test_r2"].idxmax()]
            thr_row = thr_df[thr_df["threshold_type"]==policy].iloc[0]

            st.markdown(f"""
            **Best Stage 2:** {best_s2['model_name']}
            - AUC: `{best_s2['test_auc']:.4f}`
            - F1:  `{best_s2['test_f1']:.4f}`
            - Recall: `{best_s2['test_recall']:.4f}`

            **Best Stage 1:** {best_s1['model_name']}
            - Test R²: `{best_s1['test_r2']:.4f}`

            **Threshold ({policy}):** `{thr_row['value']}`
            - F1: `{thr_row['f1']:.4f}`
            - Recall: `{thr_row['recall']:.4f}`
            """)
        except Exception:
            st.info("Run the pipeline first to see model metrics.")

    st.markdown("""
    <div style='text-align:center; margin-top:20px; color:#484f58; font-size:0.75rem;'>
        P25 · Two-Stage Crime Risk Pipeline<br>
        UCI Communities & Crime · CORGIS 1995
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<h1 style='margin-bottom:0;'>🚨 Emergency Dispatch Intelligence</h1>
<p style='color:#8b949e; margin-top:4px; font-size:0.95rem;'>
Two-Stage Crime Risk Prediction · UCI Communities & Crime · 1994 US Communities
</p>
""", unsafe_allow_html=True)

# ── Welcome state (no prediction yet) ────────────────────────────────────────
if not predict_clicked or selected_community is None:
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='info-card'>
            <div class='info-card-title'>Total Communities</div>
            <div class='info-card-value'>1,994</div>
            <div class='info-card-sub'>Across 46 US States</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='info-card'>
            <div class='info-card-title'>Best Model AUC</div>
            <div class='info-card-value'>0.9328</div>
            <div class='info-card-sub'>LogisticRegression · Stage 2</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='info-card'>
            <div class='info-card-title'>Emergency Rate</div>
            <div class='info-card-value'>15.4%</div>
            <div class='info-card-sub'>308 of 1,994 communities</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='welcome-banner'>
        <div style='font-size:2rem; margin-bottom:12px;'>🛡️</div>
        <h2 style='color:#58a6ff; margin:0 0 12px 0;'>
            Two-Stage Crime Risk Pipeline</h2>
        <p style='color:#8b949e; max-width:600px; margin:0 auto 20px auto;'>
        Select a community from the sidebar and click
        <strong style='color:#58a6ff;'>RUN INFERENCE</strong> to
        predict the emergency activation probability and dispatch tier.
        </p>
        <div style='display:flex; gap:24px; justify-content:center;
                    flex-wrap:wrap; margin-top:16px;'>
            <div style='background:#1f2937; border-radius:8px; padding:14px 20px;
                        border:1px solid #374151; text-align:left; min-width:180px;'>
                <div style='color:#58a6ff; font-weight:600; margin-bottom:6px;'>
                    Stage 1 → Regression</div>
                <div style='color:#8b949e; font-size:0.85rem;'>
                    4 models predict crime<br>intensity from 100 features</div>
            </div>
            <div style='background:#1f2937; border-radius:8px; padding:14px 20px;
                        border:1px solid #374151; text-align:left; min-width:180px;'>
                <div style='color:#d2a8ff; font-weight:600; margin-bottom:6px;'>
                    Stage 2 → Classification</div>
                <div style='color:#8b949e; font-size:0.85rem;'>
                    11 features → P(Emergency)<br>→ Dispatch decision</div>
            </div>
            <div style='background:#1f2937; border-radius:8px; padding:14px 20px;
                        border:1px solid #374151; text-align:left; min-width:180px;'>
                <div style='color:#34d399; font-weight:600; margin-bottom:6px;'>
                    3-Tier Dispatch</div>
                <div style='color:#8b949e; font-size:0.85rem;'>
                    Full Deployment / Standby<br>/ Standard Patrol</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Prediction state ──────────────────────────────────────────────────────────
else:
    with st.spinner(f"Running inference on {selected_community}..."):
        result = pipe.predict_community(selected_community)

    p = result["P_emergency"]
    tier = result["dispatch_tier"]

    # ── Row 1: Dispatch decision card (full width) ────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(dispatch_card(tier, p, pipe.threshold), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: 4 quick metrics ────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("P(Emergency)", f"{p:.4f}",
                  delta=f"Threshold: {pipe.threshold}",
                  delta_color="off")
    with c2:
        best_y1 = max(result["Y1_hat_RF"], result["Y1_hat_GBM"])
        st.metric("Crime Intensity (GBM)", f"{result['Y1_hat_GBM']:.4f}",
                  delta="Stage 1 output", delta_color="off")
    with c3:
        true_lbl = "Emergency 🔴" if result["true_emergency"]==1 else "No Emergency 🟢"
        st.metric("True Label", true_lbl)
    with c4:
        correct = ((p >= pipe.threshold) == bool(result["true_emergency"]))
        st.metric("Prediction", "✓ Correct" if correct else "✗ Incorrect",
                  delta=result["state"],
                  delta_color="off")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3: Gauge + Stage 1 bars ───────────────────────────────────────────
    col_gauge, col_bars = st.columns([1, 2])

    with col_gauge:
        if "Level-1" in tier:
            gauge_color = "#f87171"
        elif "Level-2" in tier:
            gauge_color = "#fbbf24"
        else:
            gauge_color = "#34d399"
        st.plotly_chart(
            gauge_chart(p, "P(Emergency Activation)", gauge_color),
            use_container_width=True)

    with col_bars:
        st.plotly_chart(
            bar_chart_y1hat(result),
            use_container_width=True)

    # ── Row 4: D2 features + Radar ────────────────────────────────────────────
    col_d2, col_radar = st.columns([3, 2])

    with col_d2:
        st.plotly_chart(
            bar_chart_d2(result["d2_features"]),
            use_container_width=True)

    with col_radar:
        st.markdown("<h3 style='margin-bottom:8px;'>Community Profile</h3>",
                    unsafe_allow_html=True)
        st.plotly_chart(
            feature_radar(result["top_features"]),
            use_container_width=True)

    # ── Row 5: Key feature values + pipeline steps ────────────────────────────
    col_feat, col_pipe = st.columns([1, 1])

    with col_feat:
        st.markdown("### 📋 Key D1 Features")
        st.markdown("<div style='color:#8b949e; font-size:0.82rem; "
                    "margin-bottom:10px;'>Top demographic features [0=low, 1=high]"
                    "</div>", unsafe_allow_html=True)
        feat_labels = {
            "PctIlleg":      "% Out-of-wedlock births",
            "PctKids2Par":   "% Kids with 2 parents",
            "medIncome":     "Median household income",
            "PctUnemployed": "% Unemployed",
            "racepctblack":  "% Black population",
        }
        for feat, label in feat_labels.items():
            val = result["top_features"].get(feat, 0)
            color = "#f87171" if val > 0.6 else "#fbbf24" if val > 0.3 else "#34d399"
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; "
                f"padding:8px 12px; background:#161b22; border-radius:6px; "
                f"margin-bottom:4px; border-left:3px solid {color};'>"
                f"<span style='color:#c9d1d9;'>{label}</span>"
                f"<span style='color:{color}; font-weight:600; "
                f"font-family:JetBrains Mono,monospace;'>{val:.4f}</span>"
                f"</div>",
                unsafe_allow_html=True)

    with col_pipe:
        st.markdown("### 🔗 Pipeline Trace")
        stage1_avg = np.mean([result["Y1_hat_LinReg"], result["Y1_hat_Ridge"],
                               result["Y1_hat_RF"],    result["Y1_hat_GBM"]])
        steps = [
            ("Stage 1 Input",  "100 D1 features",       "#58a6ff"),
            ("Stage 1 Output", f"Y1_hat avg = {stage1_avg:.4f}", "#79c0ff"),
            ("Stage 2 Input",  "7 D2 + 4 Y1_hat = 11",  "#d2a8ff"),
            ("Stage 2 Output", f"P(Emergency) = {p:.4f}","#ffa657"),
            ("Decision",       tier,                     gauge_color),
        ]
        for label, value, color in steps:
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:12px; "
                f"padding:10px 14px; background:#161b22; border-radius:8px; "
                f"margin-bottom:6px; border-left:4px solid {color};'>"
                f"<div>"
                f"<div style='color:#8b949e; font-size:0.75rem; "
                f"text-transform:uppercase; letter-spacing:0.08em;'>{label}</div>"
                f"<div style='color:{color}; font-weight:600; "
                f"font-family:JetBrains Mono,monospace; font-size:0.9rem;'>"
                f"{value}</div>"
                f"</div></div>",
                unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; color:#484f58; font-size:0.8rem;'>"
        f"P25 · Two-Stage Crime Risk Pipeline · "
        f"Community: <strong style='color:#8b949e'>{selected_community}</strong> · "
        f"Policy: <strong style='color:#8b949e'>{policy}</strong> · "
        f"Model: <strong style='color:#8b949e'>{pipe.s2_name}</strong>"
        f"</div>",
        unsafe_allow_html=True)
