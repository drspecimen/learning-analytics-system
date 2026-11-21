"""
presentation/theme.py
Visual identity: colors, CSS, chart template and small UI helpers.
Colors follow a validated, colorblind-safe palette; risk colors are status
colors and always appear with a text label, never color alone.
"""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# status colors (risk) — always paired with a label
RISK_COLORS = {"High": "#d03b3b", "Medium": "#fab219", "Low": "#0ca30c", "Not scored": "#a3a29b"}
# categorical identity colors (fixed order, never cycled)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
RESULT_COLORS = {"Distinction": "#2a78d6", "Pass": "#1baf7a", "Fail": "#eb6834", "Withdrawn": "#6250d6"}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8983", "#ecebe7", "#fcfcfb"
BRAND = "#184f95"

pio.templates["las"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=INK_2),
    title=dict(font=dict(size=15, color=INK), x=0, xanchor="left", y=0.98, yref="container", yanchor="top"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=SERIES,
    xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="", automargin=True),
    yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="", automargin=True),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
    margin=dict(l=10, r=10, t=80, b=10),
    hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=GRID),
    bargap=0.25,
))
pio.templates.default = "plotly_white+las"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ background: #0f1f38; }}
[data-testid="stSidebar"] * {{ color: #e8edf5 !important; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding: 6px 8px; border-radius: 8px; }}
[data-testid="stSidebar"] button {{ background: transparent; border: 1px solid rgba(255,255,255,0.35); }}
[data-testid="stSidebar"] button:hover {{ background: rgba(255,255,255,0.1); border-color: white; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: rgba(255,255,255,0.08); }}
[data-testid="stMetric"] {{
    background: white; border: 1px solid {GRID}; border-radius: 12px;
    padding: 14px 16px; box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}}
[data-testid="stMetricLabel"] p {{ color: {INK_2} !important; font-weight: 500; font-size: 0.85rem; }}
[data-testid="stMetricValue"] {{ color: {INK}; font-weight: 700; font-size: 1.65rem; }}
.las-header {{
    background: linear-gradient(120deg, #0f1f38 0%, {BRAND} 100%);
    color: white; border-radius: 14px; padding: 20px 24px; margin-bottom: 18px;
}}
.las-header h2 {{ color: white; margin: 0; font-size: 1.5rem; }}
.las-header p {{ color: #c9d6ea; margin: 4px 0 0 0; font-size: 0.92rem; }}
.las-card {{
    background: white; border: 1px solid {GRID}; border-radius: 12px; padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04); margin-bottom: 12px;
}}
.las-badge {{
    display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem;
    font-weight: 600; color: white;
}}
.las-muted {{ color: {MUTED}; font-size: 0.85rem; }}
.las-factor {{
    border-left: 3px solid #d03b3b; background: #fdf3f3; padding: 8px 12px;
    border-radius: 6px; margin-bottom: 6px; font-size: 0.9rem; color: {INK};
}}
.las-login {{
    background: white; border: 1px solid {GRID}; border-radius: 16px; padding: 28px;
    box-shadow: 0 8px 24px rgba(16,24,40,0.08);
}}
div[data-testid="stTabs"] button p {{ font-weight: 600; }}
</style>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def header(title, subtitle=""):
    st.markdown(f"<div class='las-header'><h2>{title}</h2><p>{subtitle}</p></div>",
                unsafe_allow_html=True)


def risk_badge(level):
    color = RISK_COLORS.get(level, "#a3a29b")
    icon = {"High": "▲", "Medium": "●", "Low": "▼"}.get(level, "○")
    text_color = "#0b0b0b" if level == "Medium" else "white"
    return (f"<span class='las-badge' style='background:{color};color:{text_color}'>"
            f"{icon} {level} risk</span>")


def card(html):
    st.markdown(f"<div class='las-card'>{html}</div>", unsafe_allow_html=True)


def pct(x, digits=0):
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"
