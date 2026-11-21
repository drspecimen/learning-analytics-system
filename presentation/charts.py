"""
presentation/charts.py
Reusable Plotly charts. Every chart has hover tooltips; colors carry identity
or status and are always backed by labels or a legend.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from presentation.theme import (GRID, INK, RESULT_COLORS, RISK_COLORS, SEQ_BLUE, SERIES)

RISK_ORDER = ["High", "Medium", "Low"]
RESULT_ORDER = ["Distinction", "Pass", "Fail", "Withdrawn"]


def _finish(fig, height=360):
    fig.update_layout(height=height, legend_title_text="")
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    return fig


def risk_bar(view, title="Students by risk level"):
    counts = view["risk_level"].value_counts().reindex(RISK_ORDER, fill_value=0).reset_index()
    counts.columns = ["Risk", "Enrollments"]
    fig = px.bar(counts, x="Risk", y="Enrollments", color="Risk", text="Enrollments",
                 color_discrete_map=RISK_COLORS, title=title)
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, xaxis_title=None)
    return _finish(fig, 320)


def risk_by_group(view, group, title, top=None):
    t = view.groupby([group, "risk_level"]).size().unstack(fill_value=0)
    t = t.reindex(columns=RISK_ORDER, fill_value=0)
    share = (t["High"] / t.sum(axis=1)).sort_values(ascending=True)
    if top:
        share = share.tail(top)
    t = t.loc[share.index]
    fig = go.Figure()
    for lvl in reversed(RISK_ORDER):
        fig.add_bar(y=t.index, x=t[lvl], name=lvl, orientation="h", marker_color=RISK_COLORS[lvl],
                    hovertemplate=f"%{{y}}<br>{lvl} risk: %{{x}}<extra></extra>")
    fig.update_layout(barmode="stack", title=title, xaxis_title="Enrollments", yaxis_title=None,
                      legend_traceorder="reversed")
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    return fig.update_layout(height=max(320, 28 * len(t) + 120))


def outcome_bar(view, col="predicted_result", title="Predicted outcome"):
    counts = view[col].value_counts().reindex(RESULT_ORDER, fill_value=0).reset_index()
    counts.columns = ["Outcome", "Count"]
    fig = px.bar(counts, x="Outcome", y="Count", color="Outcome", text="Count",
                 color_discrete_map=RESULT_COLORS, title=title)
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, xaxis_title=None)
    return _finish(fig, 320)


def weekly_line(weekly, compare=None, y="clicks", title="Weekly LMS engagement",
                name="Selected", compare_name="Institution average"):
    fig = go.Figure()
    fig.add_scatter(x=weekly["week"], y=weekly[y], mode="lines+markers", name=name,
                    line=dict(color=SERIES[0], width=2), marker=dict(size=8),
                    hovertemplate="Week %{x}<br>%{y:,.0f}<extra>" + name + "</extra>")
    if compare is not None and len(compare):
        fig.add_scatter(x=compare["week"], y=compare[y], mode="lines", name=compare_name,
                        line=dict(color=SERIES[1], width=2, dash="dot"),
                        hovertemplate="Week %{x}<br>%{y:,.0f}<extra>" + compare_name + "</extra>")
    fig.update_layout(title=title, xaxis_title="Week", yaxis_title=y.replace("_", " ").title(),
                      hovermode="x unified", xaxis=dict(dtick=1))
    return _finish(fig, 340)


def score_hist(view, title="Average assessment score distribution"):
    fig = px.histogram(view, x="avg_score", nbins=25, title=title,
                       color_discrete_sequence=[SERIES[0]])
    fig.update_traces(marker_line_color="white", marker_line_width=1,
                      hovertemplate="Score %{x}<br>%{y} enrollments<extra></extra>")
    fig.update_layout(xaxis_title="Average score (%)", yaxis_title="Enrollments", bargap=0.05)
    return _finish(fig, 320)


def engagement_scatter(view, title="Engagement vs performance"):
    d = view.copy()
    d["risk_level"] = pd.Categorical(d["risk_level"], RISK_ORDER + ["Not scored"])
    fig = px.scatter(d, x="total_clicks", y="avg_score", color="risk_level",
                     color_discrete_map=RISK_COLORS, opacity=0.7, title=title,
                     hover_data={"student_name": True, "course_code": True, "matric_no": True,
                                 "total_clicks": ":,", "avg_score": ":.1f", "risk_level": True},
                     category_orders={"risk_level": RISK_ORDER})
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")))
    fig.update_layout(xaxis_title="LMS clicks (to date)", yaxis_title="Average score (%)")
    return _finish(fig, 380)


def gauge(value, title="Risk score"):
    lvl = "High" if value >= 0.6 else "Medium" if value >= 0.35 else "Low"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value * 100, number={"suffix": "%", "font": {"size": 34, "color": INK}},
        gauge={"axis": {"range": [0, 100], "tickwidth": 0, "tickvals": [0, 35, 60, 100]},
               "bar": {"color": RISK_COLORS[lvl], "thickness": 0.35},
               "bgcolor": "white",
               "steps": [{"range": [0, 35], "color": "#eaf6ea"},
                         {"range": [35, 60], "color": "#fff5df"},
                         {"range": [60, 100], "color": "#fbe9e9"}]}))
    fig.update_layout(height=230, margin=dict(l=35, r=40, t=30, b=10))
    return fig


def prob_bar(probs, title="Outcome probabilities"):
    d = pd.DataFrame({"Outcome": list(probs.keys()), "Probability": list(probs.values())})
    d["Outcome"] = pd.Categorical(d["Outcome"], RESULT_ORDER)
    d = d.sort_values("Outcome")
    fig = px.bar(d, x="Probability", y="Outcome", orientation="h", color="Outcome",
                 color_discrete_map=RESULT_COLORS, title=title, text=d["Probability"].map("{:.0%}".format))
    fig.update_layout(showlegend=False, xaxis=dict(range=[0, 1.18], tickformat=".0%", tickvals=[0, .25, .5, .75, 1]), yaxis_title=None)
    fig.update_traces(textposition="outside", cliponaxis=False,
                      hovertemplate="%{y}: %{x:.1%}<extra></extra>")
    return _finish(fig, 240)


def grades_bar(grades, course_avg=None, title="Assessment scores"):
    g = grades.copy()
    g["label"] = g["title"]
    g["shown"] = g["score"].fillna(0)
    fig = go.Figure()
    fig.add_bar(x=g["label"], y=g["shown"], name="Student", marker_color=SERIES[0],
                customdata=g[["feedback"]].fillna(""),
                hovertemplate="%{x}<br>Score: %{y:.1f}<br>%{customdata[0]}<extra></extra>")
    if course_avg is not None:
        c = course_avg.set_index("title").reindex(g["label"])
        fig.add_bar(x=g["label"], y=c["avg"], name="Course average", marker_color=SERIES[1],
                    hovertemplate="%{x}<br>Course avg: %{y:.1f}<extra></extra>")
    fig.update_layout(title=title, barmode="group", yaxis=dict(range=[0, 105], title="Score"),
                      xaxis_title=None)
    return _finish(fig, 320)


def confusion(cm, labels, title):
    fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale=SEQ_BLUE,
                    labels=dict(x="Predicted", y="Actual", color="Count"), title=title, aspect="auto")
    fig.update_layout(coloraxis_showscale=False)
    return fig.update_layout(height=380)


def roc(results, title="ROC curve — identifying at-risk students"):
    fig = go.Figure()
    for i, (name, r) in enumerate(results.items()):
        fig.add_scatter(x=r["roc_curve"]["fpr"], y=r["roc_curve"]["tpr"], mode="lines",
                        name=f"{name} (AUC {r['at_risk_auc']:.2f})", line=dict(width=2, color=SERIES[i]))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                    line=dict(color=GRID, dash="dash", width=2), hoverinfo="skip")
    fig.update_layout(title=title, xaxis_title="False positive rate", yaxis_title="True positive rate",
                      legend=dict(orientation="v", x=0.98, y=0.04, xanchor="right", yanchor="bottom",
                                  bgcolor="rgba(255,255,255,0.85)"))
    return fig.update_layout(height=380)


def importance_bar(imp, labels, top=12, title="What drives the predictions"):
    s = pd.Series(imp).sort_values(ascending=False).head(top)[::-1]
    fig = go.Figure(go.Bar(x=s.values, y=[labels.get(k, k) for k in s.index], orientation="h",
                           marker_color=SERIES[0],
                           hovertemplate="%{y}<br>Importance share: %{x:.1%}<extra></extra>"))
    fig.update_layout(title=title, xaxis=dict(tickformat=".0%", title="Share of importance"))
    return fig.update_layout(height=420)


def metric_compare(results):
    rows = []
    for name, r in results.items():
        for m, label in [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                         ("f1_score", "F1"), ("auc_roc", "AUC-ROC")]:
            rows.append({"Model": name, "Metric": label, "Score": r[m] or 0})
    d = pd.DataFrame(rows)
    fig = px.bar(d, x="Metric", y="Score", color="Model", barmode="group",
                 color_discrete_sequence=SERIES, title="Model comparison (test set)")
    fig.update_traces(hovertemplate="%{x}: %{y:.3f}<extra>%{fullData.name}</extra>")
    fig.update_layout(yaxis=dict(range=[0, 1]), xaxis_title=None)
    return _finish(fig, 360)


def history_outcomes(hist, group="department", title="Past session outcomes by department"):
    t = hist.groupby([group, "final_result"]).size().reset_index(name="n")
    tot = t.groupby(group)["n"].transform("sum")
    t["share"] = t["n"] / tot
    fig = px.bar(t, x="share", y=group, color="final_result", orientation="h",
                 color_discrete_map=RESULT_COLORS, category_orders={"final_result": RESULT_ORDER},
                 title=title, custom_data=["n"])
    fig.update_traces(marker_line_color="white", marker_line_width=1,
                      hovertemplate="%{y}<br>%{fullData.name}: %{x:.0%} (%{customdata[0]})<extra></extra>")
    fig.update_layout(barmode="stack", xaxis=dict(tickformat=".0%", title=None), yaxis_title=None)
    return fig.update_layout(height=max(320, 34 * t[group].nunique() + 120))
