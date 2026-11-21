"""
presentation/pages.py
Pages shared by Administrators and Lecturers. Each page takes the signed-in `user`:
Administrators see their whole institution; lecturers only their own courses.
No page can ever reach another institution's data: every query uses user["institution_id"].
"""

import pandas as pd
import streamlit as st

from business import analytics, notifications
from database import db
from ml import ml_service
from ml.features import FEATURE_LABELS
from presentation import charts
from presentation.theme import RISK_COLORS, card, header, pct, risk_badge

RISK_ORDER = ["High", "Medium", "Low"]


# ---------------------------------------------------------------- helpers

def show(fig, key=None):
    st.plotly_chart(fig, width="stretch", theme=None, key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def scope(user):
    """(institution_id, lecturer filter). Administrators see the whole institution;
    lecturers only their own courses. Both values come from the signed-in session."""
    iid = user["institution_id"]
    return iid, (None if user["role"] == "Administrator" else user["user_id"])


@st.cache_data(show_spinner="Loading analytics...", max_entries=64, ttl=3600)
def _cached_view(iid, lecturer_id, version):
    return analytics.enrollment_view(iid, lecturer_id)


def get_view(user):
    iid, lec = scope(user)
    return _cached_view(iid, lec, db.data_version(iid))


def no_data(user):
    if user["role"] == "Administrator":
        st.info("No student data yet. Go to **Data Management** to load demo data or upload your "
                "institution's dataset.", icon="ℹ️")
    else:
        st.info("No students in your courses yet. Your administrator needs to load data and assign "
                "courses to you.", icon="ℹ️")


def open_profile(student_id):
    st.session_state.profile_student_id = int(student_id)
    st.session_state.nav_target = "Student Profile"
    st.rerun()


def empty_state(msg):
    st.info(msg, icon="ℹ️")


def risk_column_config():
    return {
        "risk_score": st.column_config.ProgressColumn("Risk score", format="percent", min_value=0, max_value=1),
        "max_risk": st.column_config.ProgressColumn("Highest risk", format="percent", min_value=0, max_value=1),
        "avg_score": st.column_config.NumberColumn("Avg score", format="%.1f"),
        "submission_rate": st.column_config.ProgressColumn("Submission rate", format="percent",
                                                           min_value=0, max_value=1),
        "total_clicks": st.column_config.NumberColumn("LMS clicks", format="%d"),
        "weeks_since_active": st.column_config.NumberColumn("Weeks inactive", format="%d"),
        "confidence_score": st.column_config.NumberColumn("Confidence", format="percent"),
    }


def filters(view, key, show_course=True, show_faculty=True):
    """One row of filters above the content. Returns the filtered view."""
    cols = st.columns([1, 1, 1, 1.4])
    out = view
    if show_faculty:
        fac = cols[0].multiselect("Faculty", sorted(view["faculty"].dropna().unique()), key=f"{key}_fac")
        if fac:
            out = out[out["faculty"].isin(fac)]
    if show_course:
        crs = cols[1].multiselect("Course", sorted(out["course_code"].unique()), key=f"{key}_crs")
        if crs:
            out = out[out["course_code"].isin(crs)]
    risk = cols[2].multiselect("Risk level", RISK_ORDER, key=f"{key}_risk")
    if risk:
        out = out[out["risk_level"].isin(risk)]
    q = cols[3].text_input("Search name or matric no.", key=f"{key}_q")
    if q:
        ql = q.lower()
        out = out[out["student_name"].str.lower().str.contains(ql, regex=False)
                  | out["matric_no"].str.lower().str.contains(ql, regex=False)]
    return out


# ---------------------------------------------------------------- overview

def overview(user):
    iid, lecturer_id = scope(user)
    title = user.get("institution_name", "Institution") if lecturer_id is None else "My courses"
    view = get_view(user)
    sess, week = analytics.current_session(iid), analytics.current_week(iid)
    header(f"{title} — overview", f"Session {sess or '—'} · week {week} of 14 · live predictions")
    if view.empty:
        no_data(user)
        return

    students = view["student_id"].nunique()
    high = view[view["risk_level"] == "High"]
    c = st.columns(6)
    c[0].metric("Students", f"{students:,}")
    c[1].metric("Course enrollments", f"{len(view):,}")
    c[2].metric("Avg score", f"{view['avg_score'].mean():.1f}%")
    c[3].metric("Submission rate", pct(view["submission_rate"].mean()))
    c[4].metric("High-risk students", f"{high['student_id'].nunique():,}",
                help=f"{high['student_id'].nunique() / students:.0%} of students are high risk in at least one course")
    inactive = view[view["weeks_since_active"] >= 2]["student_id"].nunique()
    c[5].metric("Inactive 2+ weeks", f"{inactive:,}", delta_color="off")

    l, r = st.columns(2)
    with l:
        show(charts.risk_bar(view, "Enrollments by risk level"))
    with r:
        show(charts.outcome_bar(view, title="Predicted end-of-semester outcome"))

    ids = view["enrollment_id"].tolist()
    weekly = analytics.weekly_engagement(iid, ids)
    weekly["clicks"] = weekly["clicks"] / max(len(ids), 1)
    weekly["logins"] = weekly["logins"] / max(len(ids), 1)
    l, r = st.columns(2)
    with l:
        show(charts.weekly_line(weekly, y="clicks", name="Clicks per enrollment",
                                title="Weekly engagement (average clicks per enrollment)"))
    with r:
        active = weekly.copy()
        active["active_share"] = active["active_students"] / max(len(ids), 1) * 100
        fig = charts.weekly_line(active, y="active_share", name="Active enrollments %",
                                 title="Share of enrollments active each week")
        fig.update_layout(yaxis=dict(title="% active", range=[0, 100]))
        show(fig)

    group = "department" if lecturer_id is None else "course_code"
    l, r = st.columns(2)
    with l:
        show(charts.risk_by_group(view, group, f"Risk mix by {'department' if lecturer_id is None else 'course'}"))
    with r:
        show(charts.engagement_scatter(view.sample(min(len(view), 1500), random_state=1)))

    l, r = st.columns(2)
    with l:
        show(charts.score_hist(view))
    with r:
        hist = analytics.historical_outcomes(iid)
        if lecturer_id is not None:
            codes = view["course_code"].unique()
            hist = hist[hist["course_code"].isin(codes)]
            if not hist.empty:
                show(charts.history_outcomes(hist, "course_code", "Last session's actual outcomes by course"))
        elif not hist.empty:
            show(charts.history_outcomes(hist))

    st.markdown("#### Students needing attention now")
    top = analytics.student_summary(high)
    if top.empty:
        st.success("No high-risk students right now.")
    else:
        sel = st.dataframe(
            top.head(15)[["matric_no", "student_name", "programme", "courses", "max_risk", "avg_score",
                          "submission_rate", "weeks_since_active"]],
            hide_index=True, width="stretch", column_config=risk_column_config(),
            on_select="rerun", selection_mode="single-row", key="ov_top")
        st.caption("Select a row to open the student's profile.")
        if sel.selection.rows:
            open_profile(top.head(15).iloc[sel.selection.rows[0]]["student_id"])


# ---------------------------------------------------------------- courses

def courses(user):
    iid, lecturer_id = scope(user)
    header("Courses", "Performance, engagement and risk for each course")
    view = get_view(user)
    if view.empty:
        no_data(user)
        return

    summary = view.groupby(["course_code", "course_title", "lecturer_name"], dropna=False).agg(
        enrolled=("enrollment_id", "count"), avg_score=("avg_score", "mean"),
        submission_rate=("submission_rate", "mean"), avg_clicks=("total_clicks", "mean"),
        high_risk=("risk_level", lambda s: (s == "High").sum()),
        avg_risk=("risk_score", "mean")).reset_index()
    summary["high_risk_share"] = summary["high_risk"] / summary["enrolled"]
    summary = summary.sort_values("high_risk_share", ascending=False)
    st.dataframe(summary, hide_index=True, width="stretch", column_config={
        "course_code": "Code", "course_title": "Title", "lecturer_name": "Lecturer",
        "enrolled": "Enrolled", "avg_score": st.column_config.NumberColumn("Avg score", format="%.1f"),
        "submission_rate": st.column_config.ProgressColumn("Submission", format="percent", min_value=0, max_value=1),
        "avg_clicks": st.column_config.NumberColumn("Avg clicks", format="%d"),
        "high_risk": "High risk",
        "high_risk_share": st.column_config.ProgressColumn("High-risk share", format="percent",
                                                           min_value=0, max_value=1),
        "avg_risk": st.column_config.NumberColumn("Avg risk", format="percent")})

    st.divider()
    code = st.selectbox("Course detail", summary["course_code"].tolist(),
                        format_func=lambda c: f"{c} — {summary.set_index('course_code').loc[c, 'course_title']}")
    cv = view[view["course_code"] == code]
    row = summary.set_index("course_code").loc[code]
    c = st.columns(5)
    c[0].metric("Enrolled", int(row["enrolled"]))
    c[1].metric("Avg score", f"{row['avg_score']:.1f}%")
    c[2].metric("Submission rate", pct(row["submission_rate"]))
    c[3].metric("High risk", int(row["high_risk"]))
    c[4].metric("Lecturer", row["lecturer_name"] if pd.notna(row["lecturer_name"]) else "Unassigned")

    t1, t2, t3 = st.tabs(["Engagement", "Assessments", "Students"])
    with t1:
        weekly = analytics.weekly_engagement(iid, cv["enrollment_id"].tolist())
        weekly["clicks"] = weekly["clicks"] / len(cv)
        allw = analytics.weekly_engagement(iid, view["enrollment_id"].tolist())
        allw["clicks"] = allw["clicks"] / len(view)
        l, r = st.columns(2)
        with l:
            show(charts.weekly_line(weekly, allw, name=code, compare_name="All courses in scope",
                                    title="Average clicks per student per week"))
        with r:
            show(charts.risk_bar(cv, f"{code} risk levels"))
    with t2:
        grades = db.get_grades_df(iid, cv["enrollment_id"].tolist())
        if grades.empty:
            empty_state("No assessments graded yet.")
        else:
            a = grades.groupby(["title", "due_week", "type"]).agg(
                avg=("score", lambda s: s.fillna(0).mean()),
                submitted=("submitted", "mean"), late=("days_late", lambda s: (s > 0).mean())
            ).reset_index().sort_values("due_week")
            fig = charts.px.bar(a, x="title", y="avg", text=a["avg"].round(1), title="Average score per assessment",
                                color_discrete_sequence=[charts.SERIES[0]])
            fig.update_layout(yaxis=dict(range=[0, 100], title="Average score"), xaxis_title=None, height=320)
            show(fig)
            st.dataframe(a, hide_index=True, width="stretch", column_config={
                "title": "Assessment", "due_week": "Due week", "type": "Type",
                "avg": st.column_config.NumberColumn("Avg score", format="%.1f"),
                "submitted": st.column_config.ProgressColumn("Submitted", format="percent", min_value=0, max_value=1),
                "late": st.column_config.ProgressColumn("Late", format="percent", min_value=0, max_value=1)})
    with t3:
        student_table(cv, key=f"course_{code}")


# ---------------------------------------------------------------- students directory

def student_table(view, key):
    s = analytics.student_summary(view)
    cols = ["matric_no", "student_name", "programme", "courses", "risk_level", "max_risk",
            "avg_score", "submission_rate", "total_clicks", "weeks_since_active"]
    sel = st.dataframe(s[cols], hide_index=True, width="stretch", height=430,
                       column_config={**risk_column_config(), "matric_no": "Matric no.",
                                      "student_name": "Student", "programme": "Programme",
                                      "courses": "Courses", "risk_level": "Risk level"},
                       on_select="rerun", selection_mode="single-row", key=key)
    st.caption(f"{len(s):,} students · select a row to open the full profile")
    if sel.selection.rows:
        open_profile(s.iloc[sel.selection.rows[0]]["student_id"])


def students(user):
    header("Students", "Search, filter and open any student's full profile")
    view = get_view(user)
    if view.empty:
        no_data(user)
        return
    fv = filters(view, "stu")
    student_table(fv, "stu_table")
    s = analytics.student_summary(fv)
    st.download_button("Download this list (CSV)", s.to_csv(index=False).encode(), "students.csv", "text/csv")


# ---------------------------------------------------------------- early warning

def early_warning(user):
    iid, lecturer_id = scope(user)
    header("Early warning", "Students the model predicts will fail or withdraw, with the reasons why")
    view = get_view(user)
    if view.empty or view["risk_score"].isna().all():
        empty_state("No predictions yet. An administrator can run predictions under Model Performance.")
        return
    at = view[view["risk_level"].isin(["High", "Medium"])]
    c = st.columns(4)
    c[0].metric("High risk (enrollments)", int((at["risk_level"] == "High").sum()))
    c[1].metric("Medium risk", int((at["risk_level"] == "Medium").sum()))
    c[2].metric("Students affected", at["student_id"].nunique())
    c[3].metric("Missing 2+ assessments", int((view["missed_assessments"] >= 2).sum()))

    fv = filters(at, "ew", show_faculty=lecturer_id is None)
    fv = fv.sort_values("risk_score", ascending=False)
    table = fv[["enrollment_id", "student_id", "matric_no", "student_name", "course_code", "risk_level",
                "risk_score", "predicted_result", "avg_score", "weeks_since_active", "top_factors"]]
    sel = st.dataframe(table.drop(columns=["enrollment_id", "student_id"]), hide_index=True,
                       width="stretch", height=420, on_select="rerun", selection_mode="multi-row",
                       key="ew_table", column_config={
                           **risk_column_config(), "matric_no": "Matric no.", "student_name": "Student",
                           "course_code": "Course", "risk_level": "Risk", "predicted_result": "Predicted",
                           "top_factors": st.column_config.TextColumn("Why the model flagged them", width="large")})
    picked = table.iloc[sel.selection.rows] if sel.selection.rows else table.iloc[0:0]
    st.caption(f"{len(fv):,} flagged enrollments · select rows to message them or open a profile")

    l, r = st.columns([1, 1])
    with l:
        if len(picked) == 1 and st.button("Open selected student's profile", type="primary"):
            open_profile(picked.iloc[0]["student_id"])
    with r:
        st.download_button("Download early-warning list (CSV)", fv.to_csv(index=False).encode(),
                           "early_warning.csv", "text/csv")

    with st.expander(f"Message students ({len(picked) or 'select rows above'})", expanded=bool(len(picked))):
        if not len(picked):
            st.caption("Select one or more students in the table to send a message.")
        else:
            with st.form("bulk_msg"):
                c1, c2 = st.columns(2)
                tpl = c1.selectbox("Template", list(notifications.TEMPLATES))
                ch = c2.selectbox("Channel", notifications.CHANNELS)
                if st.form_submit_button(f"Send to {picked['student_id'].nunique()} student(s)", type="primary"):
                    course = ", ".join(sorted(picked["course_code"].unique()))
                    n = notifications.send_bulk(iid, picked["student_id"].unique().tolist(), user["user_id"], ch,
                                                tpl, course, user.get("name") or user["username"])
                    st.success(f"{n} message(s) recorded.")


# ---------------------------------------------------------------- student profile

def student_profile(user):
    iid, lecturer_id = scope(user)
    view = get_view(user)
    if view.empty:
        header("Student profile")
        no_data(user)
        return
    s = analytics.student_summary(view)
    options = s["student_id"].tolist()
    current = st.session_state.get("profile_student_id")
    idx = options.index(current) if current in options else 0
    names = s.set_index("student_id")
    sid = st.selectbox("Student", options, index=idx,
                       format_func=lambda i: f"{names.loc[i, 'student_name']} · {names.loc[i, 'matric_no']}")
    st.session_state.profile_student_id = sid
    stu = db.get_student(iid, sid)
    sv = view[view["student_id"] == sid].sort_values("risk_score", ascending=False)
    worst = sv["risk_level"].iloc[0]

    st.markdown(
        f"<div class='las-header'><h2>{stu['first_name']} {stu['last_name']} &nbsp;{risk_badge(worst)}</h2>"
        f"<p>{stu['matric_no']} · {stu['programme']} · {stu['faculty']} · Level {stu['level']}<br>"
        f"{stu['email']} · {stu['phone']} · {stu['region']}</p></div>", unsafe_allow_html=True)

    c = st.columns(6)
    c[0].metric("Courses this session", len(sv))
    c[1].metric("Avg score", f"{sv['avg_score'].mean():.1f}%")
    c[2].metric("Submission rate", pct(sv["submission_rate"].mean()))
    c[3].metric("LMS logins", int(sv["total_logins"].sum()))
    c[4].metric("Minutes / week", f"{sv['minutes_per_week'].sum():.0f}")
    c[5].metric("Highest risk", pct(sv["risk_score"].max()))

    tabs = st.tabs(["Risk & prediction", "Engagement", "Grades", "Background",
                    "Messages & interventions"])

    with tabs[0]:
        code = st.radio("Course", sv["course_code"].tolist(), horizontal=True, key="prof_course")
        e = sv[sv["course_code"] == code].iloc[0]
        l, m, r = st.columns([1, 1.2, 1.4])
        with l:
            if pd.notna(e["risk_score"]):
                show(charts.gauge(float(e["risk_score"])), key=f"g_{e['enrollment_id']}")
            else:
                empty_state("Not scored yet.")
        with m:
            if ml_service.model_available(iid) and pd.notna(e["risk_score"]):
                probs = ml_service.predict(iid, sv[sv["course_code"] == code])
                p = {c.replace("p_", ""): float(probs.iloc[0][c]) for c in probs.columns if c.startswith("p_")}
                show(charts.prob_bar(p), key=f"p_{e['enrollment_id']}")
        with r:
            st.markdown(f"**Predicted outcome:** {e['predicted_result'] or '—'}  \n"
                        f"**Model confidence:** {pct(e['confidence_score'])}  \n"
                        f"**Model:** {e['model_used'] or '—'} · scored {e['predicted_at'] or '—'}")
            st.markdown("**Why:**")
            factors = (e["top_factors"] or "").split(" | ")
            if factors and factors[0] and factors[0] != "No major risk factors":
                for f in factors:
                    st.markdown(f"<div class='las-factor'>{f}</div>", unsafe_allow_html=True)
            else:
                st.success("No major risk factors identified.")
        st.markdown("##### All courses")
        st.dataframe(sv[["course_code", "course_title", "lecturer_name", "risk_level", "risk_score",
                         "predicted_result", "avg_score", "submission_rate", "total_clicks"]],
                     hide_index=True, width="stretch", column_config={
                         **risk_column_config(), "course_code": "Code", "course_title": "Course",
                         "lecturer_name": "Lecturer", "risk_level": "Risk", "predicted_result": "Predicted"})

    with tabs[1]:
        code2 = st.selectbox("Course", ["All courses"] + sv["course_code"].tolist(), key="prof_eng_course")
        my_ids = sv["enrollment_id"].tolist() if code2 == "All courses" else \
            sv[sv["course_code"] == code2]["enrollment_id"].tolist()
        mine = analytics.weekly_engagement(iid, my_ids)
        peers = view if code2 == "All courses" else view[view["course_code"] == code2]
        peer_w = analytics.weekly_engagement(iid, peers["enrollment_id"].tolist())
        if not mine.empty:
            for col in ["clicks", "logins", "minutes"]:
                mine[col] = mine[col] / len(my_ids)
                peer_w[col] = peer_w[col] / max(len(peers), 1)
            metric = st.radio("Measure", ["clicks", "logins", "minutes"], horizontal=True, key="prof_metric")
            show(charts.weekly_line(mine, peer_w, y=metric, name=stu["first_name"],
                                    compare_name="Class average", title=f"Weekly {metric} vs class average"))
        act = db.get_activity_df(iid, my_ids)
        if not act.empty:
            t = act.groupby("week")[["logins", "clicks", "minutes_active", "resources_viewed",
                                     "forum_posts", "videos_watched"]].sum().reset_index()
            st.dataframe(t, hide_index=True, width="stretch")

    with tabs[2]:
        grades = db.get_grades_df(iid, sv["enrollment_id"].tolist())
        if grades.empty:
            empty_state("No grades yet.")
        else:
            code3 = st.selectbox("Course", sv["course_code"].tolist(), key="prof_grade_course")
            eid = int(sv[sv["course_code"] == code3]["enrollment_id"].iloc[0])
            g = grades[grades["enrollment_id"] == eid].sort_values("due_week")
            course_eids = view[view["course_code"] == code3]["enrollment_id"].tolist()
            cg = db.get_grades_df(iid, course_eids)
            cavg = cg.groupby("title")["score"].apply(lambda x: x.fillna(0).mean()).reset_index(name="avg")
            show(charts.grades_bar(g, cavg, f"{code3} assessment scores vs course average"))
            g = g.assign(status=g.apply(lambda r: "Missed" if not r["submitted"] else
                                        (f"Late ({int(r['days_late'])}d)" if r["days_late"] else "On time"), axis=1))
            st.dataframe(g[["title", "type", "due_week", "weight", "score", "status", "feedback"]],
                         hide_index=True, width="stretch")
        past = db.get_student_history(iid, sid)
        st.markdown("##### Academic history (transcript)")
        if past.empty:
            st.caption("No completed courses on record.")
        else:
            st.dataframe(past, hide_index=True, width="stretch")

    with tabs[3]:
        info = {"Gender": stu["gender"], "Age band": stu["age_band"], "Region": stu["region"],
                "Entry qualification": stu["highest_education"], "Previous attempts": stu["prev_attempts"],
                "Declared disability": stu["disability"], "Programme": stu["programme"],
                "Faculty": stu["faculty"], "Level": stu["level"]}
        st.dataframe(pd.DataFrame(info.items(), columns=["Field", "Value"]).astype(str),
                     hide_index=True, width="stretch")
        if user["role"] == "Administrator":
            with st.expander("Edit contact details"):
                with st.form(f"edit_stu_{sid}"):
                    e1, e2 = st.columns(2)
                    em = e1.text_input("Email", value=stu["email"] or "")
                    ph = e2.text_input("Phone", value=stu["phone"] or "")
                    e3, e4 = st.columns(2)
                    prog = e3.text_input("Programme", value=stu["programme"] or "")
                    reg = e4.text_input("Region", value=stu["region"] or "")
                    if st.form_submit_button("Save"):
                        db.update_student(iid, sid, email=em.strip(), phone=ph.strip(), programme=prog.strip(),
                                          region=reg.strip())
                        db.insert_log(iid, user["user_id"], "STUDENT_UPDATED", stu["matric_no"])
                        st.cache_data.clear()
                        st.success("Saved.")
                        st.rerun()

    with tabs[4]:
        l, r = st.columns(2)
        with l:
            st.markdown("##### Send a message")
            tpl = st.selectbox("Start from a template", ["(blank)"] + list(notifications.TEMPLATES), key="prof_tpl")
            subj, body = "", ""
            if tpl != "(blank)":
                subj, body = notifications.render_template(tpl, stu, ", ".join(sv["course_code"]),
                                                           user.get("name") or user["username"])
            with st.form("prof_msg", clear_on_submit=True):
                ch = st.selectbox("Channel", notifications.CHANNELS)
                subject = st.text_input("Subject", value=subj)
                message = st.text_area("Message", value=body, height=180)
                if st.form_submit_button("Send", type="primary"):
                    try:
                        _, status = notifications.send(iid, sid, user["user_id"], ch, subject, message)
                        st.success(f"Message {status.lower()}.")
                    except ValueError as exc:
                        st.error(str(exc))
            if not notifications.smtp_configured():
                st.caption("No email gateway configured: messages are recorded for delivery. "
                           "Set SMTP_HOST and related variables to send real email.")
        with r:
            st.markdown("##### Log an intervention")
            with st.form("prof_int", clear_on_submit=True):
                itype = st.selectbox("Type", ["Advising meeting", "Phone call", "Tutoring referral",
                                              "Counselling referral", "Deadline extension", "Other"])
                note = st.text_area("Notes", height=120)
                if st.form_submit_button("Save intervention"):
                    if note.strip():
                        db.insert_intervention(iid, sid, user["user_id"], itype, note.strip())
                        db.insert_log(iid, user["user_id"], "INTERVENTION_LOGGED", f"{itype} for student_id {sid}")
                        st.success("Intervention saved.")
                    else:
                        st.error("Add a short note.")
        st.markdown("##### History")
        h1, h2 = st.columns(2)
        with h1:
            n = db.get_notifications(iid, student_id=sid)
            st.dataframe(n[["sent_at", "channel", "subject", "status", "sender_name"]] if not n.empty else n,
                         hide_index=True, width="stretch")
        with h2:
            iv = db.get_interventions(iid, student_id=sid)
            if iv.empty:
                st.caption("No interventions logged.")
            else:
                for _, row in iv.iterrows():
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(f"**{row['type']}** · {row['created_at']} · {row['created_by_name']}  \n{row['note']}")
                    new = c2.selectbox("Status", ["Open", "In progress", "Resolved"],
                                       index=["Open", "In progress", "Resolved"].index(row["status"])
                                       if row["status"] in ["Open", "In progress", "Resolved"] else 0,
                                       key=f"iv_{row['intervention_id']}", label_visibility="collapsed")
                    if new != row["status"]:
                        db.update_intervention_status(iid, int(row["intervention_id"]), new)
                        st.rerun()


# ---------------------------------------------------------------- communications

def communications(user):
    iid, lecturer_id = scope(user)
    header("Communications", "Every message and intervention, for follow-up and accountability")
    sender = None if lecturer_id is None else user["user_id"]
    t1, t2 = st.tabs(["Messages", "Interventions"])
    with t1:
        n = db.get_notifications(iid, sender_id=sender)
        if n.empty:
            empty_state("No messages sent yet.")
        else:
            c = st.columns(3)
            c[0].metric("Messages", len(n))
            c[1].metric("Students reached", n["student_id"].nunique())
            c[2].metric("Channels used", n["channel"].nunique())
            st.dataframe(n[["sent_at", "student_name", "matric_no", "channel", "subject", "status", "sender_name"]],
                         hide_index=True, width="stretch")
    with t2:
        iv = db.get_interventions(iid, created_by=sender)
        if iv.empty:
            empty_state("No interventions logged yet.")
        else:
            c = st.columns(3)
            c[0].metric("Interventions", len(iv))
            c[1].metric("Open", int((iv["status"] == "Open").sum()))
            c[2].metric("Resolved", int((iv["status"] == "Resolved").sum()))
            st.dataframe(iv[["created_at", "student_name", "matric_no", "type", "status", "note", "created_by_name"]],
                         hide_index=True, width="stretch")


# ---------------------------------------------------------------- reports

def reports(user):
    iid, lecturer_id = scope(user)
    header("Reports", "Download ready-made reports for meetings, senate and record keeping")
    view = get_view(user)
    if view.empty:
        no_data(user)
        return
    s = analytics.student_summary(view)
    course = view.groupby(["course_code", "course_title"]).agg(
        enrolled=("enrollment_id", "count"), avg_score=("avg_score", "mean"),
        submission_rate=("submission_rate", "mean"),
        high_risk=("risk_level", lambda x: (x == "High").sum()),
        medium_risk=("risk_level", lambda x: (x == "Medium").sum())).reset_index().round(3)
    at_risk = view[view["risk_level"] == "High"][["matric_no", "student_name", "email", "phone", "course_code",
                                                  "risk_score", "predicted_result", "top_factors"]]
    items = [
        ("Course performance summary", "One row per course: enrolment, scores, submissions and risk counts.",
         course, "course_summary.csv"),
        ("High-risk student list", "Contact details and reasons for every high-risk enrollment.",
         at_risk, "high_risk_students.csv"),
        ("Full student register", "Every student in scope with risk, scores and engagement.",
         s, "student_register.csv"),
        ("Enrollment-level detail", "Every enrollment with all engagement features and predictions.",
         view.drop(columns=["top_factors"]), "enrollment_detail.csv"),
    ]
    for title, desc, df, fname in items:
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**{title}** — {desc}  \n<span class='las-muted'>{len(df):,} rows</span>",
                    unsafe_allow_html=True)
        c2.download_button("Download CSV", df.to_csv(index=False).encode(), fname, "text/csv", key=fname)
        st.divider()

    st.markdown("#### Executive summary")
    high = (view["risk_level"] == "High").sum()
    worst = course.sort_values("high_risk", ascending=False).head(3)
    text = (f"{user.get('institution_name', '')} — session {analytics.current_session(iid)}, "
            f"week {analytics.current_week(iid)}: "
            f"{view['student_id'].nunique():,} students across {view['course_code'].nunique()} courses. "
            f"Average assessment score is {view['avg_score'].mean():.1f}% with a "
            f"{view['submission_rate'].mean():.0%} submission rate. {high:,} enrollments "
            f"({high / len(view):.0%}) are high risk. Courses with the most high-risk enrollments: "
            + ", ".join(f"{r.course_code} ({r.high_risk})" for r in worst.itertuples()) + ".")
    card(text)
    st.download_button("Download summary (TXT)", text.encode(), "executive_summary.txt", "text/plain")


# ---------------------------------------------------------------- account

def account(user):
    from business.auth import change_password
    header("My account", f"{user.get('name') or user['username']} · {user['role']}")
    with st.form("pw"):
        old = st.text_input("Current password", type="password")
        new = st.text_input("New password", type="password")
        new2 = st.text_input("Confirm new password", type="password")
        if st.form_submit_button("Change password", type="primary"):
            if new != new2:
                st.error("New passwords do not match.")
            else:
                try:
                    change_password(user["user_id"], old, new)
                    st.success("Password changed.")
                except ValueError as exc:
                    st.error(str(exc))
