"""
presentation/admin_pages.py
Administrator-only pages (always limited to the administrator's own institution):
data management & set-up wizard, model performance, users, audit log.
Plus the platform-owner page (counts per institution, no student data).
"""

import pandas as pd
import streamlit as st

from business import analytics, importer, onboarding
from business import formats
from business.auth import (ROLES, add_user, delete_user, reset_password, set_active, transfer_owner,
                           update_settings)
from database import db
from ml import ml_service, train
from ml.features import FEATURE_LABELS
from presentation import charts
from presentation.pages import empty_state, show
from presentation.theme import header, pct
from scripts.generate_institution_data import DEMO_LECTURER_PASSWORD


def _progress_runner(label):
    bar = st.progress(0.0, text=label)

    def update(fraction, text):
        bar.progress(min(max(fraction, 0.0), 1.0), text=text)
    return update


def _fmt(v, f="{:.3f}"):
    return f.format(v) if isinstance(v, (int, float)) else "—"


# ---------------------------------------------------------------- dataset requirements (shared)

def dataset_requirements(compact=False, inst=None):
    st.markdown("Your file must be a **CSV with one row per student per course**. Include **past students** "
                "(with `final_result`) so the system can learn your institution's patterns, and **current "
                "students** (blank `final_result`) so it can predict for them.")
    t = importer.columns_table()
    st.dataframe(t if not compact else t[t["Required"] == "Yes"], hide_index=True, width="stretch",
                 height=None if compact else 560)
    st.markdown(
        "**Rules that matter**\n"
        "- Count past students' activity **up to the same week of the semester** as your current students "
        "(e.g. their first 8 weeks), not their whole-semester totals.\n"
        f"- Have at least **{train.MIN_ROWS} past enrollments**, with at least **{train.MIN_PER_CLASS}** of each result. "
        "For reliable predictions aim for **500+ past enrollments** and **50+ Fail/Withdrawn**.\n"
        "- Map your own grading to the four results, e.g. A = Distinction, B–D = Pass, E–F = Fail, "
        "stopped attending = Withdrawn.")
    if inst and (inst.get("matric_format") or inst.get("student_email_format")):
        st.info(f"Your institution's formats: matric numbers must match **{inst.get('matric_format') or 'any'}**"
                + (f"; blank student emails are filled in as **{inst['student_email_format']}**"
                   if inst.get("student_email_format") else "") + ". Change them under Settings.")
    c1, c2, _ = st.columns([1, 1, 2])
    c1.download_button("Download blank template", importer.template_csv(), "las_upload_template.csv", "text/csv",
                       key=f"tpl_{compact}")
    c2.download_button("Download a sample file", importer.sample_csv(inst=inst), "las_sample_institution.csv", "text/csv",
                       key=f"smp_{compact}", help="600 rows: 400 past and 200 current students, ready to upload")


# ---------------------------------------------------------------- data management

def data_management(user):
    iid, uid = user["institution_id"], user["user_id"]
    inst = db.get_institution(iid)
    mode = {"demo": "Demo data (for exploring)", "own": f"{inst['name']} data", "empty": "Empty"}[inst["data_mode"]]
    header("Data management", f"Currently loaded: {mode}")
    counts = db.table_counts(iid)
    c = st.columns(5)
    for i, k in enumerate(["Students", "Courses", "Enrollments", "Grades", "Predictions"]):
        c[i].metric(k, f"{counts[k]:,}")

    tabs = st.tabs(["Load your institution's data", "Add more data", "Dataset requirements",
                    "Demo & reset", "History & export"])

    # ---------- the set-up wizard
    with tabs[0]:
        stage = st.session_state.get("wiz_stage", "upload")
        if inst["data_mode"] == "demo":
            st.warning("You are viewing **demo data**. Upload your institution's file below to replace it with "
                       "your own students. The system will reset, load your data, train a model on your history "
                       "and rebuild every dashboard.", icon="🧪")
        st.markdown("**Step 1 — Upload your file** · Step 2 — Confirm reset · Step 3 — Load, train and predict")
        week = st.number_input("Your data covers weeks 1 to ... of the semester", 3, 14, 8,
                               help="The week your current students are at. Past students' numbers must "
                                    "cover the same weeks.", key="wiz_week")
        up = st.file_uploader("Institution dataset (CSV)", type="csv", key="wiz_file")
        if up is None:
            st.session_state.wiz_stage = "upload"
            st.caption("Not sure what to include? See the **Dataset requirements** tab.")
        else:
            try:
                raw = pd.read_csv(up)
            except Exception as exc:
                st.error(f"Could not read the file: {exc}")
                return
            errors, clean = importer.validate(raw, inst)
            st.dataframe(raw.head(8), hide_index=True, width="stretch")
            if errors:
                for e in errors:
                    st.error(e)
                st.info("Fix these in your file and upload it again.")
                return
            summ = importer.summarize(clean)
            ok, problems = train.readiness(clean.loc[clean["final_result"] != "", "final_result"])
            k = st.columns(5)
            k[0].metric("Rows", f"{summ['rows']:,}")
            k[1].metric("Students", f"{summ['students']:,}")
            k[2].metric("Courses", summ["courses"])
            k[3].metric("Past (to learn from)", f"{summ['past']:,}")
            k[4].metric("Current (to predict)", f"{summ['current']:,}")
            if summ["results"]:
                st.caption("Past results: " + ", ".join(f"{k_}: {v}" for k_, v in summ["results"].items()))
            for p in problems + clean.attrs.get("warnings", []):
                st.warning(p)
            if not summ["current"]:
                st.warning("No current students (rows with blank final_result): there will be nothing to predict.")
            st.success("The file is valid." + ("" if ok else " Training may not be possible — see warnings."))

            if stage == "upload":
                if st.button("Continue — use this file", type="primary"):
                    st.session_state.wiz_stage = "confirm"
                    st.rerun()
            elif stage == "confirm":
                st.error(f"**Reset the system?** This permanently deletes ALL current students, courses, grades, "
                         f"predictions, messages, interventions and models for **{inst['name']}**. "
                         f"User accounts are kept. Demo lecturer accounts are removed.", icon="⚠️")
                code = st.text_input(f"Type your institution code **{inst['code']}** to confirm", key="wiz_code")
                b1, b2 = st.columns([1, 3])
                if b1.button("Yes, reset", type="primary", disabled=code.strip().upper() != inst["code"]):
                    onboarding.reset(iid, uid)
                    st.session_state.wiz_stage = "load"
                    st.cache_data.clear()
                    st.rerun()
                if b2.button("Cancel"):
                    st.session_state.wiz_stage = "upload"
                    st.rerun()
            elif stage == "load":
                st.success("The system has been reset. It is now empty.", icon="✅")
                st.markdown(f"**Load *{up.name}* now?** The system will import it, train the three models on "
                            f"your {summ['past']:,} past enrollments and predict your {summ['current']:,} current ones.")
                b1, b2 = st.columns([1, 3])
                if b1.button("Yes, load my data", type="primary"):
                    update = _progress_runner("Starting...")
                    out = onboarding.load_own_data(iid, clean, up.name, int(week), uid, progress=update)
                    st.cache_data.clear()
                    st.session_state.wiz_stage = "done"
                    st.session_state.wiz_result = out
                    st.rerun()
                if b2.button("Not now (leave the system empty)"):
                    st.session_state.wiz_stage = "upload"
                    st.rerun()
            elif stage == "done":
                out = st.session_state.get("wiz_result", {})
                imp = out.get("import", {})
                st.success(f"Loaded {imp.get('students', 0):,} students and {imp.get('enrollments', 0):,} "
                           f"enrollments.", icon="✅")
                if out.get("report"):
                    r = out["report"]
                    best = r["results"][r["best_model"]]
                    st.success(f"Trained on your history — best model: **{r['best_model']}** "
                               f"(accuracy {pct(best['accuracy'])}, AUC {_fmt(best['auc_roc'], '{:.2f}')}). "
                               f"Predicted {out['scored']:,} current enrollments. Every dashboard now shows "
                               f"your institution.", icon="🎉")
                if out.get("train_error"):
                    st.warning(f"Data loaded, but the model could not be trained: {out['train_error']}")
                if st.button("Go to the overview", type="primary"):
                    st.session_state.wiz_stage = "upload"
                    st.session_state.nav_target = "Overview"
                    st.rerun()

    # ---------- append
    with tabs[1]:
        st.markdown("Add new students or a new session **without** deleting what is already loaded. "
                    "Rows already in the system (same matric number, course and session) are skipped.")
        week2 = st.number_input("Data covers weeks 1 to ...", 3, 14, max(3, min(14, analytics.feature_cutoff(iid))),
                                key="add_week")
        up2 = st.file_uploader("CSV file", type="csv", key="add_file")
        retrain = st.checkbox("Retrain the model afterwards (do this when the file contains past results)")
        if up2 is not None:
            errors, clean2 = importer.validate(pd.read_csv(up2), inst)
            for w in clean2.attrs.get("warnings", []):
                st.warning(w)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                s2 = importer.summarize(clean2)
                st.info(f"{s2['rows']:,} rows · {s2['past']:,} past · {s2['current']:,} current")
                if st.button("Add this data", type="primary"):
                    update = _progress_runner("Importing...")
                    out = onboarding.add_data(iid, clean2, up2.name, int(week2), uid, retrain, update)
                    st.cache_data.clear()
                    imp = out["import"]
                    st.success(f"Added {imp['students']:,} students, {imp['enrollments']:,} enrollments "
                               f"({imp['skipped']:,} skipped). Scored {out['scored']:,}.")
                    if out["train_error"]:
                        st.warning(out["train_error"])

    with tabs[2]:
        dataset_requirements(inst=inst)

    # ---------- demo / reset
    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Load demo data")
            st.caption("Fills the system with 400 simulated students, 26 courses and 8 demo lecturers "
                       f"(password `{DEMO_LECTURER_PASSWORD}`), trains a "
                       "model and predicts. Replaces anything currently loaded.")
            ok_demo = st.checkbox("I understand this replaces the current data", key="demo_ok")
            if st.button("Load demo data", disabled=not ok_demo):
                update = _progress_runner("Loading demo...")
                onboarding.load_demo(iid, uid, progress=update)
                st.cache_data.clear()
                st.rerun()
        with c2:
            st.markdown("##### Reset to empty")
            st.caption("Deletes all students, courses, grades, predictions, messages and models. "
                       "Accounts are kept.")
            code = st.text_input(f"Type {inst['code']} to confirm", key="reset_code")
            if st.button("Reset system", type="primary", disabled=code.strip().upper() != inst["code"]):
                onboarding.reset(iid, uid)
                st.cache_data.clear()
                st.success("System reset.")
                st.rerun()

    with tabs[4]:
        st.dataframe(db.get_datasets(iid), hide_index=True, width="stretch")
        for label, sql in [
            ("Students", "SELECT * FROM Students WHERE institution_id = ?"),
            ("Courses", "SELECT * FROM Courses WHERE institution_id = ?"),
            ("Predictions", "SELECT * FROM Predictions WHERE institution_id = ?"),
            ("Notifications", "SELECT * FROM Notifications WHERE institution_id = ?"),
            ("Interventions", "SELECT * FROM Interventions WHERE institution_id = ?")]:
            df = db.query_df(sql, (iid,))
            st.download_button(f"Export {label} ({len(df):,} rows)", df.to_csv(index=False).encode(),
                               f"{label.lower()}.csv", "text/csv", key=f"exp_{label}")


# ---------------------------------------------------------------- model performance

def model_performance(user):
    iid, uid = user["institution_id"], user["user_id"]
    header("Model performance", "Logistic Regression vs Decision Tree vs Random Forest — trained on your past "
                                "students, applied to your current ones")
    if not ml_service.model_available(iid):
        empty_state("No trained model yet. Load data under Data Management, or click Retrain below.")
    else:
        rep = ml_service.get_report(iid)
        best = rep["results"][rep["best_model"]]
        c = st.columns(3)
        c[0].metric("Best model", rep["best_model"])
        c[1].metric("Accuracy", pct(best["accuracy"], 1))
        c[2].metric("Weighted F1", _fmt(best["f1_score"]))
        c = st.columns(3)
        c[0].metric("AUC-ROC", _fmt(best["auc_roc"]))
        c[1].metric("At-risk recall", pct(best["at_risk_recall"]),
                    help="Share of students who actually failed/withdrew that the model flagged")
        c[2].metric("Training rows", f"{rep['trained_rows']:,}")
        st.caption(f"Trained {rep['trained_at']}. Features use only the first {rep['cutoff_week']} weeks "
                   "of each course, so predictions are available mid-semester.")
        if rep["trained_rows"] < 500:
            st.warning("Fewer than 500 past enrollments: treat predictions as indicative and add more history.")

        l, r = st.columns(2)
        with l:
            show(charts.metric_compare(rep["results"]))
        with r:
            show(charts.roc(rep["results"]))
        l, r = st.columns(2)
        with l:
            name = st.selectbox("Confusion matrix for", list(rep["results"]),
                                index=list(rep["results"]).index(rep["best_model"]))
            show(charts.confusion(rep["results"][name]["confusion_matrix"], rep["labels"],
                                  f"Confusion matrix — {name}"))
        with r:
            show(charts.importance_bar(rep["feature_importance"], FEATURE_LABELS))
        table = pd.DataFrame([{"Model": k, "Accuracy": v["accuracy"], "Precision": v["precision"],
                               "Recall": v["recall"], "F1": v["f1_score"], "AUC-ROC": v["auc_roc"],
                               "CV F1": v["cv_f1"], "Best parameters": str(v["best_params"])}
                              for k, v in rep["results"].items()])
        st.dataframe(table, hide_index=True, width="stretch")
        with st.expander("Training history"):
            st.dataframe(db.get_model_history(iid), hide_index=True, width="stretch")

    st.divider()
    st.markdown("#### Retrain and re-score")
    c1, c2, c3 = st.columns([1.2, 1, 1])
    week = c1.slider("Prediction week (features from weeks 1..N)", 3, 14,
                     max(3, min(14, analytics.current_week(iid))))
    if c2.button("Retrain models", type="primary", width="stretch"):
        update = _progress_runner("Training...")
        try:
            train.train(iid, cutoff_week=week, verbose=False, user_id=uid, progress=update)
            n = analytics.score_current(iid, uid)
            st.cache_data.clear()
            st.success(f"Retrained and re-scored {n:,} enrollments.")
            st.rerun()
        except Exception as exc:
            st.error(f"Training failed: {exc}")
    if c3.button("Re-score current students", width="stretch"):
        try:
            n = analytics.score_current(iid, uid)
            st.cache_data.clear()
            st.success(f"Scored {n:,} enrollments.")
            st.rerun()
        except FileNotFoundError as exc:
            st.error(str(exc))


# ---------------------------------------------------------------- users

def users(user):
    iid, uid = user["institution_id"], user["user_id"]
    header("Users & course allocation", "Give lecturers and other administrators access to your institution")
    t1, t2 = st.tabs(["Accounts", "Course allocation"])
    with t1:
        with st.expander("Add a person", expanded=True):
            st.caption("Lecturers sign in with this email and password and see only the courses you assign to them.")
            with st.form("new_user", clear_on_submit=True):
                c1, c2, c3 = st.columns(3)
                name = c1.text_input("Full name")
                email = c2.text_input("Email (used to sign in)")
                role = c3.selectbox("Role", ["Lecturer", "Administrator"])
                c4, c5 = st.columns(2)
                dept = c4.text_input("Department")
                pw = c5.text_input("Temporary password (8+ characters, letters and numbers)", type="password")
                if st.form_submit_button("Create account", type="primary"):
                    try:
                        add_user(iid, uid, email, pw, role, name, dept)
                        st.success(f"Created {role} account for {email}. Share the email and temporary password "
                                   "with them; they can change it under My Account.")
                    except ValueError as exc:
                        st.error(str(exc))
        all_users = db.get_all_users(iid)
        for col in ["is_active", "is_owner", "is_demo"]:
            all_users[col] = all_users[col].astype(bool)
        st.dataframe(all_users, hide_index=True, width="stretch", column_config={
            "is_active": st.column_config.CheckboxColumn("Active"),
            "is_owner": st.column_config.CheckboxColumn("Master admin"),
            "is_demo": st.column_config.CheckboxColumn("Demo")})

        st.markdown("##### Manage an account")
        others = all_users[all_users["user_id"] != uid]
        if others.empty:
            st.caption("No other accounts yet.")
        else:
            idx = others.set_index("user_id")
            target = st.selectbox("Account", others["user_id"].tolist(),
                                  format_func=lambda i: f"{idx.loc[i, 'name']} ({idx.loc[i, 'email']}) — "
                                                        f"{idx.loc[i, 'role']}"
                                                        + (" · master admin" if idx.loc[i, "is_owner"] else "")
                                                        + ("" if idx.loc[i, "is_active"] else " · inactive"))
            t_row = idx.loc[target]
            a1, a2, a3, a4 = st.tabs(["Edit details", "Access", "Password", "Delete"])
            with a1:
                with st.form(f"edit_{target}"):
                    e1, e2, e3 = st.columns(3)
                    new_name = e1.text_input("Full name", value=t_row["name"] or "")
                    new_dept = e2.text_input("Department", value=t_row["department"] or "")
                    new_role = e3.selectbox("Role", ["Lecturer", "Administrator"],
                                            index=0 if t_row["role"] == "Lecturer" else 1,
                                            disabled=bool(t_row["is_owner"]))
                    if st.form_submit_button("Save changes"):
                        db.update_user_details(iid, int(target), new_name.strip(), new_dept.strip(),
                                               None if t_row["is_owner"] else new_role)
                        db.insert_log(iid, uid, "USER_UPDATED", t_row["email"])
                        st.success("Saved.")
                        st.rerun()
            with a2:
                if t_row["is_owner"]:
                    st.caption("The master administrator cannot be deactivated.")
                else:
                    active = bool(t_row["is_active"])
                    st.write("Deactivated accounts cannot sign in, but keep their history and course assignments.")
                    if st.button("Deactivate account" if active else "Reactivate account"):
                        set_active(iid, uid, int(target), not active)
                        st.rerun()
            with a3:
                npw = st.text_input("New temporary password", type="password", key=f"rpw_{target}")
                if st.button("Reset password", key=f"rbtn_{target}"):
                    try:
                        reset_password(iid, uid, int(target), npw)
                        st.success("Password reset. Share it with the person securely.")
                    except ValueError as exc:
                        st.error(str(exc))
            with a4:
                if t_row["is_owner"]:
                    st.caption("The master administrator cannot be deleted. Transfer the role first (below).")
                else:
                    n_courses = int((db.get_courses(iid)["lecturer_id"] == target).sum())
                    st.warning(f"Permanently deletes **{t_row['name']}**'s account (e.g. a staff member who has "
                               f"left). Their {n_courses} course(s) become unassigned. Messages and interventions "
                               "they recorded are kept, without their name. This cannot be undone.")
                    confirm = st.text_input("Type their email to confirm", key=f"del_{target}")
                    if st.button("Delete account permanently", type="primary",
                                 disabled=confirm.strip().lower() != t_row["email"]):
                        try:
                            delete_user(iid, uid, int(target))
                            st.cache_data.clear()
                            st.success("Account deleted.")
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))

        if user.get("is_owner"):
            admins = all_users[(all_users["role"] == "Administrator") & (all_users["user_id"] != uid)
                               & all_users["is_active"]]
            with st.expander("Transfer master administrator role"):
                if admins.empty:
                    st.caption("Add another active Administrator first.")
                else:
                    aidx = admins.set_index("user_id")
                    new_owner = st.selectbox("New master administrator", admins["user_id"].tolist(),
                                             format_func=lambda i: f"{aidx.loc[i, 'name']} ({aidx.loc[i, 'email']})")
                    if st.checkbox("I understand I will become a normal administrator"):
                        if st.button("Transfer role"):
                            transfer_owner(iid, uid, int(new_owner))
                            st.success("Role transferred.")
                            st.rerun()
    with t2:
        courses = db.get_courses(iid)
        lecturers = db.get_lecturers(iid)
        if courses.empty:
            empty_state("No courses yet — load data first.")
            return
        cidx = courses.set_index("course_id")
        fmt_course = lambda i: f"{cidx.loc[i, 'code']} — {cidx.loc[i, 'title']}"
        st.dataframe(courses[["code", "title", "department", "lecturer_name", "enrolled"]], hide_index=True,
                     width="stretch", column_config={"lecturer_name": "Lecturer"})
        unassigned = int(courses["lecturer_id"].isna().sum())
        if unassigned:
            st.caption(f"{unassigned} course(s) have no lecturer.")
        if lecturers.empty:
            empty_state("Create a Lecturer account first (Accounts tab).")
            return
        lidx = lecturers.set_index("user_id")
        fmt_lec = lambda i: lidx.loc[i, "name"] or lidx.loc[i, "username"]
        b1, b2 = st.columns(2)
        with b1:
            st.markdown("##### Assign courses to a lecturer")
            lid = st.selectbox("Lecturer", lecturers["user_id"].tolist(), format_func=fmt_lec, key="as_lec")
            picked = st.multiselect("Courses", courses["course_id"].tolist(), format_func=fmt_course, key="as_crs")
            if st.button("Assign", type="primary", disabled=not picked):
                for cid in picked:
                    db.assign_course_lecturer(iid, int(cid), int(lid))
                db.insert_log(iid, uid, "COURSE_ASSIGNED", f"{len(picked)} course(s) -> {lidx.loc[lid, 'username']}")
                st.cache_data.clear()
                st.success("Assigned. A course has one lecturer, so this replaces any previous lecturer.")
                st.rerun()
        with b2:
            st.markdown("##### Unassign courses")
            lid2 = st.selectbox("Lecturer", lecturers["user_id"].tolist(), format_func=fmt_lec, key="un_lec")
            mine = courses[courses["lecturer_id"] == lid2]
            if mine.empty:
                st.caption("This lecturer has no courses.")
            else:
                drop = st.multiselect("Their courses to remove", mine["course_id"].tolist(),
                                      format_func=fmt_course, key="un_crs")
                if st.button("Unassign", disabled=not drop):
                    db.unassign_courses(iid, [int(c) for c in drop])
                    db.insert_log(iid, uid, "COURSE_UNASSIGNED",
                                  f"{len(drop)} course(s) from {lidx.loc[lid2, 'username']}")
                    st.cache_data.clear()
                    st.success("Unassigned. The lecturer no longer sees these courses or their students.")
                    st.rerun()


# ---------------------------------------------------------------- audit log

def audit_logs(user):
    iid = user["institution_id"]
    header("Audit log", "Who did what, and when — your institution only")
    logs = db.get_logs(iid, 3000)
    if logs.empty:
        empty_state("No activity yet.")
        return
    c1, c2 = st.columns(2)
    act = c1.multiselect("Action", sorted(logs["action"].unique()))
    who = c2.multiselect("User", sorted(logs["user_email"].dropna().unique()))
    if act:
        logs = logs[logs["action"].isin(act)]
    if who:
        logs = logs[logs["user_email"].isin(who)]
    daily = logs.assign(day=logs["logged_at"].astype(str).str[:10]).groupby("day").size().reset_index(name="events")
    fig = charts.px.bar(daily, x="day", y="events", title="Events per day", color_discrete_sequence=[charts.SERIES[0]])
    fig.update_layout(height=260, xaxis_title=None)
    show(fig)
    st.dataframe(logs, hide_index=True, width="stretch", height=420)
    st.download_button("Download log (CSV)", logs.to_csv(index=False).encode(), "audit_log.csv", "text/csv")


# ---------------------------------------------------------------- platform owner

def platform(user):
    header("Platform", "Every registered institution — counts only, no student details")
    df = db.platform_overview()
    c = st.columns(4)
    c[0].metric("Institutions", len(df))
    c[1].metric("Active", int(df["is_active"].astype(int).sum()))
    c[2].metric("Users", int(df["users"].sum()))
    c[3].metric("Students", f"{int(df['students'].sum()):,}")
    df["is_active"] = df["is_active"].astype(bool)
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={"is_active": st.column_config.CheckboxColumn("Active")})
    others = df[df["institution_id"] != user["institution_id"]]
    if not others.empty:
        idx = others.set_index("institution_id")
        c1, c2 = st.columns([3, 1])
        target = c1.selectbox("Institution", others["institution_id"].tolist(),
                              format_func=lambda i: f"{idx.loc[i, 'name']} ({idx.loc[i, 'code']})")
        active = bool(idx.loc[target, "is_active"])
        c2.write("")
        if c2.button("Suspend access" if active else "Restore access", width="stretch"):
            db.set_institution_active(int(target), not active)
            db.insert_log(user["institution_id"], user["user_id"],
                          "INSTITUTION_SUSPENDED" if active else "INSTITUTION_RESTORED", idx.loc[target, "code"])
            st.rerun()


# ---------------------------------------------------------------- institution settings

def settings(user):
    iid, uid = user["institution_id"], user["user_id"]
    inst = db.get_institution(iid)
    header("Institution settings", f"{inst['name']} · code {inst['code']} (fixed)")
    with st.form("inst_settings"):
        st.markdown("##### Institution")
        c1, c2, c3 = st.columns([2, 1, 1])
        name = c1.text_input("Institution name", value=inst["name"])
        types = ["University", "Polytechnic", "College of Education", "Secondary School", "Training Institute", "Other"]
        itype = c2.selectbox("Type", types, index=types.index(inst["type"]) if inst["type"] in types else 0)
        country = c3.text_input("Country", value=inst["country"] or "")
        contact = st.text_input("Contact email", value=inst["contact_email"] or "")
        st.markdown("##### Identifier formats")
        st.caption("Enter one real sample of each and the rules are worked out for you. Leave a field blank if "
                   "your institution has no fixed format.")
        c4, c5, c6 = st.columns(3)
        example = c4.text_input("Sample student matric number", value=inst.get("matric_example") or "")
        s_sample = c5.text_input("Sample student email", value=formats.student_email(
            inst.get("student_email_format"), inst.get("matric_example") or "") if inst.get("matric_example") else "")
        staff_sample = c6.text_input("Sample staff email", value=("staff@" + inst["staff_email_domain"])
                                     if inst.get("staff_email_domain") else "")
        with st.expander("Advanced: edit the rules directly"):
            st.caption("Matric format: # = digit, ? = letter, * = anything, other characters fixed. "
                       "Student email: use {matric}, {first} or {last} before the @. "
                       "Staff domain: the part after @; separate several with commas. "
                       "Anything typed here overrides the samples above.")
            a1, a2, a3 = st.columns(3)
            adv_m = a1.text_input("Matric number format", value=inst.get("matric_format") or "")
            adv_s = a2.text_input("Student email format", value=inst.get("student_email_format") or "")
            adv_d = a3.text_input("Staff email domain(s)", value=inst.get("staff_email_domain") or "")
        if st.form_submit_button("Save settings", type="primary"):
            mfmt, semail, sdom, notes = formats.from_samples(example, s_sample, staff_sample)
            if not semail and s_sample.strip():          # sample could not be linked: keep the current rule
                semail = inst.get("student_email_format") or ""
            # the advanced boxes win when the admin changed them
            if adv_m.strip() != (inst.get("matric_format") or ""):
                mfmt = adv_m.strip()
            if adv_s.strip() != (inst.get("student_email_format") or ""):
                semail = adv_s.strip()
            if adv_d.strip() != (inst.get("staff_email_domain") or ""):
                sdom = adv_d.strip()
            for n in notes:
                st.info(n)
            try:
                update_settings(iid, uid, name=name, type=itype, country=country, contact_email=contact,
                                matric_format=mfmt, matric_example=example, student_email_format=semail,
                                staff_email_domain=sdom)
                st.success("Settings saved.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    st.markdown("##### Test your formats")
    t1, t2 = st.columns(2)
    test = t1.text_input("Try a matric number")
    if test:
        if formats.matric_matches(inst.get("matric_format"), test):
            t1.success(f"Matches {inst.get('matric_format') or '(any format)'}")
        else:
            t1.error(f"Does not match {inst.get('matric_format')}")
        if inst.get("student_email_format"):
            t1.caption(f"Student email would be: {formats.student_email(inst['student_email_format'], test)}")
    test_email = t2.text_input("Try a staff email")
    if test_email:
        if formats.staff_email_ok(inst.get("staff_email_domain"), test_email):
            t2.success("Allowed for staff accounts")
        else:
            t2.error(f"Staff emails must end with {inst.get('staff_email_domain')}")
    current = db.query_df("SELECT matric_no FROM Students WHERE institution_id = ?", (iid,))
    if inst.get("matric_format") and not current.empty:
        bad = (~current["matric_no"].map(lambda m: formats.matric_matches(inst["matric_format"], m))).sum()
        if bad:
            st.warning(f"{bad:,} student(s) already loaded do not match the current format "
                       "(for example demo data loaded before the format was set). New uploads are checked.")
