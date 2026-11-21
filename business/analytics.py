"""
business/analytics.py
Business Layer — builds the analytics views the dashboards display and runs batch
scoring. Every function is scoped to one institution (iid).
"""

import pandas as pd

from database import db
from ml import ml_service
from ml.features import FEATURES, build_features

RESULT_ORDER = ["Distinction", "Pass", "Fail", "Withdrawn"]
RISK_ORDER = ["High", "Medium", "Low"]


def current_session(iid):
    return db.current_session(iid)


def current_week(iid):
    sess = current_session(iid)
    return int(db.current_week(iid, sess) or 1) if sess else 1


def feature_cutoff(iid):
    if ml_service.model_available(iid):
        return int(ml_service.get_bundle(iid)["cutoff_week"])
    return current_week(iid)


def enrollment_view(iid, lecturer_id=None, session=None):
    """One row per current-session enrollment: student, course, latest prediction, engagement features."""
    base = db.get_enrollment_base_df(iid, lecturer_id)
    if base.empty:
        return base
    session = session or current_session(iid)
    base = base[base["session"] == session]
    if base.empty:
        return base
    ids = base["enrollment_id"].tolist()
    df = build_features(base, db.get_activity_df(iid, ids), db.get_grades_df(iid, ids), feature_cutoff(iid))
    df["student_name"] = df["first_name"].fillna("") + " " + df["last_name"].fillna("")
    df["risk_level"] = df["risk_level"].fillna("Not scored")
    for c in ["risk_score", "confidence_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def student_summary(view):
    """One row per student (their worst risk across courses)."""
    if view.empty:
        return view
    out = view.groupby("student_id").agg(
        matric_no=("matric_no", "first"), student_name=("student_name", "first"),
        programme=("programme", "first"), faculty=("faculty", "first"), gender=("gender", "first"),
        email=("email", "first"), phone=("phone", "first"), courses=("course_code", "nunique"),
        max_risk=("risk_score", "max"), avg_risk=("risk_score", "mean"),
        avg_score=("avg_score", "mean"), total_clicks=("total_clicks", "sum"),
        submission_rate=("submission_rate", "mean"), weeks_since_active=("weeks_since_active", "min"),
    ).reset_index()
    out["risk_level"] = out["max_risk"].apply(lambda r: "Not scored" if pd.isna(r) else ml_service.risk_level(r))
    return out.sort_values("max_risk", ascending=False, na_position="last")


def weekly_engagement(iid, enrollment_ids):
    act = db.get_activity_df(iid, enrollment_ids)
    if act.empty:
        return pd.DataFrame(columns=["week", "logins", "clicks", "minutes", "active_students"])
    return act.groupby("week").agg(logins=("logins", "sum"), clicks=("clicks", "sum"),
                                   minutes=("minutes_active", "sum"),
                                   active_students=("logins", lambda s: (s > 0).sum())).reset_index()


def historical_outcomes(iid):
    return db.historical_outcomes(iid)


def score_current(iid, user_id=None):
    """Predict every in-progress enrollment of the institution and store the results."""
    if not ml_service.model_available(iid):
        raise FileNotFoundError("No trained model yet. Train a model first.")
    base = db.get_enrollment_base_df(iid)
    base = base[base["final_result"].isna()]
    if base.empty:
        return 0
    ids = base["enrollment_id"].tolist()
    feats = build_features(base, db.get_activity_df(iid, ids), db.get_grades_df(iid, ids), feature_cutoff(iid))
    preds = ml_service.predict(iid, feats[FEATURES])
    rows = [(int(eid), p.predicted_result, float(p.risk_score), p.risk_level, float(p.confidence_score),
             p.top_factors, p.model_used)
            for eid, (_, p) in zip(feats["enrollment_id"], preds.iterrows())]
    db.insert_predictions(iid, rows)
    db.insert_log(iid, user_id, "PREDICTIONS_RUN", f"Scored {len(rows)} active enrollments")
    return len(rows)
