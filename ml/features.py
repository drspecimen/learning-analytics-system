"""
ml/features.py
Feature Engineering — turns raw LMS activity and grades into one row of
model features per enrollment (Figure 3.5: Feature Engineering).

Features are computed only from weeks 1..cutoff_week, so a model trained on a
completed session sees exactly what it will see for a live session at that week.
This is what makes the predictions an EARLY warning instead of a post-mortem.
"""

import numpy as np
import pandas as pd

# Sensitive attributes (gender, age band, disability) and entry qualification are NOT model inputs.
# They are still computed below so the fairness audit can compare error rates across groups.
FEATURES = [
    "prev_attempts",
    "total_logins", "total_clicks", "active_weeks", "minutes_per_week",
    "resources_viewed", "forum_posts", "videos_watched",
    "engagement_trend", "weeks_since_active", "clicks_vs_course",
    "avg_score", "submission_rate", "late_submissions", "missed_assessments",
]

FEATURE_LABELS = {
    "prev_attempts": "Previous attempts",
    "is_male": "Gender (male)",
    "has_disability": "Declared disability",
    "age_rank": "Age band",
    "education_rank": "Entry qualification",
    "total_logins": "LMS logins",
    "total_clicks": "LMS clicks",
    "active_weeks": "Active weeks",
    "minutes_per_week": "Minutes active per week",
    "resources_viewed": "Resources viewed",
    "forum_posts": "Forum posts",
    "videos_watched": "Videos watched",
    "engagement_trend": "Engagement trend",
    "weeks_since_active": "Weeks inactive",
    "clicks_vs_course": "Clicks vs course average",
    "avg_score": "Average assessment score",
    "submission_rate": "Submission rate",
    "late_submissions": "Late submissions",
    "missed_assessments": "Missed assessments",
}

# +1: a higher value means more risk; -1: a lower value means more risk; 0: not used in explanations
RISK_DIRECTION = {
    "prev_attempts": 1, "late_submissions": 1, "missed_assessments": 1, "weeks_since_active": 1,
    "total_logins": -1, "total_clicks": -1, "active_weeks": -1, "minutes_per_week": -1,
    "resources_viewed": -1, "forum_posts": -1, "videos_watched": -1, "engagement_trend": -1,
    "clicks_vs_course": -1, "avg_score": -1, "submission_rate": -1,
    "is_male": 0, "has_disability": 0, "age_rank": 0, "education_rank": 0,
}

AGE_ORDER = ["16-24", "25-34", "35-44", "45+"]
EDU_ORDER = ["SSCE/WAEC", "OND", "NCE", "HND", "First Degree"]


def _slope(values):
    """Least-squares slope of a weekly series, normalised by its mean."""
    y = np.asarray(values, dtype=float)
    if len(y) < 2 or y.mean() == 0:
        return 0.0
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0] / y.mean())


def build_features(base_df, activity_df, grades_df, cutoff_week):
    """
    base_df:     enrollments with student columns (from db.get_enrollment_base_df)
    activity_df: Weekly_Activity rows
    grades_df:   Grades rows joined with Assessments
    Returns base_df with the FEATURES columns added (one row per enrollment).
    """
    df = base_df.copy()
    act = activity_df[activity_df["week"] <= cutoff_week]
    grd = grades_df[grades_df["due_week"] <= cutoff_week]

    # ---- engagement aggregates
    agg = act.groupby("enrollment_id").agg(
        total_logins=("logins", "sum"),
        total_clicks=("clicks", "sum"),
        minutes_total=("minutes_active", "sum"),
        resources_viewed=("resources_viewed", "sum"),
        forum_posts=("forum_posts", "sum"),
        videos_watched=("videos_watched", "sum"),
    )
    active = act[act["logins"] > 0].groupby("enrollment_id").agg(
        active_weeks=("week", "nunique"), last_week=("week", "max"))

    # full weekly click series (missing weeks = 0) for the trend
    pivot = act.pivot_table(index="enrollment_id", columns="week", values="clicks",
                            aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(columns=range(1, cutoff_week + 1), fill_value=0)
    trend = pivot.apply(lambda r: _slope(r.values), axis=1).rename("engagement_trend")

    df = df.merge(agg, left_on="enrollment_id", right_index=True, how="left")
    df = df.merge(active, left_on="enrollment_id", right_index=True, how="left")
    df = df.merge(trend, left_on="enrollment_id", right_index=True, how="left")

    for c in ["total_logins", "total_clicks", "minutes_total", "resources_viewed",
              "forum_posts", "videos_watched", "active_weeks", "engagement_trend"]:
        df[c] = df[c].fillna(0)
    df["minutes_per_week"] = df["minutes_total"] / max(cutoff_week, 1)
    df["weeks_since_active"] = cutoff_week - df["last_week"].fillna(0)

    course_avg = df.groupby(["course_id", "session"])["total_clicks"].transform("mean")
    df["clicks_vs_course"] = np.where(course_avg > 0, df["total_clicks"] / course_avg, 0)

    # ---- assessment aggregates
    if len(grd):
        g = grd.copy()
        g["missed"] = (g["submitted"] == 0).astype(int)
        g["is_late"] = (g["days_late"] > 0).astype(int)
        gagg = g.groupby("enrollment_id").agg(
            avg_score=("score", lambda s: s.fillna(0).mean()),
            n_assess=("assessment_id", "count"),
            missed_assessments=("missed", "sum"),
            late_submissions=("is_late", "sum"),
        )
        gagg["submission_rate"] = 1 - gagg["missed_assessments"] / gagg["n_assess"]
        df = df.merge(gagg.drop(columns="n_assess"), left_on="enrollment_id", right_index=True, how="left")
    for c, default in [("avg_score", 0), ("missed_assessments", 0), ("late_submissions", 0),
                       ("submission_rate", 0)]:
        if c not in df:
            df[c] = default
        df[c] = df[c].fillna(default)

    # ---- demographics
    df["is_male"] = (df["gender"] == "M").astype(int)
    df["has_disability"] = (df["disability"] == "Y").astype(int)
    df["age_rank"] = df["age_band"].map({v: i for i, v in enumerate(AGE_ORDER)}).fillna(1)
    df["education_rank"] = df["highest_education"].map({v: i for i, v in enumerate(EDU_ORDER)}).fillna(0)
    df["prev_attempts"] = df["prev_attempts"].fillna(0)

    df = df.drop(columns=["minutes_total", "last_week"], errors="ignore")
    return df
