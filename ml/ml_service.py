"""
ml/ml_service.py
Machine Learning Layer — loads the trained model bundle and serves
predictions, risk scores and plain-language risk factors.
"""

import io
import json

import joblib
import numpy as np
import pandas as pd

from database import db
from ml.features import FEATURE_LABELS, RISK_DIRECTION
from ml.train import AT_RISK

_cache = {}   # institution_id -> (model_id, bundle)


def model_available(iid):
    return db.latest_model_meta(iid) is not None


def get_bundle(iid):
    meta = db.latest_model_meta(iid)
    if meta is None:
        raise FileNotFoundError("No trained model yet for this institution.")
    cached = _cache.get(iid)
    if cached and cached[0] == meta["model_id"]:
        return cached[1]
    bundle = joblib.load(io.BytesIO(db.load_model_bytes(iid, meta["model_id"])))
    _cache[iid] = (meta["model_id"], bundle)
    return bundle


def get_report(iid):
    meta = db.latest_model_meta(iid)
    if meta is None:
        raise FileNotFoundError("No trained model yet for this institution.")
    return json.loads(meta["report"])


# Alert threshold chosen in the evaluation (Chapter Five): the highest threshold that keeps
# cross-validated at-risk recall at or above 0.75. Medium band starts at 0.30.
HIGH_THRESHOLD = 0.55
MEDIUM_THRESHOLD = 0.30


def risk_level(score):
    if score >= HIGH_THRESHOLD:
        return "High"
    if score >= MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def _fmt(feature, value):
    if feature in ("avg_score",):
        return f"{value:.0f}%"
    if feature == "submission_rate":
        return f"{value * 100:.0f}%"
    if feature == "clicks_vs_course":
        return f"{value:.2f}x"
    if feature == "engagement_trend":
        return f"{value:+.2f}"
    if feature == "minutes_per_week":
        return f"{value:.0f} min"
    return f"{value:.0f}"


def explain(row, bundle, top_n=3):
    """Top risk factors for one student: features where they sit furthest on the
    risky side of the typical passing student, weighted by model importance."""
    imp, ref = bundle["importance"], bundle["reference"]
    scored = []
    for f, direction in RISK_DIRECTION.items():
        if direction == 0 or f not in ref:
            continue
        z = (row[f] - ref[f]["median"]) / (ref[f]["std"] or 1)
        harm = max(0.0, direction * z) * (imp.get(f, 0) + 0.01)
        if harm > 0:
            scored.append((harm, f))
    scored.sort(reverse=True)
    parts = []
    for _, f in scored[:top_n]:
        word = "High" if RISK_DIRECTION[f] > 0 else "Low"
        label = FEATURE_LABELS[f]
        label = label if label.startswith("LMS") else label[0].lower() + label[1:]
        parts.append(f"{word} {label} ({_fmt(f, row[f])} vs typical "
                     f"{_fmt(f, ref[f]['median'])})")
    return " | ".join(parts) if parts else "No major risk factors"


def predict(iid, features_df):
    """features_df must contain the model's feature columns.
    Returns a DataFrame with predicted_result, risk_score, risk_level,
    confidence_score, top_factors and per-class probabilities."""
    bundle = get_bundle(iid)
    model, feats, classes = bundle["model"], bundle["features"], bundle["classes"]
    X = features_df[feats]
    proba = model.predict_proba(X)
    pred = np.array(classes)[proba.argmax(axis=1)]
    risk_idx = [classes.index(c) for c in AT_RISK if c in classes]
    risk = proba[:, risk_idx].sum(axis=1)

    out = pd.DataFrame(index=features_df.index)
    out["predicted_result"] = pred
    out["risk_score"] = risk.round(4)
    out["risk_level"] = [risk_level(r) for r in risk]
    out["confidence_score"] = proba.max(axis=1).round(4)
    for i, c in enumerate(classes):
        out[f"p_{c}"] = proba[:, i].round(4)
    out["top_factors"] = [explain(r, bundle) for _, r in X.iterrows()]
    out["model_used"] = bundle["model_name"]
    return out
