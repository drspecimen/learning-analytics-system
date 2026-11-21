"""
ml/train.py
Model Development (Figure 3.5): train/test split -> Logistic Regression,
Decision Tree, Random Forest -> hyperparameter tuning (GridSearchCV, 5-fold)
-> evaluation (Accuracy, Precision, Recall, F1, AUC-ROC, confusion matrix)
-> best model selection -> saved for the prediction dashboard.

Trains on COMPLETED enrollments (those with a final_result), using features
from the first `cutoff_week` weeks only.

Run directly:  python ml/train.py
"""

import io
import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database import db                                  # noqa: E402
from ml.features import FEATURES, build_features         # noqa: E402

DEFAULT_CUTOFF = 8
MIN_PER_CLASS = 8          # needed for a stratified test split plus 5-fold cross-validation
MIN_ROWS = 40
LABELS = ["Distinction", "Pass", "Fail", "Withdrawn"]
AT_RISK = ["Fail", "Withdrawn"]


def load_training_frame(iid, cutoff_week=DEFAULT_CUTOFF):
    base = db.get_enrollment_base_df(iid)
    base = base[base["final_result"].notna()]
    if base.empty:
        raise ValueError("No past students with a final_result were found. Upload completed history first.")
    ids = base["enrollment_id"].tolist()
    return build_features(base, db.get_activity_df(iid, ids), db.get_grades_df(iid, ids), cutoff_week)


def readiness(results):
    """results: a Series of final_result values. Returns (ok, list of problems)."""
    counts = results.value_counts()
    problems = []
    if len(results) < MIN_ROWS:
        problems.append(f"Only {len(results)} past enrollments; at least {MIN_ROWS} are needed.")
    if counts.reindex(AT_RISK, fill_value=0).sum() == 0:
        problems.append("No past Fail or Withdrawn results — the model cannot learn what at-risk looks like.")
    if counts.reindex(["Pass", "Distinction"], fill_value=0).sum() == 0:
        problems.append("No past Pass or Distinction results.")
    small = [f"{k} ({v})" for k, v in counts.items() if v < MIN_PER_CLASS]
    if small:
        problems.append(f"Each result needs at least {MIN_PER_CLASS} examples; too few: {', '.join(small)}. "
                        "Rare results will be merged automatically if possible.")
    return len(problems) == 0, problems


def model_grid():
    return {
        "Logistic Regression": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42, class_weight="balanced"),
            {"max_depth": [4, 6, 10], "min_samples_leaf": [5, 20]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),
            {"n_estimators": [200], "max_depth": [8, 14], "min_samples_leaf": [2, 5]},
        ),
    }


def evaluate(model, X_test, y_test, labels):
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    try:
        auc = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted", labels=classes)
    except ValueError:
        auc = None
    # binary at-risk ROC curve (Fail or Withdrawn vs the rest)
    risk_idx = [classes.index(c) for c in AT_RISK if c in classes]
    risk_true = y_test.isin(AT_RISK).astype(int)
    risk_prob = proba[:, risk_idx].sum(axis=1)
    fpr, tpr, _ = roc_curve(risk_true, risk_prob)
    risk_auc = roc_auc_score(risk_true, risk_prob)
    step = max(1, len(fpr) // 60)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "auc_roc": round(auc, 4) if auc is not None else None,
        "at_risk_auc": round(risk_auc, 4),
        "at_risk_recall": round(recall_score(risk_true, (risk_prob >= 0.5).astype(int), zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "roc_curve": {"fpr": fpr[::step].round(4).tolist() + [1.0],
                      "tpr": tpr[::step].round(4).tolist() + [1.0]},
    }


def train(iid, cutoff_week=DEFAULT_CUTOFF, verbose=True, user_id=None, progress=None):
    log = print if verbose else (lambda *a, **k: None)
    step = progress or (lambda f, t: None)
    log(f"Building training features (first {cutoff_week} weeks of each completed course)...")
    step(0.05, "Building features from past students")
    df = load_training_frame(iid, cutoff_week)
    # merge a too-rare class into its neighbour so stratified CV can run
    counts = df["final_result"].value_counts()
    if counts.get("Distinction", 0) < MIN_PER_CLASS and "Pass" in counts:
        df.loc[df["final_result"] == "Distinction", "final_result"] = "Pass"
    counts = df["final_result"].value_counts()
    if counts.get("Withdrawn", 0) < MIN_PER_CLASS and "Fail" in counts:
        df.loc[df["final_result"] == "Withdrawn", "final_result"] = "Fail"
    counts = df["final_result"].value_counts()
    if len(df) < MIN_ROWS or counts.min() < MIN_PER_CLASS or len(counts) < 2:
        raise ValueError(f"Not enough history to train: {len(df)} past enrollments "
                         f"({counts.to_dict()}). Need at least {MIN_ROWS} with {MIN_PER_CLASS}+ of each result.")
    X, y = df[FEATURES], df["final_result"]
    labels = [l for l in LABELS if l in set(y)]
    log(f"Training rows: {len(df)} | class balance: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results, fitted = {}, {}
    for k, (name, (estimator, params)) in enumerate(model_grid().items()):
        log(f"  Tuning {name}...")
        step(0.15 + 0.25 * k, f"Training and tuning {name}")
        search = GridSearchCV(estimator, params, cv=cv, scoring="f1_weighted", n_jobs=-1)
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        metrics = evaluate(search.best_estimator_, X_test, y_test, labels)
        metrics["best_params"] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        metrics["cv_f1"] = round(search.best_score_, 4)
        results[name] = metrics
        log(f"    accuracy {metrics['accuracy']}  F1 {metrics['f1_score']}  AUC {metrics['auc_roc']}")

    best_name = max(results, key=lambda k: (results[k]["f1_score"], results[k]["auc_roc"] or 0))
    best = fitted[best_name]
    log(f"Best model: {best_name}")

    log("Computing feature importance (permutation)...")
    step(0.9, "Measuring what drives the predictions")
    perm = permutation_importance(best, X_test, y_test, n_repeats=5, random_state=42,
                                  scoring="f1_weighted", n_jobs=-1)
    importance = pd.Series(np.clip(perm.importances_mean, 0, None), index=FEATURES)
    if importance.sum() > 0:
        importance = importance / importance.sum()
    importance = importance.sort_values(ascending=False)

    ok = df[df["final_result"].isin(["Pass", "Distinction"])]
    reference = {f: {"median": float(ok[f].median()), "std": float(df[f].std() or 1)} for f in FEATURES}

    trained_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    buf = io.BytesIO()
    joblib.dump({
        "model": best, "model_name": best_name, "features": FEATURES,
        "classes": list(best.classes_), "importance": importance.to_dict(),
        "reference": reference, "cutoff_week": cutoff_week, "trained_at": trained_at,
        "trained_rows": len(df),
    }, buf, compress=3)

    report = {"best_model": best_name, "trained_at": trained_at, "cutoff_week": cutoff_week,
              "trained_rows": len(df), "test_rows": len(X_test), "labels": labels,
              "class_balance": y.value_counts().to_dict(),
              "feature_importance": importance.round(4).to_dict(), "results": results}
    db.save_model(iid, buf.getvalue(), json.dumps(report))
    db.prune_models(iid)
    for name, m in results.items():
        db.insert_model_result(iid, name, m["accuracy"], m["precision"], m["recall"], m["f1_score"],
                               m["auc_roc"], name == best_name, json.dumps(m["best_params"]), len(df))
    db.insert_log(iid, user_id, "MODEL_TRAINED", f"Best: {best_name} (F1 {results[best_name]['f1_score']})")
    log(f"Saved model for institution {iid}")
    return report


if __name__ == "__main__":
    train(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
