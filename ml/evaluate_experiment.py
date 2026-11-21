"""
ml/evaluate_experiment.py
Offline evaluation used in Chapter Five. Uses the same features, model grids and
student-grouped splitting as ml/train.py, and adds:
  * a majority-class baseline
  * cross-validated mean and standard deviation per model
  * at-risk (Fail or Withdrawn) detection metrics, McNemar's exact test
  * threshold analysis on out-of-fold predictions, calibration (Brier score)
  * a fairness audit by gender, age band and disability
  * sensitivity of performance to the prediction week (4, 8, 12)
  * scoring time for 1,000 records

Run:  python ml/evaluate_experiment.py --out evaluation_output
"""

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if "LAS_DB_PATH" not in os.environ and not os.environ.get("DATABASE_URL"):
    os.environ["LAS_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "evaluation.db")

from scipy.stats import binomtest                                            # noqa: E402
from sklearn.base import clone                                               # noqa: E402
from sklearn.calibration import calibration_curve                            # noqa: E402
from sklearn.dummy import DummyClassifier                                    # noqa: E402
from sklearn.inspection import permutation_importance                        # noqa: E402
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,  # noqa: E402
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold      # noqa: E402

from business.auth import register_institution                               # noqa: E402
from database import db                                                      # noqa: E402
from ml.features import FEATURES                                             # noqa: E402
from ml.train import AT_RISK, LABELS, load_training_frame, model_grid        # noqa: E402
from scripts.generate_institution_data import generate                       # noqa: E402

SEED = 42


def at_risk_prob(model, X):
    classes = list(model.classes_)
    p = model.predict_proba(X)
    idx = [classes.index(c) for c in AT_RISK if c in classes]
    return np.clip(p[:, idx].sum(axis=1), 0, 1) if idx else np.zeros(len(X))


def binary_metrics(y_true_risk, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_risk, pred, labels=[0, 1]).ravel()
    return {"precision": precision_score(y_true_risk, pred, zero_division=0),
            "recall": recall_score(y_true_risk, pred, zero_division=0),
            "f1": f1_score(y_true_risk, pred, zero_division=0),
            "macro_f1": f1_score(y_true_risk, pred, average="macro", zero_division=0),
            "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
            "auc": roc_auc_score(y_true_risk, prob) if len(set(y_true_risk)) > 1 and prob.std() > 0 else 0.5,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def mcnemar(y_true, pred_a, pred_b):
    a_right, b_right = pred_a == y_true, pred_b == y_true
    b = int((a_right & ~b_right).sum())
    c = int((~a_right & b_right).sum())
    p = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
    return {"a_only_correct": b, "b_only_correct": c, "p_value": p}


def run_week(iid, week, detailed):
    df = load_training_frame(iid, week)
    X, y, groups = df[FEATURES], df["final_result"], df["student_id"].values
    risk = y.isin(AT_RISK).astype(int).values
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, te = next(outer.split(X, y, groups))
    Xtr, Xte, ytr, yte, gtr = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], groups[tr]
    rte = risk[te]
    labels = [l for l in LABELS if l in set(y)]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    out = {"week": week, "rows": len(df), "train_rows": len(tr), "test_rows": len(te),
           "train_students": int(len(set(gtr))), "test_students": int(len(set(groups[te]))),
           "class_balance": y.value_counts().to_dict(), "at_risk_share": float(risk.mean()), "models": {}}
    models = {"Majority-class baseline": (DummyClassifier(strategy="most_frequent"), {})}
    models.update(model_grid())
    fitted = {}
    for name, (est, grid) in models.items():
        t0 = time.time()
        search = GridSearchCV(est, grid or {"strategy": ["most_frequent"]}, cv=cv, scoring="f1_weighted",
                              n_jobs=-1, return_train_score=False)
        search.fit(Xtr, ytr, groups=gtr)
        best = search.best_estimator_
        fitted[name] = best
        i = search.best_index_
        folds = [search.cv_results_[f"split{k}_test_score"][i] for k in range(5)]
        # out-of-fold at-risk recall on the training folds
        oof = np.zeros(len(Xtr))
        for a, b in cv.split(Xtr, ytr, gtr):
            m = clone(best).fit(Xtr.iloc[a], ytr.iloc[a])
            oof[b] = at_risk_prob(m, Xtr.iloc[b])
        rtr = risk[tr]
        fold_recall = [recall_score(rtr[b], (oof[b] >= 0.5).astype(int), zero_division=0)
                       for _, b in cv.split(Xtr, ytr, gtr)]
        pred = best.predict(Xte)
        prob = at_risk_prob(best, Xte)
        mc_auc = None
        try:
            mc_auc = roc_auc_score(yte, best.predict_proba(Xte), multi_class="ovr", average="weighted",
                                   labels=list(best.classes_))
        except ValueError:
            pass
        res = {"cv_f1_weighted_mean": float(np.mean(folds)), "cv_f1_weighted_sd": float(np.std(folds)),
               "cv_at_risk_recall_mean": float(np.mean(fold_recall)), "cv_at_risk_recall_sd": float(np.std(fold_recall)),
               "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
               "accuracy": accuracy_score(yte, pred),
               "precision_w": precision_score(yte, pred, average="weighted", zero_division=0),
               "recall_w": recall_score(yte, pred, average="weighted", zero_division=0),
               "f1_w": f1_score(yte, pred, average="weighted", zero_division=0),
               "f1_macro4": f1_score(yte, pred, average="macro", zero_division=0),
               "auc_ovr": mc_auc, "at_risk": binary_metrics(rte, prob),
               "fit_seconds": time.time() - t0}
        if detailed:
            res["confusion_matrix"] = confusion_matrix(yte, pred, labels=labels).tolist()
            fpr, tpr, _ = roc_curve(rte, prob)
            res["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            res["_oof"], res["_prob"], res["_pred_bin"] = oof, prob, (prob >= 0.5).astype(int)
        out["models"][name] = res
    ml = [n for n in out["models"] if n != "Majority-class baseline"]
    best_name = max(ml, key=lambda n: (out["models"][n]["cv_f1_weighted_mean"],
                                       out["models"][n]["cv_at_risk_recall_mean"]))
    out["best_model"] = best_name

    if detailed:
        best = fitted[best_name]
        bres = out["models"][best_name]
        out["labels"] = labels
        out["mcnemar"] = {n: mcnemar(rte, bres["_pred_bin"], out["models"][n]["_pred_bin"])
                          for n in out["models"] if n != best_name}
        # threshold analysis: chosen on out-of-fold training predictions, applied to test
        rtr = risk[tr]
        grid = np.round(np.arange(0.20, 0.81, 0.05), 2)
        thr = []
        for t in grid:
            thr.append({"threshold": float(t), **{f"oof_{k}": v for k, v in binary_metrics(rtr, bres["_oof"], t).items()
                                                  if k in ("precision", "recall", "f1")},
                        **{f"test_{k}": v for k, v in binary_metrics(rte, bres["_prob"], t).items()
                           if k in ("precision", "recall", "f1", "fpr")}})
        out["thresholds"] = thr
        ok = [r["threshold"] for r in thr if r["oof_recall"] >= 0.75]
        out["chosen_threshold"] = max(ok) if ok else 0.5
        out["chosen_threshold_test"] = binary_metrics(rte, bres["_prob"], out["chosen_threshold"])
        out["brier"] = {n: brier_score_loss(rte, out["models"][n]["_prob"]) for n in out["models"]}
        frac, mean_pred = calibration_curve(rte, bres["_prob"], n_bins=10, strategy="quantile")
        out["calibration"] = {"observed": frac.tolist(), "predicted": mean_pred.tolist()}
        # fairness audit on the test set (attributes are NOT model inputs)
        test_df = df.iloc[te].assign(risk=rte, pred=(bres["_prob"] >= out["chosen_threshold"]).astype(int))
        fair = {}
        for attr in ["gender", "age_band", "disability"]:
            rows = []
            for g, d in test_df.groupby(attr):
                pos, neg = d[d.risk == 1], d[d.risk == 0]
                rows.append({"group": str(g), "n": len(d), "at_risk_share": float(d.risk.mean()),
                             "recall": float(pos.pred.mean()) if len(pos) else None,
                             "fpr": float(neg.pred.mean()) if len(neg) else None})
            fair[attr] = rows
        out["fairness"] = fair
        perm = permutation_importance(best, Xte, yte, n_repeats=10, random_state=SEED, scoring="f1_weighted", n_jobs=-1)
        out["importance"] = sorted([{"feature": f, "mean": float(m), "sd": float(s)}
                                    for f, m, s in zip(FEATURES, perm.importances_mean, perm.importances_std)],
                                   key=lambda r: -r["mean"])
        big = pd.concat([Xte] * (1000 // len(Xte) + 1)).head(1000)
        t0 = time.time()
        best.predict_proba(big)
        out["scoring_seconds_1000"] = time.time() - t0
        for n in out["models"]:
            for k in ("_oof", "_prob", "_pred_bin"):
                out["models"][n].pop(k, None)
    return out


def figures(res, sens, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    C = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8a8983"]
    names = [n for n in res["models"] if n != "Majority-class baseline"]

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for i, n in enumerate(names):
        r = res["models"][n]
        ax.plot(r["roc"]["fpr"], r["roc"]["tpr"], color=C[i], lw=1.8, label=f"{n} (AUC = {r['at_risk']['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#bbbbbb", lw=1, label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
    ax.legend(frameon=False, loc="lower right"); fig.tight_layout(); fig.savefig(f"{out_dir}/fig5_roc.png", dpi=200)

    cm = np.array(res["models"][res["best_model"]]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.imshow(cm, cmap="Blues")
    lab = res["labels"]
    ax.set_xticks(range(len(lab)), lab); ax.set_yticks(range(len(lab)), lab)
    for i in range(len(lab)):
        for j in range(len(lab)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted outcome"); ax.set_ylabel("Actual outcome")
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig5_confusion.png", dpi=200)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="#bbbbbb", lw=1, label="Perfect calibration")
    ax.plot(res["calibration"]["predicted"], res["calibration"]["observed"], "o-", color=C[0], lw=1.8,
            label=res["best_model"])
    ax.set_xlabel("Mean predicted at-risk probability"); ax.set_ylabel("Observed share at risk")
    ax.legend(frameon=False, loc="upper left"); fig.tight_layout(); fig.savefig(f"{out_dir}/fig5_calibration.png", dpi=200)

    from ml.features import FEATURE_LABELS
    imp = res["importance"][:10][::-1]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.barh([FEATURE_LABELS[r["feature"]] for r in imp], [r["mean"] for r in imp], xerr=[r["sd"] for r in imp],
            color=C[0], ecolor="#666666")
    ax.set_xlabel("Mean drop in weighted F1-score when shuffled")
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig5_importance.png", dpi=200)

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    weeks = [s["week"] for s in sens]
    for i, n in enumerate(names):
        ax.plot(weeks, [s["models"][n]["at_risk"]["auc"] for s in sens], "o-", color=C[i], lw=1.8, label=n)
    ax.set_xticks(weeks, [f"Week {w}" for w in weeks]); ax.set_ylabel("At-risk AUC-ROC (test set)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f"{out_dir}/fig5_sensitivity.png", dpi=200)

    t = pd.DataFrame(res["thresholds"])
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(t.threshold, t.oof_recall, "o-", color=C[0], label="Recall (cross-validation)")
    ax.plot(t.threshold, t.oof_precision, "s-", color=C[1], label="Precision (cross-validation)")
    ax.axvline(res["chosen_threshold"], color="#888888", ls=":", lw=1)
    ax.axhline(0.75, color="#cccccc", ls="--", lw=1)
    ax.set_xlabel("Alert threshold (predicted at-risk probability)"); ax.set_ylabel("Score")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f"{out_dir}/fig5_threshold.png", dpi=200)
    plt.close("all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evaluation_output")
    ap.add_argument("--students", type=int, default=900)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    db.drop_all() if not os.environ.get("DATABASE_URL") else None
    iid, _ = register_institution("Evaluation Institution", "EVAL", "University", "Nigeria", "Evaluator",
                                  "evaluator@eval.edu", "Eval12345")
    generate(iid, n_students=args.students, seed=SEED, verbose=True)
    main_res = run_week(iid, 8, detailed=True)
    sens = [run_week(iid, w, detailed=False) for w in (4, 12)]
    sens = sorted(sens + [main_res], key=lambda r: r["week"])
    figures(main_res, sens, args.out)
    with open(f"{args.out}/results_week8.json", "w") as f:
        json.dump(main_res, f, indent=2, default=float)
    with open(f"{args.out}/sensitivity.json", "w") as f:
        json.dump([{"week": s["week"], "best": s["best_model"],
                    "models": {n: {"auc": m["at_risk"]["auc"], "recall": m["at_risk"]["recall"],
                                   "f1_w": m["f1_w"], "accuracy": m["accuracy"]} for n, m in s["models"].items()}}
                   for s in sens], f, indent=2, default=float)
    print(json.dumps({"best": main_res["best_model"], "threshold": main_res["chosen_threshold"]}, indent=2))


if __name__ == "__main__":
    main()
