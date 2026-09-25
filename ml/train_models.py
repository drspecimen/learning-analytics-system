"""
ml/train_models.py
Trains, tunes, evaluates, and selects the best model (Figure 3.5).
Run this ONCE (or whenever you want to retrain) to produce:
  ml/trained_models/best_model.pkl
  ml/trained_models/model_comparison.json
"""

import pandas as pd
import numpy as np
import json
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

DATA_PATH = "data/processed/training_dataset.csv"
MODEL_DIR = "ml/trained_models"


def load_data():
    print("Loading training dataset...")
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["final_result"])
    X = X.select_dtypes(include=[np.number])  # keep only numeric feature columns
    y = df["final_result"]
    return X, y


def get_model_grids():
    return {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs"]
            }
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5]
            }
        }
    }


def evaluate_model(model, X_test, y_test, is_multiclass):
    y_pred = model.predict(X_test)

    average = "weighted" if is_multiclass else "binary"

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=average, zero_division=0)
    recall = recall_score(y_test, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average, zero_division=0)

    try:
        if is_multiclass:
            y_proba = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        else:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None

    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4) if auc is not None else None,
        "confusion_matrix": cm
    }


def main():
    X, y = load_data()
    is_multiclass = y.nunique() > 2
    print(f"Detected {y.nunique()} classes -> {'multiclass' if is_multiclass else 'binary'} evaluation mode")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    grids = get_model_grids()
    results = {}
    fitted_models = {}

    for name, config in grids.items():
        print(f"\nTraining {name} with GridSearchCV...")
        search = GridSearchCV(
            config["model"], config["params"],
            cv=5, scoring="f1_weighted", n_jobs=-1
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        fitted_models[name] = best_model

        metrics = evaluate_model(best_model, X_test, y_test, is_multiclass)
        metrics["best_params"] = search.best_params_
        results[name] = metrics

        print(f"{name} — Accuracy: {metrics['accuracy']} | F1: {metrics['f1_score']} | AUC-ROC: {metrics['auc_roc']}")
        print(f"Best params: {search.best_params_}")

    # Select best model by F1 score
    best_name = max(results, key=lambda k: results[k]["f1_score"])
    best_model = fitted_models[best_name]

    print(f"\nBest model: {best_name} (F1 = {results[best_name]['f1_score']})")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")

    with open(f"{MODEL_DIR}/model_comparison.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "results": results
        }, f, indent=2)

    print(f"\nSaved best model to {MODEL_DIR}/best_model.pkl")
    print(f"Saved comparison report to {MODEL_DIR}/model_comparison.json")


if __name__ == "__main__":
    main()