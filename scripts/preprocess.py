"""
scripts/preprocess.py
Preprocessing pipeline for the OULAD dataset — Section 3.5.
Cleans, encodes, normalizes, and engineers features, producing
data/processed/training_dataset.csv for model training (Figure 3.5).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def load_data():
    print("Loading raw CSV files...")
    student_info = pd.read_csv(f"{RAW_DIR}/studentInfo.csv")
    student_assessment = pd.read_csv(f"{RAW_DIR}/studentAssessment.csv")
    assessments = pd.read_csv(f"{RAW_DIR}/assessments.csv")
    student_vle = pd.read_csv(f"{RAW_DIR}/studentVle.csv")
    student_registration = pd.read_csv(f"{RAW_DIR}/studentRegistration.csv")
    print("All files loaded successfully.")
    return student_info, student_assessment, assessments, student_vle, student_registration


def engineer_assessment_features(student_assessment, assessments):
    print("Engineering assessment-based features...")
    merged = student_assessment.merge(assessments, on="id_assessment", how="left")

    merged["score"] = pd.to_numeric(merged["score"], errors="coerce")
    merged["weight"] = pd.to_numeric(merged["weight"], errors="coerce")

    merged["weighted_score"] = merged["score"].fillna(0) * (merged["weight"].fillna(0) / 100)

    agg = merged.groupby(["id_student", "code_module", "code_presentation"]).agg(
        cumulative_score=("weighted_score", "sum"),
        avg_score=("score", "mean"),
        num_assessments=("id_assessment", "count"),
    ).reset_index()
    return agg


def engineer_vle_features(student_vle):
    print("Engineering VLE interaction features...")
    student_vle["sum_click"] = pd.to_numeric(student_vle["sum_click"], errors="coerce")

    agg = student_vle.groupby(["id_student", "code_module", "code_presentation"]).agg(
        total_clicks=("sum_click", "sum"),
        avg_daily_clicks=("sum_click", "mean"),
        active_days=("date", "nunique"),
    ).reset_index()
    return agg


def build_master_table(student_info, assessment_features, vle_features, student_registration):
    print("Merging all sources into a master table...")
    keys = ["id_student", "code_module", "code_presentation"]
    df = student_info.merge(assessment_features, on=keys, how="left")
    df = df.merge(vle_features, on=keys, how="left")
    df = df.merge(
        student_registration[["id_student", "code_module", "code_presentation", "date_registration"]],
        on=keys, how="left"
    )
    return df


def clean_data(df):
    print("Cleaning data (missing values, duplicates)...")
    df = df.drop_duplicates()

    numeric_fill_cols = ["cumulative_score", "avg_score", "num_assessments",
                          "total_clicks", "avg_daily_clicks", "active_days"]
    for col in numeric_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if "date_registration" in df.columns:
        df["date_registration"] = pd.to_numeric(df["date_registration"], errors="coerce")
        df["date_registration"] = df["date_registration"].fillna(df["date_registration"].median())

    df = df.dropna(subset=["final_result"])
    return df


def encode_categoricals(df):
    print("Encoding categorical variables...")
    categorical_cols = ["gender", "region", "highest_education", "imd_band",
                         "age_band", "disability", "code_module", "code_presentation"]
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    target_le = LabelEncoder()
    df["final_result"] = target_le.fit_transform(df["final_result"].astype(str))
    encoders["final_result"] = target_le

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    joblib.dump(target_le, f"{PROCESSED_DIR}/target_encoder.pkl")
    print(f"Saved target label encoder to {PROCESSED_DIR}/target_encoder.pkl")
    print(f"Label mapping: {dict(zip(target_le.classes_, target_le.transform(target_le.classes_)))}")

    return df, encoders


def normalize_numeric(df):
    print("Normalizing numeric features...")
    numeric_cols = ["studied_credits", "cumulative_score", "avg_score", "num_assessments",
                     "total_clicks", "avg_daily_clicks", "active_days", "date_registration"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def save_output(df):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = f"{PROCESSED_DIR}/training_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned training dataset to {output_path}")
    print(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns")


def main():
    student_info, student_assessment, assessments, student_vle, student_registration = load_data()

    assessment_features = engineer_assessment_features(student_assessment, assessments)
    vle_features = engineer_vle_features(student_vle)

    df = build_master_table(student_info, assessment_features, vle_features, student_registration)
    df = clean_data(df)
    df, encoders = encode_categoricals(df)
    df = normalize_numeric(df)

    save_output(df)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()