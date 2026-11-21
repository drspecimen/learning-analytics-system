"""
database/db.py
Data-access layer for the AI-Driven Learning Analytics System.
Wraps all SQLite operations so the Business Layer never writes raw SQL directly.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "app.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

_db_ready = False


def get_connection():
    global _db_ready
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if not _db_ready:
        with open(SCHEMA_PATH, "r") as f:
            conn.executescript(f.read())
        _db_ready = True
    return conn

def init_db():
    conn = get_connection()
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")


# ---------- Users ----------

def insert_user(username, password_hash, role, name=None, email=None):
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO Users (username, password_hash, role, name, email) VALUES (?, ?, ?, ?, ?)",
        (username, password_hash, role, name, email)
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return user_id


def get_user_by_username(username):
    conn = get_connection()
    row = conn.execute("SELECT * FROM Users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ---------- Students ----------

def insert_student(id_student, code_module, code_presentation, gender,
                    age_band, highest_education, final_result, uploaded_by):
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO Students
           (id_student, code_module, code_presentation, gender, age_band,
            highest_education, final_result, uploaded_by)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (id_student, code_module, code_presentation, gender, age_band,
         highest_education, final_result, uploaded_by)
    )
    conn.commit()
    student_id = cur.lastrowid
    conn.close()
    return student_id


def get_students_by_module(code_module):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM Students WHERE code_module = ?", (code_module,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- Datasets ----------

def insert_dataset(file_name, uploaded_by, row_count):
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO Datasets (file_name, uploaded_by, row_count) VALUES (?, ?, ?)",
        (file_name, uploaded_by, row_count)
    )
    conn.commit()
    dataset_id = cur.lastrowid
    conn.close()
    return dataset_id


def get_all_datasets():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM Datasets ORDER BY uploaded_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- Predictions ----------

def insert_prediction(student_id, dataset_id, predicted_result, confidence_score,
                       model_used, predicted_by):
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO Predictions
           (student_id, dataset_id, predicted_result, confidence_score, model_used, predicted_by)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (student_id, dataset_id, predicted_result, confidence_score, model_used, predicted_by)
    )
    conn.commit()
    prediction_id = cur.lastrowid
    conn.close()
    return prediction_id


def get_predictions_by_student(student_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM Predictions WHERE student_id = ? ORDER BY predicted_at DESC", (student_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- Model_Results ----------

def insert_model_result(prediction_id, model_name, accuracy, precision_score,
                         recall_score, f1_score, auc_roc):
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO Model_Results
           (prediction_id, model_name, accuracy, precision_score, recall_score, f1_score, auc_roc)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (prediction_id, model_name, accuracy, precision_score, recall_score, f1_score, auc_roc)
    )
    conn.commit()
    result_id = cur.lastrowid
    conn.close()
    return result_id


def get_all_model_results():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM Model_Results ORDER BY evaluated_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- System_Logs ----------

def insert_log(user_id, action, details=None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO System_Logs (user_id, action, details) VALUES (?, ?, ?)",
        (user_id, action, details)
    )
    conn.commit()
    conn.close()


def get_logs_by_user(user_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM System_Logs WHERE user_id = ? ORDER BY logged_at DESC", (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- Self-test block ----------

if __name__ == "__main__":
    init_db()
    print("All six tables created: Users, Students, Datasets, Predictions, Model_Results, System_Logs")