-- database/schema.sql
-- Schema for the AI-Driven Learning Analytics System (Figure 3.3)

CREATE TABLE IF NOT EXISTS Users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('Administrator', 'Lecturer')),
    name TEXT,
    email TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS Students (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    id_student TEXT NOT NULL,
    code_module TEXT,
    code_presentation TEXT,
    gender TEXT,
    age_band TEXT,
    highest_education TEXT,
    final_result TEXT,
    uploaded_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (uploaded_by) REFERENCES Users(user_id)
);

CREATE TABLE IF NOT EXISTS Datasets (
    dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    uploaded_by INTEGER,
    row_count INTEGER,
    uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (uploaded_by) REFERENCES Users(user_id)
);

CREATE TABLE IF NOT EXISTS Predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    dataset_id INTEGER,
    predicted_result TEXT,
    confidence_score REAL,
    model_used TEXT,
    predicted_by INTEGER,
    predicted_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES Students(student_id),
    FOREIGN KEY (dataset_id) REFERENCES Datasets(dataset_id),
    FOREIGN KEY (predicted_by) REFERENCES Users(user_id)
);

CREATE TABLE IF NOT EXISTS Model_Results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER,
    model_name TEXT NOT NULL,
    accuracy REAL,
    precision_score REAL,
    recall_score REAL,
    f1_score REAL,
    auc_roc REAL,
    evaluated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES Predictions(prediction_id)
);

CREATE TABLE IF NOT EXISTS System_Logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    action TEXT NOT NULL,
    details TEXT,
    logged_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
);