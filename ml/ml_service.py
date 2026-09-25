"""
ml/ml_service.py
Machine Learning Layer — service interface wrapping the prediction models.
Full training logic is implemented in Sprint 4; this defines the contract
the rest of the app calls against so development can proceed in parallel.
"""

import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained_models")


def train_models(training_data_path):
    """
    Will train Logistic Regression, Decision Tree, and Random Forest
    on the given dataset and save the best one. Implemented in Sprint 4.
    """
    raise NotImplementedError("Model training will be implemented in Sprint 4.")


def get_best_model():
    """
    Will load and return the best-performing serialized model.
    Implemented in Sprint 4.
    """
    raise NotImplementedError("Model loading will be implemented in Sprint 4.")


def predict(student_features):
    """
    Will return (predicted_result, confidence_score) for a given student's features.
    Implemented in Sprint 4.
    """
    raise NotImplementedError("Prediction will be implemented in Sprint 4.")