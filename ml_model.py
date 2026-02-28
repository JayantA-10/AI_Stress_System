# ml_model.py

import joblib
import numpy as np

# Load saved model and encoder
# FIX: wrapped in try/except so Flask gives a clear error if files are missing
try:
    model = joblib.load("model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except FileNotFoundError:
    raise FileNotFoundError(
        "model.pkl or label_encoder.pkl not found. "
        "Please run train_model.py first to generate these files."
    )


def predict_stress(features):
    """
    Predicts stress level from input features.

    features = [
        study_hours,        (float)
        sleep_hours,        (float)
        mood_level,         (int 1-10)
        assignment_pressure,(int 1-10)
        study_consistency,  (int 1-10)
        performance_trend   (int: -1, 0, 1)
    ]

    Returns:
        prediction_label (str): "Low", "Moderate", or "High"
        confidence (float):     percentage confidence (0-100)
    """

    features_array = np.array(features).reshape(1, -1)

    prediction_encoded = model.predict(features_array)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    # Probability for each class
    probabilities = model.predict_proba(features_array)[0]
    confidence = max(probabilities) * 100

    return prediction_label, round(confidence, 2)