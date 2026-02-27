# ml_model.py

import joblib
import numpy as np

# Load saved model and encoder
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def predict_stress(features):
    """
    features = [
        study_hours,
        sleep_hours,
        mood_level,
        assignment_pressure,
        study_consistency,
        performance_trend
    ]
    """

    features_array = np.array(features).reshape(1, -1)

    prediction_encoded = model.predict(features_array)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    # Probability for each class
    probabilities = model.predict_proba(features_array)[0]
    confidence = max(probabilities) * 100

    return prediction_label, round(confidence, 2)