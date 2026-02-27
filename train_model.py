# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib


#Load Dataset

df = pd.read_csv("dataset.csv")

print("Dataset Loaded Successfully\n")
print(df.head())


# Drop Non-ML Column

# We do NOT use student_name for ML
df = df.drop("student_name", axis=1)


# Separate Features & Target

X = df.drop("stress_level", axis=1)
y = df["stress_level"]


# Encode Target Labels

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder for later use
joblib.dump(label_encoder, "label_encoder.pkl")


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)


# Train Logistic Regression

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


#  Evaluate Model

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


#  Save Model

joblib.dump(model, "model.pkl")

print("\nModel saved successfully as model.pkl")