# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier        # UPGRADED from LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib


# ── Load Dataset ──────────────────────────────────────────────

df = pd.read_csv("dataset.csv")

# FIX: drop blank rows that exist between student groups in the CSV
df = df.dropna(how='all')
df = df[df['student_name'].notna()]

print("Dataset Loaded Successfully\n")
print(df.head())
print(f"\nTotal records: {len(df)}")


# ── Drop Non-ML Column ────────────────────────────────────────

# We do NOT use student_name for ML
df = df.drop("student_name", axis=1)


# ── Separate Features & Target ────────────────────────────────

X = df.drop("stress_level", axis=1)
y = df["stress_level"]


# ── Encode Target Labels ──────────────────────────────────────

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nClasses found:", label_encoder.classes_)

# Save label encoder for use in ml_model.py
joblib.dump(label_encoder, "label_encoder.pkl")


# ── Train-Test Split ──────────────────────────────────────────

# FIX: added stratify so each class is proportionally represented in test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


# ── Train Random Forest ───────────────────────────────────────
# UPGRADED: RandomForest handles small datasets and non-linear patterns
# much better than Logistic Regression

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=2,
    random_state=42,
    class_weight='balanced'   # handles any class imbalance
)
model.fit(X_train, y_train)


# ── Evaluate Model ────────────────────────────────────────────

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Cross-validation for more reliable accuracy estimate
cv_scores = cross_val_score(model, X, y_encoded, cv=5)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("CV Accuracy:   ", round(cv_scores.mean() * 100, 2), "% (+/-", round(cv_scores.std() * 100, 2), "%)")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


# ── Feature Importance (bonus insight) ───────────────────────

feature_names = X.columns.tolist()
importances = model.feature_importances_
print("Feature Importances:")
for name, score in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"  {name}: {round(score * 100, 1)}%")


# ── Save Model ────────────────────────────────────────────────

joblib.dump(model, "model.pkl")
print("\nModel saved successfully as model.pkl")