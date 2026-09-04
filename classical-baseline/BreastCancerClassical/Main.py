import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)

# ---------------------------------------------------------
# 1. Load data — FIXING A HEADER MISALIGNMENT
# ---------------------------------------------------------
# The raw CSV has 31 data columns but only 30 header names.
# A plain pd.read_csv() silently uses the first data column as the
# index and shifts every column name by one — so "worst fractal
# dimension" ends up holding the target label, not real data.
# We fix this by reading without treating the first row as an index
# and explicitly assigning correct names: 30 features + 1 target.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "breast_cancer_Main.csv")

feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
]
all_columns = feature_names + ["target"]

df = pd.read_csv(CSV_PATH, header=0, names=all_columns)

print("Shape:", df.shape)
print(df.head())
print("\nMissing values:", df.isnull().sum().sum())
print("\nTarget distribution:\n", df["target"].value_counts())
# In the original sklearn dataset: 0 = malignant, 1 = benign

# ---------------------------------------------------------
# 2. Split features / target
# ---------------------------------------------------------
X = df[feature_names]
y = df["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 3. Build pipelines (scaling + model)
# ---------------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
    ]),
}

# ---------------------------------------------------------
# 4. Train, cross-validate, evaluate
# ---------------------------------------------------------
results = {}
for name, pipe in models.items():
    print(f"\n=== {name} ===")

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    print(f"CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    results[name] = acc

    print(f"Test accuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["malignant", "benign"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------
# 5. Pick best model, show feature importance
# ---------------------------------------------------------
best_name = max(results, key=results.get)
print(f"\nBest model on test accuracy: {best_name} ({results[best_name]:.4f})")

rf_pipe = models["Random Forest"]
rf = rf_pipe.named_steps["clf"]
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nTop 10 important features (Random Forest):")
print(importances.head(10))