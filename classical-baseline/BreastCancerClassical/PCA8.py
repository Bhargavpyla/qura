"""
PCA (8 components) on the Breast Cancer dataset
------------------------------------------------
Run locally: python pca8.py
Requires: pandas, numpy, scikit-learn, matplotlib
Install if needed:
    pip install pandas numpy scikit-learn matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# 1. Load data (fixing the 31-columns-vs-30-headers issue)
# ---------------------------------------------------------
CSV_PATH = "breast_cancer_Main.csv"

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

X = df[feature_names]
y = df["target"].astype(int)

print("Data shape:", X.shape)

# ---------------------------------------------------------
# 2. Standardize (PCA is scale-sensitive)
# ---------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# 3. PCA with 8 components
# ---------------------------------------------------------
N_COMPONENTS = 8
pca = PCA(n_components=N_COMPONENTS, random_state=42)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
print(f"\nExplained variance per component (PC1..PC{N_COMPONENTS}):")
for i, v in enumerate(explained, 1):
    print(f"  PC{i}: {v:.4f}")
print(f"Total variance explained by {N_COMPONENTS} components: {explained.sum():.4f}")

# ---------------------------------------------------------
# 4. Scree plot (variance explained per component)
# ---------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.bar(range(1, N_COMPONENTS + 1), explained, alpha=0.7, label="Individual")
plt.plot(range(1, N_COMPONENTS + 1), np.cumsum(explained), marker="o",
          color="red", label="Cumulative")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA (8 components) - Explained Variance")
plt.legend()
plt.tight_layout()
plt.savefig("pca8_scree_plot.png", dpi=150)
print("\nSaved scree plot to pca8_scree_plot.png")

# ---------------------------------------------------------
# 5. 2D visualization using PC1 vs PC2 (colored by class)
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
for label, name, color in [(0, "malignant", "red"), (1, "benign", "green")]:
    mask = y == label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=name, alpha=0.6, c=color)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - First 2 Components")
plt.legend()
plt.tight_layout()
plt.savefig("pca8_pc1_pc2.png", dpi=150)
print("Saved 2D PCA plot to pca8_pc1_pc2.png")

# ---------------------------------------------------------
# 6. Train a classifier on the 8 PCA components
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=5000, random_state=42)

cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
print(f"\nCV accuracy (8 PCA components): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["malignant", "benign"]))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------
# 7. Which original features load most on each PC
# ---------------------------------------------------------
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(N_COMPONENTS)],
    index=feature_names,
)
print("\nTop 3 contributing features per component:")
for pc in loadings.columns:
    top_feats = loadings[pc].abs().sort_values(ascending=False).head(3)
    print(f"  {pc}: {list(top_feats.index)}")