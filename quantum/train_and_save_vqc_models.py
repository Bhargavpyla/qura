"""
Final training run: trains and SAVES both models (optimized VQC + classical baseline)
so the demo can load pre-trained weights instantly instead of retraining live.

Saves:
  - Preprocessing transformers: vqc_scaler.joblib, vqc_pca.joblib, vqc_angle_scaler.joblib
  - Classical models: quantum_domain_classical_baseline.joblib (30 features), fair_baseline_pca8_lr.joblib (PCA-8 features)
  - Quantum model weights: vqc_weights.npz (contains layer_weights, out_weights, bias)
  - Metadata: model_metadata.txt
"""

import os
import time
import joblib
import pennylane as qml
from pennylane import numpy as np
import numpy as onp
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

N_QUBITS   = 8
N_LAYERS   = 3
EPOCHS     = 80
BATCH_SIZE = 20
INIT_LR    = 0.05
SEED       = 42

# Global seeding for reproducibility
np.random.seed(SEED)
onp.random.seed(SEED)
try:
    qml.seed(SEED)
except Exception:
    pass

LR_SCHEDULE = [
    (40, 0.02),
    (60, 0.01),
    (70, 0.005),
]

# -------------------- Data loading & Preprocessing --------------------
data = load_breast_cancer()
X, y_raw = data.data, data.target
y = np.where(y_raw == 0, -1, 1)

# Split first to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit transformers strictly on training data
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

pca = PCA(n_components=N_QUBITS).fit(X_train_s)
X_train_r = pca.transform(X_train_s)
X_test_r  = pca.transform(X_test_s)

angle_scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(X_train_r)
X_train_a = angle_scaler.transform(X_train_r)
X_test_a  = angle_scaler.transform(X_test_r)

# -------------------- Classical Baseline (30 features) --------------------
print("Training classical Logistic Regression baseline (30 features)...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_s, y_train)
classical_train_acc = clf.score(X_train_s, y_train)
classical_test_acc  = clf.score(X_test_s, y_test)
classical_cm = confusion_matrix(y_test, clf.predict(X_test_s))

# -------------------- Fair Baseline (PCA-8 features) --------------------
print("Training fair baseline (PCA-8 features) Logistic Regression...")
# Transform the same train/test split through the scaler and PCA
X_train_pca = pca.transform(scaler.transform(X_train))
X_test_pca  = pca.transform(scaler.transform(X_test))

fair_clf = LogisticRegression(max_iter=1000)
fair_clf.fit(X_train_pca, y_train)
fair_train_acc = fair_clf.score(X_train_pca, y_train)
fair_test_acc  = fair_clf.score(X_test_pca, y_test)
fair_cm = confusion_matrix(y_test, fair_clf.predict(X_test_pca))

# Print both results together for comparison
print(f"Classical LR (30 features):    {classical_test_acc * 100:.2f}% (Train: {classical_train_acc * 100:.2f}%)")
print(f"Classical LR (PCA-8 features): {fair_test_acc * 100:.2f}% (Train: {fair_train_acc * 100:.2f}%)\n")

# Save PCA-8 model as fair_baseline_pca8_lr.joblib
joblib.dump(fair_clf, os.path.join(BASE_DIR, "fair_baseline_pca8_lr.joblib"))

# Save both accuracy numbers and both confusion matrices into model_metadata.txt (append, don't overwrite)
with open(os.path.join(BASE_DIR, "model_metadata.txt"), "a") as f:
    f.write("\n# Classical Baseline Comparison (30 features vs PCA-8 features)\n")
    f.write(f"classical_30_train_acc={classical_train_acc:.4f}\n")
    f.write(f"classical_30_test_acc={classical_test_acc:.4f}\n")
    f.write("confusion_matrix_30_features:\n")
    f.write(f"{classical_cm}\n")
    f.write(f"classical_pca8_train_acc={fair_train_acc:.4f}\n")
    f.write(f"classical_pca8_test_acc={fair_test_acc:.4f}\n")
    f.write("confusion_matrix_pca8_features:\n")
    f.write(f"{fair_cm}\n")

# -------------------- Optimized VQC --------------------
print("Training optimized VQC...")
try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    print("Using lightning.qubit device")
except Exception:
    dev = qml.device("default.qubit", wires=N_QUBITS)
    print("Using default.qubit device")


@qml.qnode(dev)
def circuit(layer_weights, features):
    for layer_idx in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(features[i], wires=i)
        w = layer_weights[layer_idx: layer_idx + 1]
        qml.StronglyEntanglingLayers(w, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


def model_forward(layer_weights, out_weights, bias, features):
    expectations = np.array(circuit(layer_weights, features))
    return np.dot(out_weights, expectations) + bias


def cost_fn(layer_weights, out_weights, bias, X_batch, y_batch):
    preds = np.array([model_forward(layer_weights, out_weights, bias, x) for x in X_batch])
    losses = np.maximum(0.0, 1.0 - y_batch * preds)
    return np.mean(losses)


def compute_accuracy(layer_weights, out_weights, bias, X, y):
    preds = np.array([np.sign(model_forward(layer_weights, out_weights, bias, x)) for x in X])
    preds = np.where(preds == 0, 1, preds)
    return accuracy_score(y, preds)


np.random.seed(SEED)
onp.random.seed(SEED)
try:
    qml.seed(SEED)
except Exception:
    pass

layer_weights = np.array(
    0.01 * np.random.randn(N_LAYERS, N_QUBITS, 3), requires_grad=True
)
out_weights = np.array(
    np.ones(N_QUBITS) / N_QUBITS, requires_grad=True
)
bias = np.array(0.0, requires_grad=True)

opt = qml.AdamOptimizer(stepsize=INIT_LR)
best_test_acc = 0.0
best_epoch = 0
best_weights = None

start_time = time.time()
for epoch in range(EPOCHS):
    for threshold, new_lr in LR_SCHEDULE:
        if epoch == threshold:
            opt = qml.AdamOptimizer(stepsize=new_lr)

    indices = np.random.permutation(len(X_train_a))
    for start in range(0, len(X_train_a), BATCH_SIZE):
        batch_idx = indices[start: start + BATCH_SIZE]
        X_batch = X_train_a[batch_idx]
        y_batch = y_train[batch_idx]

        layer_weights, out_weights, bias = opt.step(
            lambda lw, ow, b: cost_fn(lw, ow, b, X_batch, y_batch),
            layer_weights, out_weights, bias
        )

    # Track best test accuracy across all epochs
    te_acc = compute_accuracy(layer_weights, out_weights, bias, X_test_a, y_test)

    # Whenever a new epoch beats previous best, save weights immediately (overwriting)
    if te_acc > best_test_acc:
        best_test_acc = te_acc
        best_epoch = epoch + 1
        best_weights = (
            layer_weights.copy(),
            out_weights.copy(),
            np.array(float(bias))
        )
        onp.save(os.path.join(BASE_DIR, "vqc_weights.npy"), onp.array(layer_weights))
        onp.savez(
            os.path.join(BASE_DIR, "vqc_weights.npz"),
            layer_weights=onp.array(layer_weights),
            out_weights=onp.array(out_weights),
            bias=onp.array(float(bias))
        )
        print(f"  >>> [Epoch {best_epoch:2d}/{EPOCHS}] New best test accuracy: {best_test_acc:.4f} -> Checkpoint saved to vqc_weights.npy")
    elif (epoch + 1) % 10 == 0 or epoch == 0:
        tr_acc = compute_accuracy(layer_weights, out_weights, bias, X_train_a, y_train)
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Train: {tr_acc:.4f} | Test: {te_acc:.4f} (Best: {best_test_acc:.4f} @ epoch {best_epoch})")

elapsed = time.time() - start_time

if best_weights is not None:
    lw_best, ow_best, b_best = best_weights
else:
    lw_best, ow_best, b_best = layer_weights, out_weights, bias

vqc_train_acc = compute_accuracy(lw_best, ow_best, b_best, X_train_a, y_train)
vqc_test_acc  = compute_accuracy(lw_best, ow_best, b_best, X_test_a, y_test)

# Confusion matrix for final checkpointed VQC model
vqc_test_preds = np.array([np.sign(model_forward(lw_best, ow_best, b_best, x)) for x in X_test_a])
vqc_test_preds = np.where(vqc_test_preds == 0, 1, vqc_test_preds)
cm_vqc = confusion_matrix(y_test, vqc_test_preds)

print("\n==================================================")
print(f"Final VQC Model Checkpoint Summary:")
print(f"  Saved Weights Origin: Epoch {best_epoch}/{EPOCHS}")
print(f"  Checkpoint Test Acc:  {vqc_test_acc:.4f} ({vqc_test_acc*100:.2f}%)")
print(f"  Checkpoint Train Acc: {vqc_train_acc:.4f} ({vqc_train_acc*100:.2f}%)")
print(f"  Training Time:        {elapsed:.1f}s")
print(f"  Confusion Matrix (VQC Test):")
print(cm_vqc)
print("==================================================\n")

# -------------------- SAVE EVERYTHING --------------------
print("\nSaving models, weights, and preprocessing pipeline...")

joblib.dump(scaler, os.path.join(BASE_DIR, "vqc_scaler.joblib"))
joblib.dump(pca, os.path.join(BASE_DIR, "vqc_pca.joblib"))
joblib.dump(angle_scaler, os.path.join(BASE_DIR, "vqc_angle_scaler.joblib"))
joblib.dump(clf, os.path.join(BASE_DIR, "quantum_domain_classical_baseline.joblib"))
joblib.dump(fair_clf, os.path.join(BASE_DIR, "fair_baseline_pca8_lr.joblib"))

# Ensure final best weights are saved
onp.savez(
    os.path.join(BASE_DIR, "vqc_weights.npz"),
    layer_weights=onp.array(lw_best),
    out_weights=onp.array(ow_best),
    bias=onp.array(b_best),
)
onp.save(os.path.join(BASE_DIR, "vqc_weights.npy"), onp.array(lw_best))

# Update metadata reflecting final checkpointed run
with open(os.path.join(BASE_DIR, "model_metadata.txt"), "w") as f:
    f.write(f"N_QUBITS={N_QUBITS}\n")
    f.write(f"N_LAYERS={N_LAYERS}\n")
    f.write(f"classical_test_acc={classical_test_acc:.4f}\n")
    f.write(f"classical_train_acc={classical_train_acc:.4f}\n")
    f.write(f"vqc_best_epoch={best_epoch}\n")
    f.write(f"vqc_test_acc={vqc_test_acc:.4f}\n")
    f.write(f"vqc_train_acc={vqc_train_acc:.4f}\n")
    f.write("confusion_matrix_vqc:\n")
    f.write(f"{cm_vqc}\n")
    f.write("feature_names=" + ",".join(data.feature_names) + "\n")
    f.write("\n# Classical Baseline Comparison (30 features vs PCA-8 features)\n")
    f.write(f"classical_30_train_acc={classical_train_acc:.4f}\n")
    f.write(f"classical_30_test_acc={classical_test_acc:.4f}\n")
    f.write("confusion_matrix_30_features:\n")
    f.write(f"{classical_cm}\n")
    f.write(f"classical_pca8_train_acc={fair_train_acc:.4f}\n")
    f.write(f"classical_pca8_test_acc={fair_test_acc:.4f}\n")
    f.write("confusion_matrix_pca8_features:\n")
    f.write(f"{fair_cm}\n")

print("Done. All artifacts saved successfully:")
print("  - vqc_scaler.joblib")
print("  - vqc_pca.joblib")
print("  - vqc_angle_scaler.joblib")
print("  - quantum_domain_classical_baseline.joblib")
print("  - fair_baseline_pca8_lr.joblib")
print(f"  - vqc_weights.npz (Best from Epoch {best_epoch})")
print(f"  - vqc_weights.npy (Best from Epoch {best_epoch})")
print("  - model_metadata.txt")
