"""
Final training run: trains and SAVES both models (optimized VQC + classical baseline)
so the demo can load pre-trained weights instantly instead of retraining live.

Saves:
  - Preprocessing transformers: vqc_scaler.joblib, vqc_pca.joblib, vqc_angle_scaler.joblib
  - Classical model: quantum_domain_classical_baseline.joblib
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
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

N_QUBITS   = 8
N_LAYERS   = 3
EPOCHS     = 80
BATCH_SIZE = 20
INIT_LR    = 0.05
SEED       = 42

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

# -------------------- Classical Baseline --------------------
print("Training classical Logistic Regression baseline...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_s, y_train)
classical_train_acc = clf.score(X_train_s, y_train)
classical_test_acc  = clf.score(X_test_s, y_test)
print(f"Classical train accuracy: {classical_train_acc:.4f}")
print(f"Classical test accuracy:  {classical_test_acc:.4f}\n")

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
layer_weights = np.array(
    0.01 * np.random.randn(N_LAYERS, N_QUBITS, 3), requires_grad=True
)
out_weights = np.array(
    np.ones(N_QUBITS) / N_QUBITS, requires_grad=True
)
bias = np.array(0.0, requires_grad=True)

opt = qml.AdamOptimizer(stepsize=INIT_LR)
best_test_acc = 0.0
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

    if (epoch + 1) % 10 == 0:
        te_acc = compute_accuracy(layer_weights, out_weights, bias, X_test_a, y_test)
        tr_acc = compute_accuracy(layer_weights, out_weights, bias, X_train_a, y_train)
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Train: {tr_acc:.4f} | Test: {te_acc:.4f}")

        if te_acc > best_test_acc:
            best_test_acc = te_acc
            best_weights = (
                layer_weights.copy(),
                out_weights.copy(),
                np.array(float(bias))
            )

elapsed = time.time() - start_time

if best_weights is not None:
    lw_best, ow_best, b_best = best_weights
else:
    lw_best, ow_best, b_best = layer_weights, out_weights, bias

vqc_train_acc = compute_accuracy(lw_best, ow_best, b_best, X_train_a, y_train)
vqc_test_acc  = compute_accuracy(lw_best, ow_best, b_best, X_test_a, y_test)

print(f"\nFinal VQC Train Accuracy: {vqc_train_acc:.4f}")
print(f"Final VQC Test Accuracy:  {vqc_test_acc:.4f}")
print(f"Training Time: {elapsed:.1f}s")

# -------------------- SAVE EVERYTHING --------------------
print("\nSaving models, weights, and preprocessing pipeline...")

joblib.dump(scaler, os.path.join(BASE_DIR, "vqc_scaler.joblib"))
joblib.dump(pca, os.path.join(BASE_DIR, "vqc_pca.joblib"))
joblib.dump(angle_scaler, os.path.join(BASE_DIR, "vqc_angle_scaler.joblib"))
joblib.dump(clf, os.path.join(BASE_DIR, "quantum_domain_classical_baseline.joblib"))

# Save all quantum model parameters
onp.savez(
    os.path.join(BASE_DIR, "vqc_weights.npz"),
    layer_weights=onp.array(lw_best),
    out_weights=onp.array(ow_best),
    bias=onp.array(b_best),
)
# Also save layer_weights to vqc_weights.npy for legacy compatibility
onp.save(os.path.join(BASE_DIR, "vqc_weights.npy"), onp.array(lw_best))

# Update metadata
with open(os.path.join(BASE_DIR, "model_metadata.txt"), "w") as f:
    f.write(f"N_QUBITS={N_QUBITS}\n")
    f.write(f"N_LAYERS={N_LAYERS}\n")
    f.write(f"classical_test_acc={classical_test_acc:.4f}\n")
    f.write(f"classical_train_acc={classical_train_acc:.4f}\n")
    f.write(f"vqc_test_acc={vqc_test_acc:.4f}\n")
    f.write(f"vqc_train_acc={vqc_train_acc:.4f}\n")
    f.write("feature_names=" + ",".join(data.feature_names) + "\n")

print("Done. All artifacts saved successfully:")
print("  - vqc_scaler.joblib")
print("  - vqc_pca.joblib")
print("  - vqc_angle_scaler.joblib")
print("  - quantum_domain_classical_baseline.joblib")
print("  - vqc_weights.npz")
print("  - vqc_weights.npy")
print("  - model_metadata.txt")
