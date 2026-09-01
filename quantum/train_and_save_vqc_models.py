"""
Final training run: trains and SAVES both models (VQC + classical baseline)
so the demo can load pre-trained weights instantly instead of retraining live.

Also saves the fitted preprocessing objects (scaler, PCA, angle scaler) since
new demo inputs need to go through the exact same transformation pipeline.
"""
import os
import pennylane as qml
from pennylane import numpy as np
import numpy as onp  # plain numpy for saving (pennylane's numpy wraps autograd)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

N_QUBITS = 8
N_LAYERS = 3
EPOCHS = 60
BATCH_SIZE = 20

data = load_breast_cancer()
X, y_raw = data.data, data.target
y = np.where(y_raw == 0, -1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=N_QUBITS)
X_reduced = pca.fit_transform(X_scaled)

angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
X_angles = angle_scaler.fit_transform(X_reduced)

X_train, X_test, y_train, y_test = train_test_split(
    X_angles, y, test_size=0.2, random_state=42
)
# keep matching classical split (same indices, pre-PCA scaled features)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------- Classical baseline --------------------
print("Training classical baseline...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_c, y_train_c)
classical_acc = clf.score(X_test_c, y_test_c)
print(f"Classical test accuracy: {classical_acc:.3f}")

# -------------------- VQC (StronglyEntanglingLayers) --------------------
print("\nTraining VQC (this takes ~1-2 minutes)...")
dev = qml.device("default.qubit", wires=N_QUBITS)

def encode(features):
    for i in range(N_QUBITS):
        qml.RY(features[i], wires=i)

shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)

@qml.qnode(dev)
def circuit(weights, features):
    encode(features)
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

def predict_raw(weights, features):
    return circuit(weights, features)

def cost(weights, X, y):
    preds = [predict_raw(weights, x) for x in X]
    return np.mean((y - np.stack(preds)) ** 2)

def accuracy(weights, X, y):
    preds = np.sign([predict_raw(weights, x) for x in X])
    return accuracy_score(y, preds)

onp.random.seed(42)
weights = 0.01 * np.random.randn(*shape, requires_grad=True)
opt = qml.AdamOptimizer(stepsize=0.1)

start = time.time()
for epoch in range(EPOCHS):
    batch_idx = onp.random.choice(len(X_train), BATCH_SIZE, replace=False)
    X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
    weights = opt.step(lambda w: cost(w, X_batch, y_batch), weights)
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS}")
elapsed = time.time() - start

vqc_train_acc = accuracy(weights, X_train, y_train)
vqc_test_acc = accuracy(weights, X_test, y_test)
print(f"\nVQC train accuracy: {vqc_train_acc:.3f}")
print(f"VQC test accuracy:  {vqc_test_acc:.3f}")
print(f"Training time: {elapsed:.1f}s")

# -------------------- SAVE EVERYTHING --------------------
print("\nSaving models and preprocessing pipeline...")

joblib.dump(scaler, os.path.join(BASE_DIR, "vqc_scaler.joblib"))
joblib.dump(pca, os.path.join(BASE_DIR, "vqc_pca.joblib"))
joblib.dump(angle_scaler, os.path.join(BASE_DIR, "vqc_angle_scaler.joblib"))
joblib.dump(clf, os.path.join(BASE_DIR, "quantum_domain_classical_baseline.joblib"))
onp.save(os.path.join(BASE_DIR, "vqc_weights.npy"), onp.array(weights))

# save a small metadata file too
with open(os.path.join(BASE_DIR, "model_metadata.txt"), "w") as f:
    f.write(f"N_QUBITS={N_QUBITS}\n")
    f.write(f"N_LAYERS={N_LAYERS}\n")
    f.write(f"classical_test_acc={classical_acc:.4f}\n")
    f.write(f"vqc_test_acc={vqc_test_acc:.4f}\n")
    f.write(f"vqc_train_acc={vqc_train_acc:.4f}\n")
    f.write("feature_names=" + ",".join(data.feature_names) + "\n")

print("Done. Saved files:")
print("  vqc_scaler.joblib")
print("  vqc_pca.joblib")
print("  vqc_angle_scaler.joblib")
print("  quantum_domain_classical_baseline.joblib")
print("  vqc_weights.npy")
print("  model_metadata.txt")
