# NOTE: Historical exploratory run/script — legacy prototype with data leakage in PCA/scaler fitting, superseded by train_and_save_vqc_models.py. Not a valid final result. See model_metadata.txt and README.md for current numbers.
"""
Hybrid Quantum-Classical VQC for Breast Cancer Detection
==========================================================
Pipeline: load data -> PCA to shrink features -> scale to rotation range
          -> angle-encode into qubits -> variational layer -> train
          -> compare against classical baseline
"""

import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------------
# 1. LOAD + PREPROCESS DATA
# -------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target          # y: 0 = malignant, 1 = benign
y = np.where(y == 0, -1, 1)            # relabel to -1 / +1 (needed for our loss below)

N_QUBITS = 8   # shrink 30 raw features down to 8 -> keeps the circuit small & fast to simulate

# Standardize first (PCA assumes roughly centered/scaled data)
X_scaled = StandardScaler().fit_transform(X)

# PCA: compress 30 features down to N_QUBITS "principal" features
pca = PCA(n_components=N_QUBITS)
X_reduced = pca.fit_transform(X_scaled)

# Rescale PCA output into [0, pi] so it's a sensible rotation angle
angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
X_angles = angle_scaler.fit_transform(X_reduced)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_angles, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------------------
# 2. DEFINE THE QUANTUM CIRCUIT (encoding + variational layers)
# -------------------------------------------------------------------
N_LAYERS = 3   # how many times we stack the trainable variational block

dev = qml.device("default.qubit", wires=N_QUBITS)

def encode(features):
    """Angle-encode each classical feature onto its own qubit."""
    for i in range(N_QUBITS):
        qml.RY(features[i], wires=i)

def variational_block(weights):
    """One trainable layer: per-qubit rotations + a ring of entangling CNOTs."""
    for i in range(N_QUBITS):
        qml.RY(weights[i], wires=i)
    for i in range(N_QUBITS):
        qml.CNOT(wires=[i, (i + 1) % N_QUBITS])   # ring entanglement

@qml.qnode(dev)
def circuit(weights, features):
    encode(features)
    for layer in range(N_LAYERS):
        variational_block(weights[layer])
    return qml.expval(qml.PauliZ(0))   # read out qubit 0's state as our "score"

# -------------------------------------------------------------------
# 3. TRAINING LOOP
# -------------------------------------------------------------------
def predict(weights, features):
    return circuit(weights, features)

def square_loss(labels, predictions):
    return np.mean((labels - predictions) ** 2)

def cost(weights, X, y):
    predictions = [predict(weights, x) for x in X]
    return square_loss(y, np.stack(predictions))

def accuracy(weights, X, y):
    predictions = np.sign([predict(weights, x) for x in X])
    return accuracy_score(y, predictions)

# Initialize random weights: shape = (layers, qubits)
np.random.seed(42)
weights = 0.01 * np.random.randn(N_LAYERS, N_QUBITS, requires_grad=True)

opt = qml.AdamOptimizer(stepsize=0.1)
EPOCHS = 30
BATCH_SIZE = 20

print("Training VQC...")
for epoch in range(EPOCHS):
    # simple mini-batching
    batch_idx = np.random.choice(len(X_train), BATCH_SIZE, replace=False)
    X_batch = X_train[batch_idx]
    y_batch = y_train[batch_idx]

    weights = opt.step(lambda w: cost(w, X_batch, y_batch), weights)

    if (epoch + 1) % 5 == 0:
        train_acc = accuracy(weights, X_train, y_train)
        test_acc = accuracy(weights, X_test, y_test)
        print(f"Epoch {epoch+1:2d} | train acc: {train_acc:.3f} | test acc: {test_acc:.3f}")

final_test_acc = accuracy(weights, X_test, y_test)
print(f"\nFinal VQC test accuracy: {final_test_acc:.3f}")

# -------------------------------------------------------------------
# 4. CLASSICAL BASELINE FOR COMPARISON (this is your "why quantum" evidence)
# -------------------------------------------------------------------
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
classical_acc = clf.score(X_test, y_test)
print(f"Classical Logistic Regression test accuracy: {classical_acc:.3f}")

