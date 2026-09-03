# NOTE: Historical exploratory run/script — unseeded split with data leakage in PCA/scaler fitting and single-qubit readout. Not a valid final result. See model_metadata.txt and README.md for current numbers.
"""
Advanced experiment: try PennyLane's built-in StronglyEntanglingLayers
template (a more sophisticated ansatz than our hand-rolled ring-CNOT one),
and push epochs higher on the best config found so far (8 qubits, 3 layers).
"""
import os
import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data = load_breast_cancer()
X, y_raw = data.data, data.target
y = np.where(y_raw == 0, -1, 1)
X_scaled = StandardScaler().fit_transform(X)

def run_strongly_entangling(n_qubits, n_layers, epochs=60, batch_size=20, seed=42):
    pca = PCA(n_components=n_qubits)
    X_reduced = pca.fit_transform(X_scaled)
    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_angles = angle_scaler.fit_transform(X_reduced)
    X_train, X_test, y_train, y_test = train_test_split(
        X_angles, y, test_size=0.2, random_state=42
    )

    dev = qml.device("default.qubit", wires=n_qubits)

    def encode(features):
        for i in range(n_qubits):
            qml.RY(features[i], wires=i)

    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights, features):
        encode(features)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    def predict(weights, features):
        return circuit(weights, features)

    def cost(weights, X, y):
        preds = [predict(weights, x) for x in X]
        return np.mean((y - np.stack(preds)) ** 2)

    def accuracy(weights, X, y):
        preds = np.sign([predict(weights, x) for x in X])
        return accuracy_score(y, preds)

    np.random.seed(seed)
    weights = 0.01 * np.random.randn(*shape, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.1)

    start = time.time()
    history = []
    for epoch in range(epochs):
        batch_idx = np.random.choice(len(X_train), min(batch_size, len(X_train)), replace=False)
        X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
        weights = opt.step(lambda w: cost(w, X_batch, y_batch), weights)
        if (epoch + 1) % 10 == 0:
            test_acc = accuracy(weights, X_test, y_test)
            history.append((epoch + 1, test_acc))
    elapsed = time.time() - start

    train_acc = accuracy(weights, X_train, y_train)
    test_acc = accuracy(weights, X_test, y_test)
    return train_acc, test_acc, elapsed, history

print("Running StronglyEntanglingLayers ansatz, 8 qubits, 3 layers, 60 epochs...")
train_acc, test_acc, elapsed, history = run_strongly_entangling(8, 3, epochs=60)

print(f"\nFinal train acc: {train_acc:.3f}")
print(f"Final test acc:  {test_acc:.3f}")
print(f"Time: {elapsed:.1f}s")
print("\nEpoch-by-epoch test accuracy:")
for epoch, acc in history:
    print(f"  Epoch {epoch:3d}: {acc:.3f}")

with open(os.path.join(BASE_DIR, "advanced_results.txt"), "w") as f:
    f.write("StronglyEntanglingLayers ansatz, 8 qubits, 3 layers, 60 epochs\n")
    f.write(f"Final train acc: {train_acc:.3f}\n")
    f.write(f"Final test acc:  {test_acc:.3f}\n")
    f.write(f"Time: {elapsed:.1f}s\n\n")
    f.write("Epoch-by-epoch test accuracy:\n")
    for epoch, acc in history:
        f.write(f"  Epoch {epoch:3d}: {acc:.3f}\n")
    f.write("\nComparison:\n")
    f.write("  Custom ring-CNOT ansatz (8q,3L, 30 epochs): 93.9%\n")
    f.write(f"  StronglyEntanglingLayers (8q,3L, 60 epochs): {test_acc*100:.1f}%\n")
    f.write("  Classical Logistic Regression: 98.2%\n")

print("\nSaved to advanced_results.txt")