"""
Experiment sweep: different qubit counts / layer counts for the VQC,
compared against the classical baseline. Results logged to results.txt
"""
import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

data = load_breast_cancer()
X, y_raw = data.data, data.target
y = np.where(y_raw == 0, -1, 1)
X_scaled = StandardScaler().fit_transform(X)

def run_experiment(n_qubits, n_layers, epochs=30, batch_size=20, seed=42):
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

    def variational_block(weights):
        for i in range(n_qubits):
            qml.RY(weights[i], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

    @qml.qnode(dev)
    def circuit(weights, features):
        encode(features)
        for layer in range(n_layers):
            variational_block(weights[layer])
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
    weights = 0.01 * np.random.randn(n_layers, n_qubits, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.1)

    start = time.time()
    for epoch in range(epochs):
        batch_idx = np.random.choice(len(X_train), min(batch_size, len(X_train)), replace=False)
        X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
        weights = opt.step(lambda w: cost(w, X_batch, y_batch), weights)
    elapsed = time.time() - start

    train_acc = accuracy(weights, X_train, y_train)
    test_acc = accuracy(weights, X_test, y_test)
    return train_acc, test_acc, elapsed

# Classical baseline (fixed, doesn't depend on qubit count)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_c, y_train_c)
classical_acc = clf.score(X_test_c, y_test_c)

configs = [
    (4, 1), (4, 2), (4, 3),
    (6, 1), (6, 2), (6, 3),
    (8, 1), (8, 2), (8, 3),
]

results = []
results.append(f"Classical Logistic Regression baseline accuracy: {classical_acc:.3f}\n")
results.append(f"{'Qubits':<8}{'Layers':<8}{'Train Acc':<12}{'Test Acc':<12}{'Time (s)':<10}")
results.append("-" * 50)

for n_qubits, n_layers in configs:
    train_acc, test_acc, elapsed = run_experiment(n_qubits, n_layers)
    line = f"{n_qubits:<8}{n_layers:<8}{train_acc:<12.3f}{test_acc:<12.3f}{elapsed:<10.1f}"
    results.append(line)
    print(line, flush=True)

with open("results.txt", "w") as f:
    f.write("\n".join(results))

print("\nDone. Results saved to results.txt")