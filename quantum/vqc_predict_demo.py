"""
Demo-ready prediction interface. Loads pre-trained models (no retraining!)
and exposes a single predict() function for both VQC and classical model.

For your UI teammate: import `predict_both(raw_features)` where raw_features
is a list/array of the 30 original Breast Cancer Wisconsin feature values,
in the same order as sklearn's load_breast_cancer().feature_names.

Run this file directly to see a demo prediction on real test samples.
"""
import os
import pennylane as qml
from pennylane import numpy as np
import numpy as onp
import joblib
from sklearn.datasets import load_breast_cancer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

N_QUBITS = 8
N_LAYERS = 3

# -------------------- Load saved pipeline + models --------------------
scaler = joblib.load(os.path.join(BASE_DIR, "vqc_scaler.joblib"))
pca = joblib.load(os.path.join(BASE_DIR, "vqc_pca.joblib"))
angle_scaler = joblib.load(os.path.join(BASE_DIR, "vqc_angle_scaler.joblib"))
classical_model = joblib.load(os.path.join(BASE_DIR, "quantum_domain_classical_baseline.joblib"))

# Load quantum model parameters (supports .npz with full multi-qubit weights or fallback .npy)
weights_npz_path = os.path.join(BASE_DIR, "vqc_weights.npz")
if os.path.exists(weights_npz_path):
    data_weights = onp.load(weights_npz_path)
    layer_weights = np.array(data_weights["layer_weights"], requires_grad=False)
    out_weights = np.array(data_weights["out_weights"], requires_grad=False)
    bias = float(data_weights["bias"])
else:
    layer_weights = np.array(onp.load(os.path.join(BASE_DIR, "vqc_weights.npy")), requires_grad=False)
    out_weights = np.array(np.ones(N_QUBITS) / N_QUBITS, requires_grad=False)
    bias = 0.0

try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
except Exception:
    dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev)
def circuit(weights, features):
    for layer_idx in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(features[i], wires=i)
        w = weights[layer_idx: layer_idx + 1]
        qml.StronglyEntanglingLayers(w, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


def predict_classical(raw_features):
    """raw_features: array-like of 30 values (original feature scale)."""
    X = onp.array(raw_features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = classical_model.predict(X_scaled)[0]
    proba = classical_model.predict_proba(X_scaled)[0]
    label = "Benign" if pred == 1 else "Malignant"
    confidence = float(max(proba))
    return label, confidence


def predict_vqc(raw_features):
    """raw_features: array-like of 30 values (original feature scale)."""
    X = onp.array(raw_features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    X_reduced = pca.transform(X_scaled)
    X_angles = angle_scaler.transform(X_reduced)[0]

    expectations = np.array(circuit(layer_weights, X_angles))
    score = float(np.dot(out_weights, expectations) + bias)

    label = "Benign" if score >= 0 else "Malignant"
    # Sigmoidal calibrated confidence
    confidence = float(1.0 / (1.0 + onp.exp(-abs(score))))
    return label, confidence, score


def predict_both(raw_features):
    """Convenience wrapper: returns both predictions in one call."""
    c_label, c_conf = predict_classical(raw_features)
    q_label, q_conf, q_score = predict_vqc(raw_features)
    return {
        "classical": {"label": c_label, "confidence": round(float(c_conf), 3)},
        "quantum": {"label": q_label, "confidence": round(float(q_conf), 3), "raw_score": round(q_score, 3)},
    }


if __name__ == "__main__":
    data = load_breast_cancer()
    print("Testing pre-trained models on real patient samples...\n")

    for idx in [0, 1, 20, 50]:
        raw_features = data.data[idx]
        true_label = "Benign" if data.target[idx] == 1 else "Malignant"
        result = predict_both(raw_features)
        print(f"Sample #{idx} (Ground Truth: {true_label})")
        print(f"  Classical -> {result['classical']['label']} (Confidence: {result['classical']['confidence']*100:.1f}%)")
        print(f"  Quantum   -> {result['quantum']['label']} (Confidence: {result['quantum']['confidence']*100:.1f}%, Score: {result['quantum']['raw_score']})")
        print()
