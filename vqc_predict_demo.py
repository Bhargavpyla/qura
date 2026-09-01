"""
Demo-ready prediction interface. Loads pre-trained models (no retraining!)
and exposes a single predict() function for both VQC and classical model.

For your UI teammate: import `predict_both(raw_features)` where raw_features
is a list/array of the 30 original Breast Cancer Wisconsin feature values,
in the same order as sklearn's load_breast_cancer().feature_names.

Run this file directly to see a demo prediction on a real test sample.
"""
import pennylane as qml
from pennylane import numpy as np
import numpy as onp
import joblib
from sklearn.datasets import load_breast_cancer

N_QUBITS = 8
N_LAYERS = 3

# -------------------- Load saved pipeline + models --------------------
scaler = joblib.load("saved_scaler.joblib")
pca = joblib.load("saved_pca.joblib")
angle_scaler = joblib.load("saved_angle_scaler.joblib")
classical_model = joblib.load("saved_classical_model.joblib")
vqc_weights = np.array(onp.load("saved_vqc_weights.npy"), requires_grad=False)

dev = qml.device("default.qubit", wires=N_QUBITS)

def encode(features):
    for i in range(N_QUBITS):
        qml.RY(features[i], wires=i)

@qml.qnode(dev)
def circuit(weights, features):
    encode(features)
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


def predict_classical(raw_features):
    """raw_features: array-like of 30 values (original feature scale)."""
    X = onp.array(raw_features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = classical_model.predict(X_scaled)[0]        # 1 = malignant(0)->relabeled, careful see note
    proba = classical_model.predict_proba(X_scaled)[0]
    label = "Benign" if pred == 1 else "Malignant"
    confidence = max(proba)
    return label, confidence


def predict_vqc(raw_features):
    """raw_features: array-like of 30 values (original feature scale)."""
    X = onp.array(raw_features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    X_reduced = pca.transform(X_scaled)
    X_angles = angle_scaler.transform(X_reduced)[0]
    score = float(circuit(vqc_weights, X_angles))       # in [-1, 1]
    label = "Benign" if score > 0 else "Malignant"
    confidence = (abs(score) + 1) / 2                    # rough confidence mapping to [0.5, 1]
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
    # Demo: grab a couple of real samples from the dataset and predict
    data = load_breast_cancer()
    print("NOTE: sklearn labels 0=malignant, 1=benign in the raw target,")
    print("      but predict_classical relabels internally -- trust the printed label.\n")

    for idx in [0, 1, 20, 50]:
        raw_features = data.data[idx]
        true_label = "Benign" if data.target[idx] == 1 else "Malignant"
        result = predict_both(raw_features)
        print(f"Sample #{idx} (true label: {true_label})")
        print(f"  Classical -> {result['classical']['label']} (confidence: {result['classical']['confidence']})")
        print(f"  Quantum   -> {result['quantum']['label']} (confidence: {result['quantum']['confidence']}, raw score: {result['quantum']['raw_score']})")
        print()
