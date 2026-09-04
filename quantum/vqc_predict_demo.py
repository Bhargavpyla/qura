"""
Unified Hybrid Quantum-Classical Diagnostic Engine for QURA.
Exposes modular inference for:
  1. 8-Qubit Variational Quantum Classifier (VQC) with data re-uploading and multi-qubit Pauli-Z expectation readout.
  2. Fair Baseline Classical Model (Logistic Regression on 8 PCA features - 99.12% benchmark accuracy).
  3. Full Baseline Classical Model (Logistic Regression on all 30 features - 97.37% benchmark accuracy).
  4. Tri-Engine Consensus & Calibration Analyzer.

Usage:
  from vqc_predict_demo import predict_all, predict_both, predict_quantum
  result = predict_all(raw_features_30)
"""

import os
import time
import joblib
import pennylane as qml
from pennylane import numpy as np
import numpy as onp
from sklearn.datasets import load_breast_cancer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

N_QUBITS = 8
N_LAYERS = 3

# -------------------- Load Pre-trained Transformers & Artifacts --------------------
scaler = joblib.load(os.path.join(BASE_DIR, "vqc_scaler.joblib"))
pca = joblib.load(os.path.join(BASE_DIR, "vqc_pca.joblib"))
angle_scaler = joblib.load(os.path.join(BASE_DIR, "vqc_angle_scaler.joblib"))
classical_full_model = joblib.load(os.path.join(BASE_DIR, "quantum_domain_classical_baseline.joblib"))
fair_pca8_model = joblib.load(os.path.join(BASE_DIR, "fair_baseline_pca8_lr.joblib"))

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


def preprocess_features(raw_features):
    """
    Transforms 30 raw feature values through standard scaler, PCA (8 components),
    and angle scaling [0, pi] for quantum gate rotations.
    """
    X = onp.array(raw_features, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    X_angles = angle_scaler.transform(X_pca)[0]
    return X_scaled, X_pca, X_angles


def predict_quantum(raw_features):
    """
    Evaluates 8-qubit VQC on input features.
    Returns dictionary with label, calibrated confidence, raw score, qubit expectations,
    PCA components, angles, and latency.
    """
    start_time = time.time()
    _, X_pca, X_angles = preprocess_features(raw_features)

    expectations = np.array(circuit(layer_weights, X_angles))
    score = float(np.dot(out_weights, expectations) + bias)

    label = "Benign" if score >= 0 else "Malignant"
    # Sigmoidal calibrated confidence
    confidence = float(1.0 / (1.0 + onp.exp(-abs(score))))
    latency_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "raw_score": round(score, 4),
        "latency_ms": latency_ms,
        "qubit_expectations": [round(float(e), 4) for e in expectations],
        "pca_components": [round(float(p), 4) for p in X_pca[0]],
        "quantum_angles_rad": [round(float(a), 4) for a in X_angles],
    }


def predict_classical_full(raw_features):
    """
    Evaluates Full Classical Logistic Regression on all 30 scaled features.
    """
    X_scaled, _, _ = preprocess_features(raw_features)
    pred_raw = classical_full_model.predict(X_scaled)[0]
    proba = classical_full_model.predict_proba(X_scaled)[0]
    label = "Benign" if pred_raw == 1 else "Malignant"
    conf = float(max(proba))

    # Classes: -1 is Malignant, 1 is Benign
    p_malig = float(proba[0] if classical_full_model.classes_[0] == -1 else proba[1])
    p_benign = float(proba[1] if classical_full_model.classes_[1] == 1 else proba[0])

    return {
        "label": label,
        "confidence": round(conf * 100, 2),
        "probability_malignant": round(p_malig * 100, 2),
        "probability_benign": round(p_benign * 100, 2),
    }


def predict_classical_fair(raw_features):
    """
    Evaluates Fair Baseline Classical Logistic Regression on 8 PCA features (apples-to-apples).
    """
    _, X_pca, _ = preprocess_features(raw_features)
    pred_raw = fair_pca8_model.predict(X_pca)[0]
    proba = fair_pca8_model.predict_proba(X_pca)[0]
    label = "Benign" if pred_raw == 1 else "Malignant"
    conf = float(max(proba))

    p_malig = float(proba[0] if fair_pca8_model.classes_[0] == -1 else proba[1])
    p_benign = float(proba[1] if fair_pca8_model.classes_[1] == 1 else proba[0])

    return {
        "label": label,
        "confidence": round(conf * 100, 2),
        "probability_malignant": round(p_malig * 100, 2),
        "probability_benign": round(p_benign * 100, 2),
    }


def predict_all(raw_features):
    """
    Executes unified tri-engine inference:
      - 8-Qubit VQC (8 PCA features)
      - Fair Classical Baseline (8 PCA features)
      - Full Classical Baseline (30 features)
    Computes unanimous and pairwise consensus metrics.
    """
    total_start = time.time()
    q_result = predict_quantum(raw_features)
    c_full = predict_classical_full(raw_features)
    c_fair = predict_classical_fair(raw_features)
    total_latency_ms = round((time.time() - total_start) * 1000, 2)

    q_label = q_result["label"]
    cf_label = c_full["label"]
    fair_label = c_fair["label"]

    unanimous_consensus = (q_label == cf_label == fair_label)
    quantum_fair_agreement = (q_label == fair_label)
    quantum_full_agreement = (q_label == cf_label)

    if unanimous_consensus:
        status_text = "Unanimous Consensus (3/3)"
        status_desc = f"All three models (Quantum VQC, Fair PCA-8, and Full 30-Feat) unanimously agree on {q_label} pathology."
    elif quantum_fair_agreement:
        status_text = "Feature-Matched Consensus (2/3)"
        status_desc = f"Quantum VQC and Fair PCA-8 Baseline agree on {q_label} (Full 30-Feat baseline predicts {cf_label})."
    elif quantum_full_agreement:
        status_text = "Quantum-Full Agreement (2/3)"
        status_desc = f"Quantum VQC and Full 30-Feat baseline agree on {q_label} (Fair PCA-8 predicts {fair_label})."
    else:
        status_text = "Model Divergence"
        status_desc = f"Models diverge: Quantum ({q_label}), Fair PCA-8 ({fair_label}), Full 30-Feat ({cf_label})."

    return {
        "consensus": unanimous_consensus,
        "quantum_fair_agreement": quantum_fair_agreement,
        "consensus_status": status_text,
        "consensus_desc": status_desc,
        "primary_label": q_label,
        "quantum": q_result,
        "classical": c_full,
        "fair_classical": c_fair,
        "total_latency_ms": total_latency_ms,
    }


# Backwards compatibility wrappers
def predict_classical(raw_features):
    res = predict_classical_full(raw_features)
    return res["label"], res["confidence"] / 100.0


def predict_vqc(raw_features):
    res = predict_quantum(raw_features)
    return res["label"], res["confidence"] / 100.0, res["raw_score"]


def predict_both(raw_features):
    all_res = predict_all(raw_features)
    return {
        "classical": {"label": all_res["classical"]["label"], "confidence": round(all_res["classical"]["confidence"] / 100.0, 3)},
        "fair_classical": {"label": all_res["fair_classical"]["label"], "confidence": round(all_res["fair_classical"]["confidence"] / 100.0, 3)},
        "quantum": {"label": all_res["quantum"]["label"], "confidence": round(all_res["quantum"]["confidence"] / 100.0, 3), "raw_score": all_res["quantum"]["raw_score"]},
    }


if __name__ == "__main__":
    data = load_breast_cancer()
    print("Testing unified diagnostic engine on benchmark patient samples...\n")

    for idx in [0, 1, 20, 50]:
        raw_features = data.data[idx]
        true_label = "Benign" if data.target[idx] == 1 else "Malignant"
        result = predict_all(raw_features)
        print(f"Sample #{idx} (Ground Truth: {true_label})")
        print(f"  Quantum VQC        -> {result['quantum']['label']} (Confidence: {result['quantum']['confidence']}%, Score: {result['quantum']['raw_score']})")
        print(f"  Fair Baseline PCA-8-> {result['fair_classical']['label']} (Confidence: {result['fair_classical']['confidence']}%)")
        print(f"  Full Baseline 30-F -> {result['classical']['label']} (Confidence: {result['classical']['confidence']}%)")
        print(f"  Consensus Status   -> {result['consensus_status']}")
        print()
