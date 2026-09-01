"""
Flask Backend API for Qura - Hybrid Quantum-Classical Medical Diagnostic UI.
Serves static UI and provides REST endpoints for instant model inference, batch CSV analysis,
and explainability metrics.
"""

import os
import sys
import time
import io
import csv
import joblib
import pennylane as qml
from pennylane import numpy as np
import numpy as onp
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from sklearn.datasets import load_breast_cancer

# Ensure quantum package can be loaded
UI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(UI_DIR, ".."))
QUANTUM_DIR = os.path.join(PROJECT_DIR, "quantum")
sys.path.append(QUANTUM_DIR)

app = Flask(__name__, template_folder=os.path.join(UI_DIR, "templates"), static_folder=os.path.join(UI_DIR, "static"))
CORS(app)

# -------------------- Load Dataset Metadata & Models --------------------
raw_dataset = load_breast_cancer()
FEATURE_NAMES = list(raw_dataset.feature_names)

N_QUBITS = 8
N_LAYERS = 3

print("Loading saved preprocessing transformers and model weights...")
scaler = joblib.load(os.path.join(QUANTUM_DIR, "vqc_scaler.joblib"))
pca = joblib.load(os.path.join(QUANTUM_DIR, "vqc_pca.joblib"))
angle_scaler = joblib.load(os.path.join(QUANTUM_DIR, "vqc_angle_scaler.joblib"))
classical_model = joblib.load(os.path.join(QUANTUM_DIR, "quantum_domain_classical_baseline.joblib"))

# Load quantum model parameters
weights_npz = os.path.join(QUANTUM_DIR, "vqc_weights.npz")
if os.path.exists(weights_npz):
    npz_data = onp.load(weights_npz)
    layer_weights = np.array(npz_data["layer_weights"], requires_grad=False)
    out_weights = np.array(npz_data["out_weights"], requires_grad=False)
    bias = float(npz_data["bias"])
else:
    layer_weights = np.array(onp.load(os.path.join(QUANTUM_DIR, "vqc_weights.npy")), requires_grad=False)
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


# Pre-calculate sample presets for quick UI demonstration
PRESET_SAMPLES = {}
preset_indices = {
    "sample_0_malignant": {"index": 0, "name": "Patient #001 (Malignant)", "badge": "High Risk Malignant"},
    "sample_1_malignant": {"index": 1, "name": "Patient #002 (Malignant)", "badge": "High Risk Malignant"},
    "sample_20_benign": {"index": 20, "name": "Patient #020 (Benign)", "badge": "Low Risk Benign"},
    "sample_50_benign": {"index": 50, "name": "Patient #050 (Benign)", "badge": "Low Risk Benign"},
}

for key, meta in preset_indices.items():
    idx = meta["index"]
    PRESET_SAMPLES[key] = {
        "id": key,
        "name": meta["name"],
        "badge": meta["badge"],
        "ground_truth": "Benign" if raw_dataset.target[idx] == 1 else "Malignant",
        "features": [float(v) for v in raw_dataset.data[idx]],
    }


def single_prediction(raw_features_list):
    """Utility for running combined quantum + classical inference on a 30-feature vector."""
    start_time = time.time()
    X_raw = onp.array(raw_features_list, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X_raw)
    X_pca = pca.transform(X_scaled)
    X_angles = angle_scaler.transform(X_pca)[0]

    # Classical
    c_pred_raw = classical_model.predict(X_scaled)[0]
    c_proba = classical_model.predict_proba(X_scaled)[0]
    c_label = "Benign" if c_pred_raw == 1 else "Malignant"
    c_conf = float(max(c_proba))

    # Quantum
    q_start = time.time()
    expectations = np.array(circuit(layer_weights, X_angles))
    q_score = float(np.dot(out_weights, expectations) + bias)
    q_label = "Benign" if q_score >= 0 else "Malignant"
    q_conf = float(1.0 / (1.0 + onp.exp(-abs(q_score))))
    q_time_ms = round((time.time() - q_start) * 1000, 2)
    total_time_ms = round((time.time() - start_time) * 1000, 2)

    consensus = (c_label == q_label)

    return {
        "consensus": consensus,
        "primary_label": q_label,
        "quantum": {
            "label": q_label,
            "confidence": round(q_conf * 100, 2),
            "raw_score": round(q_score, 4),
            "latency_ms": q_time_ms,
            "qubit_expectations": [round(float(e), 4) for e in expectations],
            "pca_components": [round(float(p), 4) for p in X_pca[0]],
            "quantum_angles_rad": [round(float(a), 4) for a in X_angles],
        },
        "classical": {
            "label": c_label,
            "confidence": round(c_conf * 100, 2),
            "probability_malignant": round(float(c_proba[0] if classical_model.classes_[0] == -1 else c_proba[1]) * 100, 2),
            "probability_benign": round(float(c_proba[1] if classical_model.classes_[1] == 1 else c_proba[0]) * 100, 2),
        },
        "total_latency_ms": total_time_ms,
    }


# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    return jsonify({
        "feature_names": FEATURE_NAMES,
        "n_qubits": N_QUBITS,
        "n_layers": N_LAYERS,
        "accuracy": {
            "vqc_test_acc": 0.9737,
            "vqc_train_acc": 0.9670,
            "classical_test_acc": 0.9737,
            "classical_train_acc": 0.9824,
        },
        "history": [
            {"model": "VQC (6q, 2L Custom)", "accuracy": 0.8860},
            {"model": "VQC (8q, 3L Ring-CNOT)", "accuracy": 0.9386},
            {"model": "VQC (8q, 3L StronglyEntangling)", "accuracy": 0.9474},
            {"model": "VQC (Optimized Pipeline)", "accuracy": 0.9737},
            {"model": "Classical Baseline", "accuracy": 0.9737},
        ],
    })


@app.route("/api/samples", methods=["GET"])
def get_samples():
    return jsonify(PRESET_SAMPLES)


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    features = payload.get("features", [])

    if len(features) != 30:
        return jsonify({"error": f"Expected 30 features, received {len(features)}"}), 400

    result = single_prediction(features)
    return jsonify(result)


@app.route("/api/batch_predict", methods=["POST"])
def batch_predict():
    payload = request.get_json(force=True)
    rows = payload.get("rows", [])

    if not rows:
        return jsonify({"error": "No rows provided"}), 400

    results = []
    malignant_count = 0
    benign_count = 0
    consensus_count = 0

    for i, row in enumerate(rows):
        features = row.get("features", [])
        if len(features) == 30:
            pred = single_prediction(features)
            patient_id = row.get("id", f"Patient-{i+1:03d}")
            pred["patient_id"] = patient_id
            results.append(pred)

            if pred["quantum"]["label"] == "Malignant":
                malignant_count += 1
            else:
                benign_count += 1

            if pred["consensus"]:
                consensus_count += 1

    return jsonify({
        "total_cases": len(results),
        "malignant_count": malignant_count,
        "benign_count": benign_count,
        "consensus_rate": round((consensus_count / len(results)) * 100, 2) if results else 0,
        "results": results,
    })


@app.route("/api/sample_csv", methods=["GET"])
def sample_csv():
    """Generates a downloadable sample CSV with 10 real patient records."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header: patient_id + 30 feature names
    header = ["patient_id"] + [name.replace(" ", "_") for name in FEATURE_NAMES]
    writer.writerow(header)

    sample_indices = [0, 1, 2, 3, 4, 20, 21, 22, 50, 51]
    for i, idx in enumerate(sample_indices):
        patient_id = f"PATIENT_{idx+1:03d}"
        row = [patient_id] + list(raw_dataset.data[idx])
        writer.writerow(row)

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=qura_sample_patient_biopsies.csv"}
    )


@app.route("/api/feature_importance", methods=["GET"])
def feature_importance():
    """Returns the top contributing features to the quantum representation based on PCA loading magnitudes."""
    # PCA components shape is (n_components=8, n_features=30)
    # Total importance across top 8 components
    loadings = onp.abs(pca.components_).mean(axis=0)
    top_indices = onp.argsort(loadings)[::-1]
    
    ranked_features = []
    for idx in top_indices:
        ranked_features.append({
            "feature": FEATURE_NAMES[idx],
            "importance": round(float(loadings[idx]), 4),
            "group": "mean" if idx < 10 else ("se" if idx < 20 else "worst")
        })

    return jsonify({"rankings": ranked_features})


if __name__ == "__main__":
    print(f"Starting Qura Web Server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
