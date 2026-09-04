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
import numpy as onp
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from sklearn.datasets import load_breast_cancer

# Ensure quantum package can be loaded reliably from any working directory
UI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(UI_DIR, ".."))
QUANTUM_DIR = os.path.join(PROJECT_DIR, "quantum")
if QUANTUM_DIR not in sys.path:
    sys.path.insert(0, QUANTUM_DIR)

from vqc_predict_demo import (
    predict_all,
    pca,
    N_QUBITS,
    N_LAYERS,
)

app = Flask(
    __name__,
    template_folder=os.path.join(UI_DIR, "templates"),
    static_folder=os.path.join(UI_DIR, "static"),
)
CORS(app)

# -------------------- Load Dataset Metadata & Presets --------------------
raw_dataset = load_breast_cancer()
FEATURE_NAMES = list(raw_dataset.feature_names)

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
    """Utility for running combined quantum + fair-classical + full-classical inference."""
    return predict_all(raw_features_list)


# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    metadata_path = os.path.join(QUANTUM_DIR, "model_metadata.txt")
    accuracies = {
        "vqc_test_acc": 0.9737,
        "vqc_train_acc": 0.9670,
        "classical_test_acc": 0.9737,
        "classical_train_acc": 0.9868,
        "classical_pca8_test_acc": 0.9912,
        "classical_pca8_train_acc": 0.9780,
    }
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        try:
                            accuracies[k.strip()] = float(v.strip())
                        except ValueError:
                            pass
        except Exception:
            pass

    return jsonify({
        "feature_names": FEATURE_NAMES,
        "n_qubits": N_QUBITS,
        "n_layers": N_LAYERS,
        "accuracy": accuracies,
        "models": {
            "quantum_vqc": {
                "name": "Optimized VQC (8Q, 3L)",
                "features": 8,
                "feature_type": "PCA-8 Angles [0, π]",
                "test_acc": accuracies.get("vqc_test_acc", 0.9737),
                "train_acc": accuracies.get("vqc_train_acc", 0.9670),
            },
            "fair_classical": {
                "name": "Fair Baseline LR (PCA-8)",
                "features": 8,
                "feature_type": "PCA-8 Components",
                "test_acc": accuracies.get("classical_pca8_test_acc", 0.9912),
                "train_acc": accuracies.get("classical_pca8_train_acc", 0.9780),
            },
            "classical_full": {
                "name": "Full Baseline LR (30-Feat)",
                "features": 30,
                "feature_type": "Standardized 30 FNA Features",
                "test_acc": accuracies.get("classical_test_acc", 0.9737),
                "train_acc": accuracies.get("classical_train_acc", 0.9868),
            },
        },
        "history": [
            {"model": "VQC (6q, 2L Custom)", "accuracy": 0.8860},
            {"model": "VQC (8q, 3L Ring-CNOT)", "accuracy": 0.9386},
            {"model": "VQC (8q, 3L StronglyEntangling)", "accuracy": 0.9474},
            {"model": "VQC (Optimized Pipeline)", "accuracy": 0.9737},
            {"model": "Classical Baseline (30-feat)", "accuracy": 0.9737},
            {"model": "Classical Fair Baseline (PCA-8)", "accuracy": 0.9912},
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

    result = predict_all(features)
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
    unanimous_count = 0
    quantum_fair_count = 0

    for i, row in enumerate(rows):
        features = row.get("features", [])
        if len(features) == 30:
            pred = predict_all(features)
            patient_id = row.get("id", f"Patient-{i+1:03d}")
            pred["patient_id"] = patient_id
            results.append(pred)

            if pred["quantum"]["label"] == "Malignant":
                malignant_count += 1
            else:
                benign_count += 1

            if pred["consensus"]:
                unanimous_count += 1
            if pred.get("quantum_fair_agreement", False):
                quantum_fair_count += 1

    total = len(results)
    return jsonify({
        "total_cases": total,
        "malignant_count": malignant_count,
        "benign_count": benign_count,
        "consensus_rate": round((unanimous_count / total) * 100, 2) if total else 0,
        "quantum_fair_consensus_rate": round((quantum_fair_count / total) * 100, 2) if total else 0,
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
    for idx in sample_indices:
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

