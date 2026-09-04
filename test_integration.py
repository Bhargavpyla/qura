"""
Integration Test Suite for QURA — Hybrid Quantum-Classical Medical Diagnostic System.
Validates:
  1. Artifact integrity (transformers, quantum weights, classical baselines).
  2. Inference consistency (quantum, fair-classical, full-classical).
  3. Flask API endpoint responses and schema.
  4. Location-independent script paths.
"""

import os
import sys
import unittest
import numpy as np
from sklearn.datasets import load_breast_cancer

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
QUANTUM_DIR = os.path.join(TEST_DIR, "quantum")
UI_DIR = os.path.join(TEST_DIR, "ui")
CLASSICAL_DIR = os.path.join(TEST_DIR, "classical-baseline", "BreastCancerClassical")

for p in [TEST_DIR, QUANTUM_DIR, UI_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


class TestQuraIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from vqc_predict_demo import (
            predict_all,
            predict_both,
            predict_quantum,
            predict_classical_full,
            predict_classical_fair,
            scaler,
            pca,
            angle_scaler,
            classical_full_model,
            fair_pca8_model,
        )
        from app import app

        cls.predict_all = staticmethod(predict_all)
        cls.predict_both = staticmethod(predict_both)
        cls.predict_quantum = staticmethod(predict_quantum)
        cls.predict_classical_full = staticmethod(predict_classical_full)
        cls.predict_classical_fair = staticmethod(predict_classical_fair)
        cls.scaler = scaler
        cls.pca = pca
        cls.angle_scaler = angle_scaler
        cls.classical_full_model = classical_full_model
        cls.fair_pca8_model = fair_pca8_model

        app.config["TESTING"] = True
        cls.client = app.test_client()
        cls.dataset = load_breast_cancer()

    def test_01_artifact_loading(self):
        """Verify that all pre-trained transformers and model checkpoints are loaded and functional."""
        self.assertIsNotNone(self.scaler, "StandardScaler failed to load")
        self.assertIsNotNone(self.pca, "PCA transformer failed to load")
        self.assertIsNotNone(self.angle_scaler, "Angle scaler failed to load")
        self.assertIsNotNone(self.classical_full_model, "30-feature classical model failed to load")
        self.assertIsNotNone(self.fair_pca8_model, "Fair PCA-8 classical model failed to load")
        self.assertEqual(self.pca.n_components, 8, "PCA must have 8 components matching the 8 qubits")

    def test_02_inference_schema_and_types(self):
        """Verify predict_all() returns complete tri-engine diagnostic schema."""
        sample_features = self.dataset.data[0]
        result = self.predict_all(sample_features)

        self.assertIn("quantum", result)
        self.assertIn("classical", result)
        self.assertIn("fair_classical", result)
        self.assertIn("consensus", result)
        self.assertIn("consensus_status", result)
        self.assertIn("primary_label", result)

        q = result["quantum"]
        self.assertIn(q["label"], ["Benign", "Malignant"])
        self.assertTrue(0 <= q["confidence"] <= 100)
        self.assertEqual(len(q["qubit_expectations"]), 8)
        self.assertEqual(len(q["quantum_angles_rad"]), 8)

        f = result["fair_classical"]
        self.assertIn(f["label"], ["Benign", "Malignant"])
        self.assertTrue(0 <= f["confidence"] <= 100)
        self.assertAlmostEqual(f["probability_malignant"] + f["probability_benign"], 100.0, delta=0.5)

        c = result["classical"]
        self.assertIn(c["label"], ["Benign", "Malignant"])
        self.assertTrue(0 <= c["confidence"] <= 100)
        self.assertAlmostEqual(c["probability_malignant"] + c["probability_benign"], 100.0, delta=0.5)

    def test_03_benchmark_cases_fidelity(self):
        """Verify predictions on known benchmark patients match expectations."""
        # Patient 0 is known Malignant
        p0 = self.predict_all(self.dataset.data[0])
        self.assertEqual(p0["quantum"]["label"], "Malignant")
        self.assertEqual(p0["classical"]["label"], "Malignant")
        self.assertEqual(p0["fair_classical"]["label"], "Malignant")
        self.assertTrue(p0["consensus"])

        # Patient 20 is known Benign
        p20 = self.predict_all(self.dataset.data[20])
        self.assertEqual(p20["quantum"]["label"], "Benign")
        self.assertEqual(p20["classical"]["label"], "Benign")
        self.assertEqual(p20["fair_classical"]["label"], "Benign")
        self.assertTrue(p20["consensus"])

    def test_04_api_metadata(self):
        """Test GET /api/metadata endpoint."""
        res = self.client.get("/api/metadata")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data["feature_names"]), 30)
        self.assertEqual(data["n_qubits"], 8)
        self.assertIn("accuracy", data)
        self.assertIn("models", data)
        self.assertIn("fair_classical", data["models"])

    def test_05_api_samples(self):
        """Test GET /api/samples endpoint."""
        res = self.client.get("/api/samples")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("sample_0_malignant", data)
        self.assertIn("sample_20_benign", data)
        self.assertEqual(len(data["sample_0_malignant"]["features"]), 30)

    def test_06_api_predict(self):
        """Test POST /api/predict endpoint."""
        payload = {"features": list(self.dataset.data[0])}
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("quantum", data)
        self.assertIn("fair_classical", data)
        self.assertIn("classical", data)
        self.assertIn("consensus", data)

    def test_07_api_batch_predict(self):
        """Test POST /api/batch_predict endpoint."""
        rows = [
            {"id": "P-001", "features": list(self.dataset.data[0])},
            {"id": "P-020", "features": list(self.dataset.data[20])},
        ]
        res = self.client.post("/api/batch_predict", json={"rows": rows})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["total_cases"], 2)
        self.assertEqual(data["malignant_count"], 1)
        self.assertEqual(data["benign_count"], 1)
        self.assertIn("quantum_fair_consensus_rate", data)
        self.assertEqual(len(data["results"]), 2)

    def test_08_api_feature_importance(self):
        """Test GET /api/feature_importance endpoint."""
        res = self.client.get("/api/feature_importance")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("rankings", data)
        self.assertEqual(len(data["rankings"]), 30)

    def test_09_classical_baseline_csv_path(self):
        """Verify classical baseline CSV exists and is reachable."""
        csv_file = os.path.join(CLASSICAL_DIR, "breast_cancer_Main.csv")
        self.assertTrue(os.path.exists(csv_file), f"Missing classical dataset at {csv_file}")

    def test_10_requirements_utf8(self):
        """Verify requirements.txt is readable UTF-8 without errors."""
        req_path = os.path.join(TEST_DIR, "requirements.txt")
        with open(req_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("pennylane", content)
        self.assertIn("flask", content)


def main():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestQuraIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    main()
