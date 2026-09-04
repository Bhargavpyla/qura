"""
QURA — Master Project Launcher
Allows launching the Web UI, running CLI inference, or executing the integration test suite.

Usage (from inside the qura/ repository):
  python run.py             # Launch the Interactive Clinical Web Dashboard
  python run.py --cli       # Run CLI benchmark prediction demo
  python run.py --test      # Run the integration test suite
"""

import os
import sys
import argparse

# Resolve paths relative to this repository root (qura/)
QURA_DIR = os.path.dirname(os.path.abspath(__file__))
QUANTUM_DIR = os.path.join(QURA_DIR, "quantum")
UI_DIR = os.path.join(QURA_DIR, "ui")

for path in [QURA_DIR, QUANTUM_DIR, UI_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)


def launch_web(port=5000, host="127.0.0.1", debug=True):
    print("=" * 70)
    print("  QURA — Hybrid Quantum-Classical Oncology Intelligence (SIH 2026)")
    print(f"  Starting Clinical Diagnostic Dashboard on http://{host}:{port}")
    print("=" * 70)
    from app import app
    app.run(host=host, port=port, debug=debug)


def launch_cli():
    print("=" * 70)
    print("  QURA — CLI Tri-Engine Diagnostic Benchmark Demo")
    print("=" * 70)
    from vqc_predict_demo import predict_all
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    print("Evaluating pre-trained models on real Wisconsin benchmark patient samples...\n")

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


def run_tests():
    print("=" * 70)
    print("  QURA — Running Integration Verification Suite")
    print("=" * 70)
    import test_integration
    test_integration.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QURA Hybrid Quantum-Classical System Launcher")
    parser.add_argument("--cli", action="store_true", help="Run benchmark patient inference in CLI")
    parser.add_argument("--test", action="store_true", help="Run integration test suite")
    parser.add_argument("--port", type=int, default=5000, help="Web server port (default: 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web server host (default: 127.0.0.1)")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")

    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.cli:
        launch_cli()
    else:
        launch_web(port=args.port, host=args.host, debug=not args.no_debug)
