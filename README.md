# Qura — Hybrid Quantum-Classical ML for Early Disease Detection
### SIH26139 — Ministry submission, PS 139

## Overview
Qura is a prototype for early disease detection combining a Variational
Quantum Classifier (VQC) with a classical machine learning baseline,
benchmarked side-by-side on real diagnostic data (Breast Cancer Wisconsin
dataset). The goal is a working demo that takes tumor measurements as
input and returns predictions from both a quantum and a classical model,
with an honest comparison between the two.

## Repo Structure
- `quantum/` — VQC implementation, training scripts, saved model weights,
  and prediction interface (complete)
- `classical/` — classical baseline models and evaluation (in progress)
- `ui/` — demo web interface wiring both models together (in progress)
- `requirements.txt` — Python dependencies for the whole project

## Setup
```bash
pip install -r requirements.txt
```

## Quantum Benchmark Results

| Model / Configuration | Features | Test Accuracy | Train Accuracy |
|---|---|---|---|
| **Classical Logistic Regression** | 30 (All) | **97.37%** | 98.24% |
| **VQC (Optimized Pipeline)** | **8 (PCA)** | **97.37%** | **96.70%** |
| VQC (StronglyEntanglingLayers, 60 epochs) | 8 (PCA) | 94.74% | 90.11% |
| VQC (Custom ring-CNOT ansatz, 30 epochs) | 8 (PCA) | 93.86% | 90.77% |
| VQC (6 qubits, 2 layers baseline) | 6 (PCA) | 88.60% | 89.01% |

Full experiment logs: `quantum/max_accuracy_results.txt`, `quantum/advanced_results.txt`, `quantum/results.txt`

### Key Architectural Highlights
- **Data Re-Uploading**: Features are re-encoded at every variational layer via $R_Y$ rotations for enhanced expressivity.
- **Multi-Qubit Entangled Measurement**: Expectation values across all 8 qubits are captured with trainable linear combination weights and bias.
- **Hinge Loss + LR Scheduling**: Optimized margin classification and dynamic learning rate decay.
- **Dimensionality Reduction**: The quantum model achieves parity with the classical baseline using only **8 PCA features** vs. the full 30 features required classically.

## Why Quantum?
While classical models excel on small tabular datasets, the quantum classifier demonstrates that high-dimensional clinical feature spaces can be compressed into a compact quantum Hilbert space without losing diagnostic precision. This validates the hybrid quantum-classical methodology for future scaling onto genomic and multi-omic disease datasets where classical models face the curse of dimensionality.

## Team
6-member team, SIH internal hackathon. Domains: Quantum (VQC), Classical
Baseline, UI/Demo — 2 members each, pitch deck built collectively.
