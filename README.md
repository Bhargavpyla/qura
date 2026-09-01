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
pip install -r requirements.txt

## Quantum Results (so far)
| Config | Test Accuracy |
|---|---|
| Classical Logistic Regression (quantum-side baseline) | 97.4% |
| VQC — StronglyEntanglingLayers, 8 qubits, 3 layers, 60 epochs | 94.7% |
| VQC — custom ring-CNOT ansatz, 8 qubits, 3 layers | 93.9% |
| VQC — 6 qubits, 2 layers (first working version) | 88.6% |

Full experiment logs: `quantum/results.txt`, `quantum/advanced_results.txt`

## Why Quantum?
Classical models currently outperform the VQC on this dataset because it's
small and low-dimensional — there's no complexity here that needs
quantum's extra expressive power. The value case is forward-looking: as
quantum hardware scales, these techniques are expected to help on
higher-dimensional problems (genomics, molecular/drug discovery data)
where classical models start to struggle. This project demonstrates a
working, honestly-benchmarked hybrid pipeline and methodology that
generalizes to those harder problems.

## Team
6-member team, SIH internal hackathon. Domains: Quantum (VQC), Classical
Baseline, UI/Demo — 2 members each, pitch deck built collectively.
