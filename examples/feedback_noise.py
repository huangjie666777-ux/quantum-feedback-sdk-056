"""Entangled measurement feedback plus phase-flip noise demonstration."""

import numpy as np

from quantum_feedback import run_circuit


operations = [
    {"op": "H", "qubit": 0},
    {"op": "CX", "control": 0, "target": 1},
    {"op": "phase_flip", "qubit": 1, "p": 0.2},
    {"op": "measure", "qubit": 0, "cbit": 0},
    {"op": "X", "qubit": 1, "condition": {"cbit": 0, "equals": 1}},
]

result = run_circuit(2, 2, operations)
np.set_printoptions(precision=4, suppress=True)
print("classical outcome probabilities:", result.classical_probabilities)
print("final density matrix (real part):")
print(result.density_matrix.real)
print("seeded samples:", result.sample(12, seed=2026))
