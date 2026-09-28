from quantum_feedback import run


operations = [
    {"op": "H", "qubit": 0},
    {"op": "CX", "control": 0, "target": 1},
    {"op": "PHASE_FLIP", "qubit": 1, "p": 0.1},
    {"op": "MEASURE", "qubit": 0, "clbit": 0},
    {"op": "X", "qubit": 1, "condition": [0, 1]},
]

result = run(num_qubits=2, num_clbits=2, operations=operations)
print("classical probabilities:", result.classical_probabilities)
print("seeded counts:", result.sample_counts(shots=1000, seed=20260928))
print("final trace:", complex(result.density_matrix.trace()))
