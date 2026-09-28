"""Quantum operators and density-matrix channel primitives."""

from __future__ import annotations

import numpy as np


def reduce_kron(matrices: list[np.ndarray]) -> np.ndarray:
    result = np.array([[1.0]], dtype=np.complex128)
    for matrix in matrices:
        result = np.kron(result, matrix.astype(np.complex128, copy=False))
    return result


def _single_qubit_matrix(gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    # Basis index bit 0 is the least-significant bit, so the tensor order is
    # reversed when constructing the full Hilbert-space operator.
    matrices = [
        gate if current == qubit else np.eye(2, dtype=np.complex128)
        for current in range(n_qubits)
    ]
    return reduce_kron(matrices[::-1])


def h_matrix() -> np.ndarray:
    return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)


def x_matrix() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def z_matrix() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def rz_matrix(angle: float) -> np.ndarray:
    half = angle / 2.0
    return np.array(
        [[np.exp(-1j * half), 0.0], [0.0, np.exp(1j * half)]], dtype=np.complex128
    )


def cx_matrix(control: int, target: int, n_qubits: int) -> np.ndarray:
    dimension = 1 << n_qubits
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    for basis_index in range(dimension):
        output = basis_index ^ (
            1 << target if (basis_index >> control) & 1 else 0
        )
        result[output, basis_index] = 1.0
    return result


def gate_operator(name: str, operation: dict, n_qubits: int) -> np.ndarray:
    if name == "H":
        single = h_matrix()
    elif name == "X":
        single = x_matrix()
    elif name == "Z":
        single = z_matrix()
    elif name == "RZ":
        single = rz_matrix(float(operation["angle"]))
    else:
        raise ValueError(f"not a supported gate: {name!r}")
    return _single_qubit_matrix(single, int(operation["qubit"]), n_qubits)


def projectors(qubit: int, n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    p0_basis = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    p1_basis = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    return (
        _single_qubit_matrix(p0_basis, qubit, n_qubits),
        _single_qubit_matrix(p1_basis, qubit, n_qubits),
    )


def apply_unitary(rho: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    return unitary @ rho @ unitary.conj().T


def phase_flip_channel(
    rho: np.ndarray, qubit: int, p: float, n_qubits: int
) -> np.ndarray:
    """Probabilistic mixture (1-p) rho + p Z rho Z at the specified point."""
    z_operator = _single_qubit_matrix(z_matrix(), qubit, n_qubits)
    return (1.0 - p) * rho + p * apply_unitary(rho, z_operator)


def reset_channel(rho: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Reset one qubit to |0> while preserving the other qubits' reduced state.

    The target qubit is traced out and reattached in |0>. Its entanglement
    with other qubits is discarded, exactly as a physical reset requires.
    """
    dimension = 1 << n_qubits
    tensor = rho.reshape((2,) * (2 * n_qubits))
    # Axes are row qubits 0..n-1 followed by column qubits 0..n-1.
    tensor_axis = n_qubits - 1 - qubit
    reduced = np.trace(tensor, axis1=tensor_axis, axis2=tensor_axis + n_qubits)
    if n_qubits == 1:
        reduced = np.array([[1.0]], dtype=np.complex128)

    other_count = n_qubits - 1
    isometry = np.zeros((dimension, 1 << other_count), dtype=np.complex128)
    for reduced_index in range(1 << other_count):
        full_index = _insert_bit(reduced_index, qubit, 0)
        isometry[full_index, reduced_index] = 1.0
    return isometry @ reduced @ isometry.conj().T


def _insert_bit(value: int, position: int, bit: int) -> int:
    low_mask = (1 << position) - 1
    low = value & low_mask
    high = value & ~low_mask
    return low | (high << 1) | (bit << position)


__all__ = [
    "apply_unitary",
    "cx_matrix",
    "gate_operator",
    "phase_flip_channel",
    "projectors",
    "reduce_kron",
    "reset_channel",
]
