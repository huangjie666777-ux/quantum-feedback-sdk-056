"""Result representation, classical sampling, and the public circuit API."""

from __future__ import annotations

from typing import Any

import numpy as np

from .engine import BranchEngine
from .validation import validate_circuit

TOLERANCE = 1e-10


class SimulationResult:
    """Exact final quantum state and classical register outcome distribution."""

    def __init__(
        self,
        n_qubits: int,
        n_cbits: int,
        branches: dict[int, np.ndarray],
        tolerance: float = TOLERANCE,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_cbits = n_cbits
        self.tolerance = tolerance
        total_rho = sum(branches.values())
        self._density_matrix = (total_rho + total_rho.conj().T) / 2.0

        probabilities: dict[str, float] = {}
        for mask, rho in branches.items():
            probability = float(np.trace(rho).real)
            if probability < -tolerance:
                raise ValueError("negative branch probability encountered")
            probabilities[self.format_classical_bits(mask)] = max(0.0, probability)
        self._classical_probabilities = probabilities

    @property
    def density_matrix(self) -> np.ndarray:
        """Copy of the final 2^n by 2^n density matrix."""
        return self._density_matrix.copy()

    @property
    def classical_probabilities(self) -> dict[str, float]:
        """Mapping from high-bit-left classical strings to exact probabilities."""
        return dict(self._classical_probabilities)

    def format_classical_bits(self, mask: int) -> str:
        return "".join(
            str((mask >> cbit) & 1) for cbit in range(self.n_cbits - 1, -1, -1)
        )

    def sample(self, shots: int, seed: int | None = None) -> list[str]:
        """Draw classical readout strings using a private NumPy RNG.

        Sampling does not alter the exact simulation result and never touches
        NumPy's global random state.
        """
        if not isinstance(shots, int) or isinstance(shots, bool) or shots < 0:
            raise ValueError("shots must be a non-negative integer")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("seed must be an integer or None")
        outcomes = list(self._classical_probabilities.keys())
        weights = np.array(
            [self._classical_probabilities[outcome] for outcome in outcomes],
            dtype=np.float64,
        )
        weight_sum = weights.sum()
        if not np.isfinite(weight_sum) or abs(weight_sum - 1.0) > self.tolerance:
            raise ValueError("classical probabilities do not sum to 1")
        weights = weights / weight_sum
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(outcomes), size=shots, p=weights)
        return [outcomes[int(index)] for index in indices]


def run_circuit(
    n_qubits: int,
    n_cbits: int,
    operations: list[dict[str, Any]],
    tolerance: float = TOLERANCE,
) -> SimulationResult:
    """Validate and deterministically simulate a circuit.

    The supplied operation list is copied during validation and is never
    mutated. Validation runs before simulation, so invalid input cannot return
    partial results.
    """
    normalized = validate_circuit(n_qubits, n_cbits, operations)
    engine = BranchEngine(int(n_qubits), int(n_cbits))
    branches = engine.run(normalized)
    return SimulationResult(int(n_qubits), int(n_cbits), branches, tolerance)


__all__ = ["SimulationResult", "TOLERANCE", "run_circuit"]
