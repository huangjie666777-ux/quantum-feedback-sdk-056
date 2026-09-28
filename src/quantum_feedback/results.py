"""Simulation result and deterministic classical sampling."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .errors import SamplingError


class ExecutionResult:
    def __init__(
        self,
        rho: np.ndarray,
        branch_probabilities: dict[tuple[int, ...], float],
        num_qubits: int,
        num_clbits: int,
    ) -> None:
        self._rho = rho
        self._branch_probabilities = dict(branch_probabilities)
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits

    @property
    def density_matrix(self) -> np.ndarray:
        return self._rho.copy()

    @property
    def classical_probabilities(self) -> dict[str, float]:
        return {
            self._format_bits(bits): probability
            for bits, probability in self._branch_probabilities.items()
            if probability > 0.0
        }

    def samples(self, shots: int, seed: int) -> list[str]:
        if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
            raise SamplingError("shots must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise SamplingError("seed must be an integer")

        bitstrings = list(self.classical_probabilities)
        probabilities = np.asarray([self.classical_probabilities[key] for key in bitstrings])
        probabilities /= probabilities.sum()
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(bitstrings), size=shots, p=probabilities)
        return [bitstrings[index] for index in indices]

    def sample_counts(self, shots: int, seed: int) -> dict[str, int]:
        return dict(Counter(self.samples(shots, seed)))

    def _format_bits(self, bits: tuple[int, ...]) -> str:
        return "".join(str(value) for value in reversed(bits))
