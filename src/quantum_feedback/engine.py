"""Deterministic density-matrix execution with explicit measurement branches."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .operators import (
    apply_unitary,
    cx_matrix,
    gate_operator,
    phase_flip_channel,
    projectors,
    reset_channel,
)


class BranchEngine:
    """Evolves one unnormalized density matrix per distinct classical record.

    A branch's trace is its exact joint probability. Measurement projections
    never renormalize within a result, so all outcome information is retained.
    Branch weights are added incoherently; distinct outcomes are never made
    coherent by later operations.
    """

    def __init__(self, n_qubits: int, n_cbits: int) -> None:
        self.n_qubits = n_qubits
        self.n_cbits = n_cbits
        dimension = 1 << n_qubits
        rho = np.zeros((dimension, dimension), dtype=np.complex128)
        rho[0, 0] = 1.0
        # Classical bit masks use bit c for classical register index c.
        self.branches: dict[int, np.ndarray] = {0: rho}

    def run(self, operations: list[Mapping]) -> dict[int, np.ndarray]:
        for operation in operations:
            name = operation["op"]
            if name in {"H", "X", "Z", "RZ"}:
                self._apply_single_gate(name, operation)
            elif name == "CX":
                self._apply_cx(operation)
            elif name == "measure":
                self._measure(operation)
            elif name == "reset":
                self._reset(operation)
            elif name == "phase_flip":
                self._phase_flip(operation)
        return self.branches

    def _condition_allows(self, mask: int, operation: Mapping) -> bool:
        condition = operation.get("condition")
        if condition is None:
            return True
        bit = (mask >> int(condition["cbit"])) & 1
        return bit == int(condition["equals"])

    def _apply_single_gate(self, name: str, operation: Mapping) -> None:
        unitary = gate_operator(name, operation, self.n_qubits)
        for mask, rho in list(self.branches.items()):
            if self._condition_allows(mask, operation):
                self.branches[mask] = apply_unitary(rho, unitary)

    def _apply_cx(self, operation: Mapping) -> None:
        unitary = cx_matrix(
            int(operation["control"]),
            int(operation["target"]),
            self.n_qubits,
        )
        for mask, rho in list(self.branches.items()):
            if self._condition_allows(mask, operation):
                self.branches[mask] = apply_unitary(rho, unitary)

    def _measure(self, operation: Mapping) -> None:
        qubit = int(operation["qubit"])
        cbit = int(operation["cbit"])
        p0, p1 = projectors(qubit, self.n_qubits)
        incoming = self.branches
        self.branches = {}

        for old_mask, rho in incoming.items():
            overwritten_mask = old_mask & ~(1 << cbit)
            projected = (p0 @ rho @ p0, p1 @ rho @ p1)
            for result_bit, post_state in enumerate(projected):
                weight = float(np.trace(post_state).real)
                if weight <= 0.0:
                    continue
                new_mask = overwritten_mask | (result_bit << cbit)
                if new_mask in self.branches:
                    # Same classical record: add probabilities incoherently.
                    self.branches[new_mask] = self.branches[new_mask] + post_state
                else:
                    self.branches[new_mask] = post_state

    def _reset(self, operation: Mapping) -> None:
        qubit = int(operation["qubit"])
        for mask, rho in list(self.branches.items()):
            self.branches[mask] = reset_channel(rho, qubit, self.n_qubits)

    def _phase_flip(self, operation: Mapping) -> None:
        qubit = int(operation["qubit"])
        probability = float(operation["p"])
        for mask, rho in list(self.branches.items()):
            self.branches[mask] = phase_flip_channel(
                rho, qubit, probability, self.n_qubits
            )


__all__ = ["BranchEngine"]
