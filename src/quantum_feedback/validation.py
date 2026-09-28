"""Circuit specification validation.

All structural and parameter checks happen *before* any quantum state is
touched, so an invalid circuit never produces partial results.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .errors import CircuitError

_UNITARY_OPS = frozenset({"H", "X", "Z", "RZ"})
_SUPPORTED_OPS = frozenset(
    {"H", "X", "Z", "RZ", "CX", "measure", "reset", "phase_flip"}
)


def _fail(index: int | None, message: str) -> None:
    if index is None:
        raise CircuitError(message)
    raise CircuitError(f"operation {index}: {message}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_qubit(value: Any, n_qubits: int, index: int, field: str) -> None:
    if not _is_int(value):
        _fail(index, f"{field} must be an integer, got {type(value).__name__}")
    if not 0 <= value < n_qubits:
        _fail(index, f"{field} {value} out of range for {n_qubits} qubit(s)")


def _check_cbit(value: Any, n_cbits: int, index: int, field: str) -> None:
    if not _is_int(value):
        _fail(index, f"{field} must be an integer, got {type(value).__name__}")
    if not 0 <= value < n_cbits:
        _fail(index, f"{field} {value} out of range for {n_cbits} classical bit(s)")


def _check_condition(condition: Any, n_cbits: int, index: int) -> None:
    if not isinstance(condition, Mapping) or len(condition) != 2:
        _fail(index, "condition must be a mapping with keys 'cbit' and 'equals'")
    if "cbit" not in condition or "equals" not in condition:
        _fail(index, "condition must contain 'cbit' and 'equals'")
    _check_cbit(condition["cbit"], n_cbits, index, "condition cbit")
    equals = condition["equals"]
    if not _is_int(equals) or equals not in (0, 1):
        _fail(index, "condition 'equals' must be 0 or 1")


def validate_circuit(
    n_qubits: Any, n_cbits: Any, operations: Any
) -> list[Mapping[str, Any]]:
    """Validate dimensions and the full ordered operation list.

    Returns a deep-copied, normalized operation list so the engine can never
    mutate the caller's data.
    """
    if not _is_int(n_qubits) or not 1 <= n_qubits <= 6:
        raise CircuitError("n_qubits must be an integer between 1 and 6")
    if not _is_int(n_cbits) or not 1 <= n_cbits <= 6:
        raise CircuitError("n_cbits must be an integer between 1 and 6")
    if not isinstance(operations, (list, tuple)):
        raise CircuitError("operations must be an ordered list")

    normalized: list[Mapping[str, Any]] = []
    for index, raw in enumerate(operations):
        if not isinstance(raw, Mapping):
            _fail(index, "operation must be a mapping")
        op = raw.get("op")
        if not isinstance(op, str):
            _fail(index, "missing or non-string 'op' field")
        if op not in _SUPPORTED_OPS:
            _fail(index, f"unknown operation {op!r}")

        copied = dict(raw)
        condition = copied.get("condition")
        if condition is not None:
            if op not in _UNITARY_OPS and op != "CX":
                _fail(index, f"operation {op!r} cannot be classically conditioned")
            _check_condition(condition, n_cbits, index)
            copied["condition"] = dict(condition)

        if op in _UNITARY_OPS:
            _check_qubit(copied.get("qubit"), n_qubits, index, "qubit")
        if op == "RZ":
            angle = copied.get("angle")
            if not isinstance(angle, (int, float)) or isinstance(angle, bool):
                _fail(index, "RZ angle must be a real number")
            if not math.isfinite(float(angle)):
                _fail(index, "RZ angle must be finite")
            copied["angle"] = float(angle)
        if op == "CX":
            _check_qubit(copied.get("control"), n_qubits, index, "CX control")
            _check_qubit(copied.get("target"), n_qubits, index, "CX target")
            if copied["control"] == copied["target"]:
                _fail(index, "CX control and target must be different qubits")
        if op == "measure":
            _check_qubit(copied.get("qubit"), n_qubits, index, "measured qubit")
            _check_cbit(copied.get("cbit"), n_cbits, index, "target cbit")
        if op == "reset":
            _check_qubit(copied.get("qubit"), n_qubits, index, "reset qubit")
        if op == "phase_flip":
            _check_qubit(copied.get("qubit"), n_qubits, index, "phase_flip qubit")
            probability = copied.get("p")
            if not isinstance(probability, (int, float)) or isinstance(
                probability, bool
            ):
                _fail(index, "phase_flip probability p must be a real number")
            probability = float(probability)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                _fail(index, "phase_flip probability p must be within [0, 1]")
            copied["p"] = probability

        normalized.append(copied)

    return normalized


__all__ = ["CircuitError", "validate_circuit"]
