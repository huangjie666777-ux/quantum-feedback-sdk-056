"""Exception types raised by the quantum feedback SDK."""


class CircuitError(ValueError):
    """Raised when a circuit specification is invalid.

    The message always identifies the offending operation position
    (0-based index into the operation list), when one exists.
    """


__all__ = ["CircuitError"]
