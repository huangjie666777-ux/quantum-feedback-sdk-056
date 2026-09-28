"""Exception types raised by the quantum feedback SDK."""


class CircuitValidationError(ValueError):
    """Raised when a circuit or an operation is invalid."""


class SamplingError(ValueError):
    """Raised when sampling arguments are invalid."""
