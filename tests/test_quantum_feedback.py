import copy
import unittest

import numpy as np

from quantum_feedback import CircuitValidationError, run


class QuantumFeedbackTests(unittest.TestCase):
    def assert_density_close(self, actual, expected, tol=1e-10):
        np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol)

    def test_bell_measurement_feedback_corrects_target(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "CX", "control": 0, "target": 1},
            {"op": "MEASURE", "qubit": 0, "clbit": 0},
            {"op": "X", "qubit": 1, "condition": [0, 1]},
        ]
        result = run(2, 1, operations)
        probabilities = result.classical_probabilities

        self.assertEqual(set(probabilities), {"0", "1"})
        np.testing.assert_allclose(probabilities["0"], 0.5)
        np.testing.assert_allclose(probabilities["1"], 0.5)

        rho = result.density_matrix
        expected = np.diag([0.5, 0.5, 0.0, 0.0]).astype(complex)
        self.assert_density_close(rho, expected)

    def test_measurement_bit_ordering_and_unmeasured_bits(self):
        operations = [
            {"op": "X", "qubit": 0},
            {"op": "MEASURE", "qubit": 0, "clbit": 0},
        ]
        result = run(2, 3, operations)
        self.assertEqual(result.classical_probabilities, {"001": 1.0})

    def test_reset_keeps_reduced_state_of_other_qubits(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "CX", "control": 0, "target": 1},
            {"op": "RESET", "qubit": 1},
        ]
        result = run(2, 1, operations)
        rho = result.density_matrix
        reduced = rho[0:2, 0:2] + rho[2:4, 2:4]
        expected = np.eye(2, dtype=complex) * 0.5
        self.assert_density_close(reduced, expected)
        self.assert_density_close(rho[2:4, 2:4], np.zeros((2, 2), dtype=complex))

    def test_phase_flip_is_deterministic_mixture(self):
        result = run(1, 1, [
            {"op": "H", "qubit": 0},
            {"op": "PHASE_FLIP", "qubit": 0, "p": 0.25},
        ])
        expected = np.full((2, 2), 0.5, dtype=complex)
        expected[0, 1] = expected[1, 0] = 0.25
        self.assert_density_close(result.density_matrix, expected)

    def test_rz_diagonal(self):
        angle = 0.7
        result = run(1, 1, [
            {"op": "H", "qubit": 0},
            {"op": "RZ", "qubit": 0, "angle": angle},
        ])
        rho = result.density_matrix
        np.testing.assert_allclose(rho[0, 1], 0.5 * np.exp(-1j * angle), atol=1e-12)

    def test_condition_does_not_act_on_mismatched_branch(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "MEASURE", "qubit": 0, "clbit": 0},
            {"op": "X", "qubit": 1, "condition": [0, 0]},
        ]
        result = run(2, 1, operations)
        rho = result.density_matrix
        self.assertAlmostEqual(rho[1, 1], 0.5)
        self.assertAlmostEqual(rho[2, 2], 0.5)
        self.assertAlmostEqual(np.trace(rho), 1.0)

    def test_sampling_is_seeded_and_non_global(self):
        result = run(1, 1, [{"op": "H", "qubit": 0}, {"op": "MEASURE", "qubit": 0, "clbit": 0}])
        first = result.samples(8, seed=123)
        second = result.samples(8, seed=123)
        self.assertEqual(first, second)
        before = np.random.get_state()[1][:5].copy()
        result.samples(4, seed=9)
        after = np.random.get_state()[1][:5]
        np.testing.assert_array_equal(before, after)

    def test_caller_data_is_not_mutated(self):
        operations = [
            {"op": "H", "qubit": 0},
            {"op": "RZ", "qubit": 0, "angle": 0.5, "condition": [0, 0]},
        ]
        snapshot = copy.deepcopy(operations)
        run(1, 1, operations)
        self.assertEqual(operations, snapshot)

    def test_validation_reports_operation_position(self):
        bad_circuits = [
            [{"op": "X", "qubit": 2}],
            [{"op": "CX", "control": 1, "target": 1}],
            [{"op": "BOGUS"}],
            [{"op": "RZ", "qubit": 0, "angle": float("nan")}],
            [{"op": "PHASE_FLIP", "qubit": 0, "p": 1.5}],
            [{"op": "MEASURE", "qubit": 0, "clbit": 1}],
        ]
        for operations in bad_circuits:
            with self.assertRaisesRegex(CircuitValidationError, "operation 0"):
                run(2, 1, operations)


if __name__ == "__main__":
    unittest.main()
