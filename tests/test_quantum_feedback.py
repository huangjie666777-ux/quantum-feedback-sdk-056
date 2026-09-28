import math
import unittest

import numpy as np

from quantum_feedback import CircuitError, run_circuit


class QuantumFeedbackTests(unittest.TestCase):
    def assertDensityClose(self, actual, expected):
        self.assertTrue(np.allclose(actual, expected, atol=1e-10))

    def test_hadamard_population_before_measurement(self):
        result = run_circuit(1, 1, [{"op": "H", "qubit": 0}])
        self.assertTrue(
            np.allclose(result.density_matrix, np.full((2, 2), 0.5), atol=1e-12)
        )
        self.assertEqual(set(result.classical_probabilities), {"0"})
        self.assertAlmostEqual(result.classical_probabilities["0"], 1.0, places=12)
        self.assertEqual(result.sample(5, seed=3), ["0"] * 5)

    def test_entangled_measurement_feedback_corrects_bell_branches(self):
        # Measure q0 in the X basis using surrounding H gates. Condition Z on
        # q1 on the X-basis sign, then transform q1 back: both exact branches
        # converge to |00>; an early incoherent mix would fail to converge.
        ops = [
            {"op": "H", "qubit": 0},
            {"op": "CX", "control": 0, "target": 1},
            {"op": "H", "qubit": 0},
            {"op": "measure", "qubit": 0, "cbit": 0},
            {"op": "Z", "qubit": 1, "condition": {"cbit": 0, "equals": 1}},
            {"op": "X", "qubit": 0, "condition": {"cbit": 0, "equals": 1}},
            {"op": "H", "qubit": 1},
        ]
        result = run_circuit(2, 1, ops)
        expected = np.zeros((4, 4), dtype=np.complex128)
        expected[0, 0] = 1.0
        self.assertDensityClose(result.density_matrix, expected)
        probs = result.classical_probabilities
        self.assertAlmostEqual(probs["0"], 0.5, places=12)
        self.assertAlmostEqual(probs["1"], 0.5, places=12)
        samples = result.sample(100, seed=7)
        self.assertAlmostEqual(samples.count("0") / 100, 0.5, delta=0.2)
        self.assertAlmostEqual(samples.count("1") / 100, 0.5, delta=0.2)

    def test_conditional_zero_branch_runs_gate(self):
        ops = [
            {"op": "X", "qubit": 0, "condition": {"cbit": 0, "equals": 0}},
            {"op": "measure", "qubit": 0, "cbit": 0},
        ]
        result = run_circuit(1, 1, ops)
        self.assertEqual(result.classical_probabilities, {"1": 1.0})

    def test_measurement_overwrite_remaps_branches(self):
        ops = [
            {"op": "X", "qubit": 0},
            {"op": "measure", "qubit": 0, "cbit": 0},
            {"op": "X", "qubit": 0},
            {"op": "measure", "qubit": 0, "cbit": 0},
        ]
        result = run_circuit(1, 1, ops)
        self.assertEqual(result.classical_probabilities, {"0": 1.0})

    def test_rz_relative_phase(self):
        angle = 0.7
        result = run_circuit(
            1,
            1,
            [
                {"op": "H", "qubit": 0},
                {"op": "RZ", "qubit": 0, "angle": angle},
                {"op": "H", "qubit": 0},
            ],
        )
        p0 = math.cos(angle / 2.0) ** 2
        self.assertAlmostEqual(result.density_matrix[0, 0].real, p0, places=12)
        self.assertAlmostEqual(result.density_matrix[1, 1].real, 1 - p0, places=12)

    def test_measurement_dephases_rather_than_recoheres(self):
        ops = [
            {"op": "H", "qubit": 0},
            {"op": "measure", "qubit": 0, "cbit": 0},
            {"op": "H", "qubit": 0},
        ]
        result = run_circuit(1, 1, ops)
        self.assertDensityClose(
            result.density_matrix, np.diag([0.5, 0.5]).astype(complex)
        )

    def test_phase_flip_noise_is_probability_mixture(self):
        result = run_circuit(
            1,
            1,
            [
                {"op": "H", "qubit": 0},
                {"op": "phase_flip", "qubit": 0, "p": 0.25},
            ],
        )
        expected = np.array([[0.5, 0.25], [0.25, 0.5]], dtype=np.complex128)
        self.assertDensityClose(result.density_matrix, expected)

    def test_reset_preserves_reduced_state_of_other_qubit(self):
        ops = [
            {"op": "H", "qubit": 0},
            {"op": "CX", "control": 0, "target": 1},
            {"op": "reset", "qubit": 0},
        ]
        result = run_circuit(2, 1, ops)
        expected = np.zeros((4, 4), dtype=np.complex128)
        expected[0, 0] = 0.5
        expected[2, 2] = 0.5
        self.assertDensityClose(result.density_matrix, expected)

    def test_reset_does_not_clear_classical_bits(self):
        ops = [
            {"op": "X", "qubit": 0},
            {"op": "measure", "qubit": 0, "cbit": 1},
            {"op": "reset", "qubit": 0},
        ]
        result = run_circuit(1, 2, ops)
        self.assertEqual(result.classical_probabilities, {"10": 1.0})
        self.assertAlmostEqual(result.density_matrix[0, 0].real, 1.0)

    def test_unmeasured_classical_bits_are_zero_and_output_high_bits_left(self):
        ops = [{"op": "X", "qubit": 1}, {"op": "measure", "qubit": 1, "cbit": 0}]
        result = run_circuit(2, 3, ops)
        self.assertEqual(result.classical_probabilities, {"001": 1.0})
        self.assertAlmostEqual(result.density_matrix[2, 2].real, 1.0)

    def test_returned_state_and_probabilities_are_copies(self):
        result = run_circuit(1, 1, [{"op": "H", "qubit": 0}])
        rho = result.density_matrix
        probs = result.classical_probabilities
        rho[0, 0] = 99
        probs["x"] = 99
        self.assertNotEqual(result.density_matrix[0, 0], 99)
        self.assertNotIn("x", result.classical_probabilities)

    def test_caller_operations_are_not_modified(self):
        ops = [{"op": "RZ", "qubit": 0, "angle": 1}]
        original = dict(ops[0])
        run_circuit(1, 1, ops)
        self.assertEqual(ops[0], original)

    def test_seeded_sampling_is_isolated(self):
        ops = [
            {"op": "H", "qubit": 0},
            {"op": "measure", "qubit": 0, "cbit": 0},
        ]
        result = run_circuit(1, 1, ops)
        self.assertEqual(result.sample(8, seed=123), result.sample(8, seed=123))
        np.random.seed(999)
        result.sample(5, seed=1)
        self.assertEqual(np.random.get_state()[1][0], 999)

    def test_invalid_circuits_report_operation_position(self):
        cases = [
            (2, 1, [{"op": "H", "qubit": 2}]),
            (2, 1, [{"op": "CX", "control": 1, "target": 1}]),
            (1, 1, [{"op": "NOPE", "qubit": 0}]),
            (
                1,
                1,
                [{"op": "H", "qubit": 0}, {"op": "measure", "qubit": 9, "cbit": 0}],
            ),
            (1, 1, [{"op": "RZ", "qubit": 0, "angle": float("nan")}]),
            (1, 1, [{"op": "phase_flip", "qubit": 0, "p": 1.1}]),
            (1, 1, [{"op": "measure", "qubit": 0, "cbit": 2}]),
            (
                1,
                1,
                [{"op": "X", "qubit": 0, "condition": {"cbit": 2, "equals": 0}}],
            ),
        ]
        for nq, nc, ops in cases:
            with self.subTest(ops=ops):
                with self.assertRaises(CircuitError) as context:
                    run_circuit(nq, nc, ops)
                self.assertIn("operation", str(context.exception))

    def test_invalid_dimensions(self):
        with self.assertRaises(CircuitError):
            run_circuit(0, 1, [])
        with self.assertRaises(CircuitError):
            run_circuit(1, 7, [])


if __name__ == "__main__":
    unittest.main()
