import math
import unittest

import tensors as ts
from tensors.graph.state import get_graph_state, reset_graph_state


class GradcheckTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_gradcheck_verifies_a_composite_function(self):
        left = ts.Tensor([[0.2, -0.4], [0.7, 0.3]])
        right = ts.Tensor([[1.1, 0.8], [-0.2, 0.5]])

        result = ts.gradcheck(
            lambda x, y: ts.mean(ts.softmax(x * y + y, axis=1) ** 2.0),
            (left, right),
        )

        self.assertTrue(result)

    def test_gradcheck_verifies_axis_norm(self):
        value = ts.Tensor([[3.0, 4.0], [5.0, 12.0]])

        self.assertTrue(ts.gradcheck(lambda x: ts.norm(x, axis=1), value))

    def test_gradcheck_detects_a_detached_result(self):
        value = ts.Tensor([2.0])

        with self.assertRaisesRegex(ts.GradcheckError, "Gradient mismatch"):
            ts.gradcheck(lambda x: ts.Variable(x.data ** 2.0), value)

    def test_gradcheck_restores_the_callers_graph_state(self):
        state = get_graph_state()
        value = ts.Variable([2.0])
        original_nodes = list(state.nodes)

        ts.gradcheck(lambda x: x ** 3.0, value)

        self.assertIs(get_graph_state(), state)
        self.assertEqual(state.nodes, original_nodes)

    def test_gradcheck_can_return_false_instead_of_raising(self):
        value = ts.Tensor([2.0])

        def incorrect(value):
            detached_square = ts.Variable(value.data ** 2.0, requires_grad=False)
            return value * 0.0 + detached_square

        self.assertFalse(ts.gradcheck(incorrect, value, raise_exception=False))

    def test_gradcheck_rejects_nonfinite_settings(self):
        cases = (
            {"eps": math.nan},
            {"eps": math.inf},
            {"atol": math.nan},
            {"atol": math.inf},
            {"rtol": math.nan},
            {"rtol": math.inf},
        )

        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    ts.gradcheck(
                        lambda value: value * value,
                        ts.Tensor([1.0]),
                        **arguments,
                    )


if __name__ == "__main__":
    unittest.main()
