import unittest

import tensors as ts
from tensors.graph import Operation
from tensors.math.max import Max
from tensors.math.min import Min
from tensors.math.sum import Sum
from tensors.ops import Add, Div


class OperationContractTests(unittest.TestCase):
    def test_concrete_operations_inherit_the_operation_base(self):
        for operation in (Add(), Div(), Max(), Min()):
            with self.subTest(operation=operation.name):
                self.assertIsInstance(operation, Operation)

    def test_operation_requires_forward_and_backward(self):
        class ForwardOnly(Operation):
            def forward(self, value):
                return value

        with self.assertRaisesRegex(TypeError, "abstract"):
            ForwardOnly()

    def test_backward_graph_is_optional(self):
        class Identity(Operation):
            name = "identity"

            def forward(self, value):
                return value

            def backward(self, gradient, value, *, needs_input_grad):
                return [gradient]

        operation = Identity()
        value = ts.Tensor([1.0])

        self.assertIs(operation.forward(value), value)
        with self.assertRaisesRegex(
            NotImplementedError,
            "Higher-order derivatives are not implemented for identity",
        ):
            operation.backward_graph(
                value,
                value,
                needs_input_grad=(True,),
            )

    def test_operation_instances_are_immutable(self):
        operation = Sum(axis=(1,), keepdims=True)

        self.assertEqual(operation.axis, (1,))
        self.assertTrue(operation.keepdims)
        with self.assertRaisesRegex(AttributeError, "immutable"):
            operation.axis = (0,)
        with self.assertRaisesRegex(AttributeError, "immutable"):
            del operation.keepdims

    def test_operations_do_not_retain_forward_execution_state(self):
        operation = Sum(axis=None, keepdims=False)
        operation.forward(ts.Tensor([1.0, 2.0]))

        self.assertEqual(
            sorted(type(operation).__slots__),
            ["axis", "keepdims"],
        )
        for name in ("axis", "keepdims"):
            self.assertNotIsInstance(getattr(operation, name), ts.Tensor)


if __name__ == "__main__":
    unittest.main()
