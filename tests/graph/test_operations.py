import unittest

import tensors as ts
from tensors.ops import Operation
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


class OperationOwnershipTests(unittest.TestCase):
    """Operation belongs to the operations subsystem, not the graph."""

    def test_operation_lives_in_the_ops_subsystem(self):
        import tensors.ops.operation as module

        self.assertIs(Operation, module.Operation)
        self.assertIs(ts.ops.Operation, Operation)
        self.assertEqual(Operation.__module__, "tensors.ops.operation")

    def test_the_graph_package_does_not_define_an_operation(self):
        import tensors.graph as graph

        self.assertFalse(hasattr(graph, "Operation"))
        self.assertNotIn("Operation", graph.__all__)
        with self.assertRaises(ModuleNotFoundError):
            __import__("tensors.graph.operation")

    def test_every_concrete_operation_inherits_the_moved_base(self):
        stack = [Operation]
        concrete = []
        while stack:
            for subclass in stack.pop().__subclasses__():
                stack.append(subclass)
                concrete.append(subclass)
        self.assertGreater(len(concrete), 40)
        for subclass in concrete:
            with self.subTest(operation=subclass.__name__):
                self.assertTrue(issubclass(subclass, Operation))

    def test_graph_and_execution_forms_both_reference_an_operation(self):
        from tensors.graph import OperationNode
        from tensors.graph.computation.computation import (
            Computation,
            Instruction,
        )

        left = ts.Variable([2.0])
        right = ts.Variable([3.0])
        result = left * right

        node = result.node.producer
        self.assertIsInstance(node, OperationNode)
        self.assertIsInstance(node.operation, Operation)

        instruction = Computation(result)._instructions[0]
        self.assertIsInstance(instruction, Instruction)
        self.assertIsInstance(instruction.operation, Operation)
        self.assertIs(instruction.operation, node.operation)

    def test_graph_execution_still_works_through_the_moved_base(self):
        value = ts.Variable([2.0])
        output = ts.sum(value * 3.0 + 1.0)

        ts.backward(output)

        self.assertEqual(output.data.tolist(), [7.0])
        self.assertEqual(value.grad.tolist(), [3.0])


if __name__ == "__main__":
    unittest.main()
