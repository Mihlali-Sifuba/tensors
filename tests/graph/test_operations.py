import unittest

import tensors as ts
from tensors.ops import Operation
from tensors.operations.reductions.max import Max
from tensors.operations.reductions.min import Min
from tensors.operations.reductions.sum import Sum
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

    def test_an_operation_defines_one_derivative(self):
        """``backward`` is the whole contract; there is no second method.

        It used to be joined by ``backward_graph``, which a reverse pass
        building a derivative graph called instead. Nothing selects between
        them now, so an operation that implements ``backward`` against
        operations rather than against Tensors serves both passes, and one
        that does not simply answers the numerical pass.
        """

        class Identity(Operation):
            name = "identity"

            def forward(self, value):
                return value

            def backward(self, gradient, value, *, needs_input_grad):
                return [gradient]

        operation = Identity()
        value = ts.Tensor([1.0])

        self.assertIs(operation.forward(value), value)
        self.assertFalse(hasattr(operation, "backward_graph"))
        self.assertEqual(
            [name for name in dir(Operation) if "backward" in name], ["backward"]
        )

    def test_no_operation_keeps_a_second_derivative_method(self):
        """The superseded method is gone from every subclass, not just the base."""
        stack, concrete = [Operation], []
        while stack:
            for subclass in stack.pop().__subclasses__():
                stack.append(subclass)
                concrete.append(subclass)
        self.assertGreater(len(concrete), 40)
        offenders = [
            subclass.__name__
            for subclass in concrete
            if "backward_graph" in vars(subclass)
        ]
        self.assertEqual(offenders, [])

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

        # Every slot is configuration the operation was built with.
        # ``on_selected_backend`` is one: it is the execution contract the
        # addition VJP asks its reduction for, held here so a recorded
        # reduction replays under the same contract. None of them may hold a
        # value the forward pass produced.
        self.assertEqual(
            sorted(type(operation).__slots__),
            ["axis", "keepdims", "on_selected_backend"],
        )
        self.assertFalse(Sum(axis=None, keepdims=False).on_selected_backend)
        for name in type(operation).__slots__:
            self.assertNotIsInstance(getattr(operation, name), ts.Tensor)


class OperationOwnershipTests(unittest.TestCase):
    """Operation belongs to the operations subsystem, not the graph."""

    def test_operation_lives_in_the_ops_subsystem(self):
        import tensors.operations.base as module

        self.assertIs(Operation, module.Operation)
        self.assertIs(ts.ops.Operation, Operation)
        self.assertEqual(Operation.__module__, "tensors.operations.base")

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
        from tensors.graph.computation.computation import Computation
        from tensors.graph.computation.instruction import Instruction

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
