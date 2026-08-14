import unittest

import tensors as ts
from tensors.math.max import Max
from tensors.ops import Add, Div


class OperationProtocolTests(unittest.TestCase):
    def test_operations_satisfy_protocols_structurally(self):
        self.assertIsInstance(Add, ts.graph.Operation)
        self.assertIsInstance(Add, ts.graph.HigherOrderOperation)
        self.assertIsInstance(Div, ts.graph.ReverseOperation)
        self.assertNotIsInstance(Max, ts.graph.HigherOrderOperation)

    def test_node_rejects_an_incomplete_operation_interface(self):
        class ForwardOnly:
            @staticmethod
            def forward(value):
                return value

        with self.assertRaisesRegex(TypeError, "forward.*backward"):
            ts.graph.Node(label="incomplete", op_cls=ForwardOnly)

    def test_operation_does_not_require_protocol_inheritance(self):
        class Identity:
            @staticmethod
            def forward(value):
                return value

            @staticmethod
            def backward(gradient, value):
                return [gradient]

        node = ts.graph.Node(label="identity", op_cls=Identity)

        self.assertIs(node.op_cls, Identity)
        self.assertIsInstance(Identity, ts.graph.Operation)


if __name__ == "__main__":
    unittest.main()
