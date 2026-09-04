import importlib
import importlib.util
import subprocess
import sys
import unittest

import tensors as ts
from tensors import graph as graph_package
from tensors.graph import computation as computation_package


class ComputationPackageLayoutTests(unittest.TestCase):
    """The executable form of a graph lives in its own subpackage."""

    #: The modules the computation subpackage owns.
    MODULES = (
        "computation",
        "autograd",
        "derivatives",
        "gradcheck",
        "fusion",
    )

    def test_computation_is_a_package(self):
        self.assertTrue(hasattr(computation_package, "__path__"))
        for name in self.MODULES:
            with self.subTest(module=name):
                module = importlib.import_module(
                    f"tensors.graph.computation.{name}"
                )
                self.assertEqual(
                    module.__name__, f"tensors.graph.computation.{name}"
                )

    def test_graph_package_no_longer_owns_the_execution_modules(self):
        for name in ("autograd", "derivatives", "gradcheck", "fusion"):
            with self.subTest(module=name):
                self.assertIsNone(
                    importlib.util.find_spec(f"tensors.graph.{name}")
                )

    def test_implementations_belong_to_the_new_modules(self):
        expected = {
            ts.graph.Computation: "tensors.graph.computation.computation",
            ts.graph.backward: "tensors.graph.computation.autograd",
            ts.graph.grad: "tensors.graph.computation.autograd",
            ts.graph.jacobian: "tensors.graph.computation.derivatives",
            ts.graph.hessian: "tensors.graph.computation.derivatives",
            ts.graph.gradcheck: "tensors.graph.computation.gradcheck",
            ts.graph.GradcheckError: "tensors.graph.computation.gradcheck",
        }
        for value, module in expected.items():
            with self.subTest(name=getattr(value, "__name__", value)):
                self.assertEqual(value.__module__, module)

        from tensors.graph.computation.computation import Instruction

        self.assertEqual(
            Instruction.__module__, "tensors.graph.computation.computation"
        )

    def test_subpackage_facade_exposes_what_the_graph_package_needs(self):
        self.assertEqual(
            sorted(computation_package.__all__),
            [
                "Computation", "GradcheckError", "backward", "grad",
                "gradcheck", "hessian", "jacobian",
            ],
        )
        for name in computation_package.__all__:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(computation_package, name),
                    getattr(graph_package, name),
                )

    def test_execution_machinery_stays_inside_the_subpackage(self):
        internals = (
            "Instruction",
            "computation_for",
            "plan_fusions",
            "fused_operation",
            "start_fusion",
            "extend_fusion",
            "execute_fused_forward",
            "execute_fused_backward",
            "Fusion",
            "FusedStep",
        )
        for name in internals:
            with self.subTest(name=name):
                self.assertNotIn(name, computation_package.__all__)
                self.assertNotIn(name, graph_package.__all__)
                self.assertFalse(hasattr(graph_package, name))
                self.assertFalse(hasattr(ts, name))

    def test_every_module_imports_first_without_a_cycle(self):
        # Importing each module as the very first import of a fresh
        # interpreter fails loudly if the package grew an import cycle.
        for name in self.MODULES:
            with self.subTest(module=name):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        f"import tensors.graph.computation.{name}",
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


class PublicApiTests(unittest.TestCase):
    """The move is invisible from outside the package."""

    def test_graph_namespace_is_unchanged(self):
        self.assertEqual(
            sorted(ts.graph.__all__),
            [
                "Computation", "Edge", "GradcheckError", "Graph", "Node",
                "OperationNode", "VariableNode", "backward", "grad",
                "gradcheck", "hessian", "jacobian",
            ],
        )
        for name in ts.graph.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(ts.graph, name))

    def test_root_namespace_still_reexports_the_functional_api(self):
        for name in ("backward", "grad", "gradcheck", "hessian", "jacobian"):
            with self.subTest(name=name):
                self.assertIn(name, ts.__all__)
                self.assertIs(getattr(ts, name), getattr(ts.graph, name))

    def test_documented_import_paths_still_resolve(self):
        from tensors import backward, grad, gradcheck, hessian, jacobian
        from tensors.graph import Computation, GradcheckError

        self.assertIs(backward, ts.graph.backward)
        self.assertIs(grad, ts.graph.grad)
        self.assertIs(gradcheck, ts.graph.gradcheck)
        self.assertIs(hessian, ts.graph.hessian)
        self.assertIs(jacobian, ts.graph.jacobian)
        self.assertIs(Computation, ts.graph.Computation)
        self.assertIs(GradcheckError, ts.graph.GradcheckError)

    def test_structural_types_stay_in_the_graph_package(self):
        self.assertEqual(ts.graph.Node.__module__, "tensors.graph.node")
        self.assertEqual(ts.graph.Edge.__module__, "tensors.graph.edge")
        self.assertEqual(ts.graph.Graph.__module__, "tensors.graph.graph")


if __name__ == "__main__":
    unittest.main()
