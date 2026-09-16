"""Architectural boundaries around the semantic operation hierarchy.

``tensors.operations`` is the canonical home of every computation the library
exposes. It sits above dispatch and below nothing::

    tensors.operations
        -> tensors.backend.dispatch
            -> tensors.backend.{python,numpy,cuda}

with ``tensors.utils`` below all of them. An operation module names *what* a
computation means and hands the arithmetic to dispatch; choosing a backend is
the job of dispatch, so an operation that imports ``tensors.backend.numpy``
has taken a decision that is not its own.

The one upward dependency operations do have is the graph operand boundary: a
public entry point asks whether it was handed a graph value and, if so,
records the application instead of evaluating it.
:class:`OperationGraphBoundaryTests` pins that to five names in two modules.

The opposite direction -- ``tensors.graph`` naming concrete operation classes
-- is a known inversion and is deliberately **not** enforced here. It is
recorded as a separate follow-up so that this migration stays a structural
move.

These tests read source with :mod:`ast` rather than importing it. A runtime
probe cannot answer the question: importing any submodule of ``tensors`` runs
``tensors/__init__.py`` first, which loads the whole public API regardless of
what the module under test actually needs. Reading the source also catches an
import deferred inside a function, which is where the interesting ones hide.
"""

import ast
import pathlib
import subprocess
import sys
import unittest

import tensors
import tensors.linalg
import tensors.math
import tensors.ops
from tests.backend.test_layering import imported_modules, modules_under

OPERATIONS = "tensors/operations"


def _posix(path: pathlib.Path) -> str:
    return str(path).replace("\\", "/")


def _definitions(path: pathlib.Path):
    """Yield the public classes and entry points a module defines."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign) and isinstance(
            node.value, (ast.Attribute, ast.Call)
        ):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        else:
            continue
        for name in names:
            if not name.startswith("_"):
                yield name


class OperationBackendBoundaryTests(unittest.TestCase):
    """An operation asks dispatch to execute; it never picks the executor."""

    #: The backends themselves. Only ``tensors.backend.dispatch`` and the
    #: ``execute_*`` names re-exported from ``tensors.backend`` are for
    #: operations to call.
    CONCRETE = (
        "tensors.backend.python",
        "tensors.backend.numpy",
        "tensors.backend.cuda",
    )

    def test_no_operation_imports_a_concrete_backend(self):
        violations = []
        for path in modules_under(OPERATIONS):
            for module, line in imported_modules(path):
                if module.startswith(self.CONCRETE):
                    violations.append(f"{_posix(path)}:{line} imports {module}")
        self.assertEqual(
            violations,
            [],
            "operations must reach the backends through dispatch:\n"
            + "\n".join(violations),
        )


class OperationGraphBoundaryTests(unittest.TestCase):
    """Operations touch the graph only to route a graph operand."""

    #: Every name an operation module may import from ``tensors.graph``.
    #: ``as_tensor_operand`` coerces an input, ``is_graph_operand`` and
    #: ``VariableNode`` recognize one, and ``as_graph_operand`` with
    #: ``apply_operation`` record the application instead of evaluating it.
    BOUNDARY = {
        "tensors.graph.expression": frozenset(
            {
                "apply_operation",
                "as_graph_operand",
                "as_tensor_operand",
                "is_graph_operand",
            }
        ),
        "tensors.graph.node": frozenset({"VariableNode"}),
    }

    def test_operations_import_no_other_graph_module(self):
        violations = []
        for path in modules_under(OPERATIONS):
            for module, line in imported_modules(path):
                if not module.startswith("tensors.graph"):
                    continue
                if module not in self.BOUNDARY:
                    violations.append(f"{_posix(path)}:{line} imports {module}")
        self.assertEqual(
            violations,
            [],
            "the operand boundary is the only graph dependency allowed:\n"
            + "\n".join(violations),
        )

    def test_operations_import_no_other_graph_name(self):
        violations = []
        for path in modules_under(OPERATIONS):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                allowed = self.BOUNDARY.get(node.module or "")
                if allowed is None:
                    continue
                for alias in node.names:
                    if alias.name not in allowed:
                        violations.append(
                            f"{_posix(path)}:{node.lineno} imports {alias.name}"
                        )
        self.assertEqual(
            violations,
            [],
            "an operation may only import the operand boundary names:\n"
            + "\n".join(violations),
        )

    def test_the_boundary_names_all_exist(self):
        """A stale allowlist silently stops constraining anything."""
        for module, names in self.BOUNDARY.items():
            source = pathlib.Path(*module.split(".")).with_suffix(".py")
            defined = {
                node.name
                for node in ast.parse(source.read_text(encoding="utf-8")).body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            }
            for name in names:
                with self.subTest(module=module, name=name):
                    self.assertIn(name, defined)


class ObsoleteModuleTests(unittest.TestCase):
    """The old implementation locations are gone, not forwarded."""

    #: A sample of the paths that used to hold implementations, one or more
    #: per retired package, named explicitly so the test says what it means.
    RETIRED = (
        "tensors/ops/operation.py",
        "tensors/ops/add.py",
        "tensors/ops/pow.py",
        "tensors/ops/_utils.py",
        "tensors/math/sum.py",
        "tensors/math/exp.py",
        "tensors/math/comparison.py",
        "tensors/math/convolution.py",
        "tensors/linalg/dot.py",
        "tensors/linalg/matmul.py",
        "tensors/linalg/norm.py",
    )

    def test_the_retired_implementation_modules_are_gone(self):
        for name in self.RETIRED:
            with self.subTest(module=name):
                self.assertFalse(
                    pathlib.Path(name).exists(),
                    f"{name} should have moved into tensors/operations",
                )

    def test_the_facades_hold_nothing_but_their_re_exports(self):
        """A facade with a module in it is an implementation owner again."""
        for package in ("tensors/math", "tensors/ops", "tensors/linalg"):
            with self.subTest(package=package):
                modules = sorted(
                    path.name for path in pathlib.Path(package).glob("*.py")
                )
                self.assertEqual(modules, ["__init__.py"])


class FacadeDirectionTests(unittest.TestCase):
    """The facades are for users of the library, not for the library.

    ``tensors.math``, ``tensors.ops``, and ``tensors.linalg`` re-export from
    ``tensors.operations``, so an operation that imports one of them depends
    on a module that depends on it. The import works, because these are all
    deferred inside functions, but it makes the canonical location ambiguous:
    ``sum`` would have two import paths with no rule saying which is meant.

    Internal code names the canonical module; the facades face outward.
    """

    FACADES = ("tensors.math", "tensors.ops", "tensors.linalg")

    #: ``tensors/__init__.py`` assembles the root namespace from the three
    #: documented namespaces. That is one public surface composing another,
    #: not an implementation reaching sideways through one.
    COMPOSES_THE_PUBLIC_API = ("tensors/__init__.py",)

    def _violations(self, *roots):
        found = []
        for path in modules_under(*roots):
            posix = _posix(path)
            if posix in self.COMPOSES_THE_PUBLIC_API:
                continue
            if posix in {f"{f.replace('.', '/')}/__init__.py" for f in self.FACADES}:
                continue  # a facade re-exporting is the point of a facade
            for module, line in imported_modules(path):
                if module in self.FACADES or module.startswith(
                    tuple(f"{name}." for name in self.FACADES)
                ):
                    found.append(f"{posix}:{line} imports {module}")
        return found

    def test_no_operation_imports_a_facade(self):
        violations = self._violations(OPERATIONS)
        self.assertEqual(
            violations,
            [],
            "an operation must name its canonical sibling, not a facade:\n"
            + "\n".join(violations),
        )

    def test_no_internal_module_imports_a_facade(self):
        """The same rule for the rest of the package.

        ``Tensor.__abs__`` and ``Tensor.__matmul__`` reached through
        ``tensors.math`` and ``tensors.linalg`` while their neighbours already
        named the operation module, so the guard covers every internal module
        rather than only the operation layer.
        """
        violations = self._violations("tensors")
        self.assertEqual(
            violations,
            [],
            "internal modules must name canonical operation modules:\n"
            + "\n".join(violations),
        )


class FacadeResolutionTests(unittest.TestCase):
    """Every documented namespace is a view onto the canonical objects."""

    FACADES = ("tensors.math", "tensors.linalg", "tensors.ops")

    @staticmethod
    def _defining_module(value):
        module = getattr(value, "__module__", None)
        if module is None and hasattr(value, "__self__"):
            # ``add = Add().forward`` exports a bound method.
            module = type(value.__self__).__module__
        return module

    def test_every_facade_export_resolves_into_operations(self):
        for name in self.FACADES:
            facade = sys.modules[name]
            for exported in facade.__all__:
                with self.subTest(facade=name, export=exported):
                    module = self._defining_module(getattr(facade, exported))
                    self.assertIsNotNone(module, f"{exported} has no module")
                    self.assertTrue(
                        module.startswith("tensors.operations"),
                        f"{name}.{exported} comes from {module}",
                    )

    def test_a_facade_export_is_the_same_object_as_the_canonical_one(self):
        """Re-export, not re-definition: identity has to hold."""
        pairs = (
            (tensors.operations.elementary.exp, "tensors.operations.elementary", "exp"),
            (
                tensors.operations.reductions.mean,
                "tensors.operations.reductions",
                "mean",
            ),
            (tensors.operations.linalg.matmul, "tensors.operations.linalg", "matmul"),
            (
                tensors.operations.reductions.norm,
                "tensors.operations.reductions",
                "norm",
            ),
            (tensors.ops.Operation, "tensors.operations", "Operation"),
            (tensors.add, "tensors.operations.arithmetic", "add"),
            (tensors.matmul, "tensors.operations.linalg", "matmul"),
            (tensors.dot, "tensors.operations.linalg", "dot"),
        )
        for value, module, name in pairs:
            with self.subTest(canonical=f"{module}.{name}"):
                canonical = getattr(__import__(module, fromlist=[name]), name)
                self.assertIs(value, canonical)


class CanonicalOwnerTests(unittest.TestCase):
    """One operation, one implementation module."""

    def test_no_public_name_is_defined_in_two_operation_modules(self):
        owners: dict[str, list[str]] = {}
        for path in modules_under(OPERATIONS):
            if path.name == "__init__.py":
                continue
            # A name is defined once per module however many ``@overload``
            # stubs precede the implementation.
            for name in sorted(set(_definitions(path))):
                owners.setdefault(name, []).append(_posix(path))
        duplicates = {
            name: places for name, places in owners.items() if len(places) > 1
        }
        self.assertEqual(
            duplicates,
            {},
            "each operation needs exactly one owner:\n"
            + "\n".join(f"{n}: {p}" for n, p in sorted(duplicates.items())),
        )

    def test_matmul_and_dot_are_one_operation(self):
        """``ts.dot`` is a second name for ``MatMul``, not a second product."""
        from tensors.operations.linalg import MatMul, dot, matmul

        self.assertIs(matmul, tensors.matmul)
        self.assertIs(dot, tensors.dot)
        self.assertEqual(MatMul.name, "matmul")
        left = tensors.Tensor([[1.0, 2.0], [3.0, 4.0]])
        right = tensors.Tensor([[5.0, 6.0], [7.0, 8.0]])
        self.assertEqual(dot(left, right).tolist(), matmul(left, right).tolist())

    def test_the_operation_base_class_lives_in_operations_base(self):
        """Fusion and graph metadata key on the class, so identity matters."""
        self.assertEqual(tensors.ops.Operation.__module__, "tensors.operations.base")


class LazyOptionalBackendTests(unittest.TestCase):
    """Selecting the Python backend must not import NumPy or CuPy."""

    def _probe(self, expression):
        result = subprocess.run(
            [sys.executable, "-c", f"import sys, tensors; print({expression})"],
            capture_output=True,
            text=True,
            cwd=str(pathlib.Path.cwd()),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def test_importing_tensors_loads_no_optional_dependency(self):
        loaded = self._probe("[m for m in ('numpy', 'cupy') if m in sys.modules]")
        self.assertEqual(loaded, "[]")

    def test_no_kernel_module_is_imported_before_it_executes(self):
        """The storage classes load eagerly; they are needed for isinstance
        checks in :mod:`tensors.tensor` and import NumPy through
        :mod:`importlib` only when a buffer is actually allocated. The kernel
        packages are what each backend promises to defer until execution.
        """
        loaded = self._probe(
            "sorted(m for m in sys.modules if m.startswith("
            "('tensors.backend.numpy.kernels', 'tensors.backend.cuda.kernels')))"
        )
        self.assertEqual(loaded, "[]")


if __name__ == "__main__":
    unittest.main()
