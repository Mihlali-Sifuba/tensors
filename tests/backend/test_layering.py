"""Architectural boundaries between the operation, dispatch, and backend layers.

The dependency direction is one-way:

    tensors.math / tensors.ops / tensors.graph
        -> tensors.backend.dispatch
            -> tensors.backend.{python,numpy,cuda}

with ``tensors.utils`` sitting below every one of them. A backend kernel that
reaches back up into the operation layer inverts that, and drags the graph and
autograd machinery into a module that should only evaluate arithmetic. These
tests read the source rather than importing it, so a violation is reported as
the offending file and line whether the import is absolute or relative,
module-level or deferred inside a function.
"""

import ast
import pathlib
import unittest


def _package_of(path: pathlib.Path) -> str:
    """Return the dotted package a module lives in."""
    return ".".join(path.parts[:-1])


def _resolve(node: ast.ImportFrom, path: pathlib.Path) -> str:
    """Return the absolute module name an ``ImportFrom`` refers to."""
    if not node.level:
        return node.module or ""
    parts = _package_of(path).split(".")
    parts = parts[: len(parts) - node.level + 1]
    return ".".join(parts + ([node.module] if node.module else []))


def imported_modules(path: pathlib.Path):
    """Yield ``(module, lineno)`` for every import, at any nesting depth."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            yield _resolve(node, path), node.lineno
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno


def modules_under(*roots: str):
    for root in roots:
        for path in sorted(pathlib.Path(root).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path


def _source_of(module: str) -> pathlib.Path | None:
    """Return the file backing a dotted ``tensors`` module, if it has one."""
    for candidate in (
        pathlib.Path(*module.split(".")).with_suffix(".py"),
        pathlib.Path(*module.split(".")) / "__init__.py",
    ):
        if candidate.exists():
            return candidate
    return None


def _closure(module: str) -> set[str]:
    """Return every ``tensors`` module reachable from ``module`` by import.

    Package ``__init__`` files are not followed: importing
    ``tensors.utils.summation`` executes ``tensors/__init__.py``, but that is
    a property of Python's package initialization rather than a dependency
    the module itself declares.
    """
    seen: set[str] = set()
    pending = [module]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        path = _source_of(current)
        if path is None or path.name == "__init__.py":
            continue
        for imported, _ in imported_modules(path):
            if imported.startswith("tensors") and imported not in seen:
                pending.append(imported)
    return seen - {module}


class BackendLayeringTests(unittest.TestCase):
    """No module under ``tensors/backend`` may import an upper layer."""

    #: Layers that sit above the backend and must never be imported from it.
    FORBIDDEN = (
        "tensors.math",
        "tensors.ops",
        "tensors.graph",
        "tensors.optim",
        "tensors.linalg",
        "tensors.nn",
        "tensors.init",
        "tensors.creation",
        "tensors.variable",
    )

    #: Deliberately empty. Every dependency the backend had on the operation
    #: layer was either a numerical primitive, which now lives in a focused
    #: neutral module under ``tensors.utils``, or a structural helper, which
    #: moved there too. Add an entry here only with a written justification.
    ALLOWED: dict[str, tuple[str, ...]] = {}

    def test_no_backend_module_imports_an_upper_layer(self):
        violations = []
        for path in modules_under("tensors/backend"):
            allowed = self.ALLOWED.get(str(path).replace("\\", "/"), ())
            for module, line in imported_modules(path):
                if not module.startswith(self.FORBIDDEN):
                    continue
                if module in allowed:
                    continue
                violations.append(
                    f"{str(path).replace(chr(92), '/')}:{line} imports {module}"
                )
        self.assertEqual(
            violations,
            [],
            "backend modules must not depend on the operation layer:\n"
            + "\n".join(violations),
        )

    def test_the_allowlist_stays_empty(self):
        """A regression guard: every entry needs a justification in review."""
        self.assertEqual(self.ALLOWED, {})


class NeutralUtilityLayeringTests(unittest.TestCase):
    """``tensors/utils`` sits below every other layer and stays there."""

    FORBIDDEN = (
        "tensors.backend",
        "tensors.dispatch",
        "tensors.math",
        "tensors.ops",
        "tensors.graph",
        "tensors.optim",
        "tensors.linalg",
        "tensors.nn",
        "tensors.init",
        "tensors.creation",
        "tensors.variable",
    )

    def test_no_utility_imports_a_layer_above_it(self):
        violations = []
        for path in modules_under("tensors/utils"):
            for module, line in imported_modules(path):
                if module.startswith(self.FORBIDDEN):
                    violations.append(
                        f"{str(path).replace(chr(92), '/')}:{line} imports {module}"
                    )
        self.assertEqual(
            violations,
            [],
            "neutral utilities must not depend on any layer above them:\n"
            + "\n".join(violations),
        )

    def test_the_relocated_primitives_need_no_tensor(self):
        """The modules this boundary was introduced for stay value-agnostic.

        ``broadcasting`` legitimately materializes Tensors, so it is excluded;
        the primitives below take plain numbers, sequences, and shapes.
        """
        relocated = (
            "tensors/utils/convolution.py",
            "tensors/utils/deviation.py",
            "tensors/utils/normalization.py",
            "tensors/utils/reductions.py",
            "tensors/utils/summation.py",
        )
        for name in relocated:
            path = pathlib.Path(name)
            with self.subTest(module=name):
                self.assertTrue(path.exists(), f"{name} is missing")
                modules = [module for module, _ in imported_modules(path)]
                self.assertNotIn("tensors.tensor", modules)

    def test_the_relocated_dependency_closure_stays_below_every_layer(self):
        """Follow each module's imports transitively, not just one level.

        A runtime probe cannot show this: importing ``tensors.utils.summation``
        makes Python execute ``tensors/__init__.py`` first, which loads the
        whole public API regardless of what the module itself needs. Walking
        the sources instead measures the module's own dependency closure.
        """
        for name in (
            "tensors.utils.convolution",
            "tensors.utils.deviation",
            "tensors.utils.normalization",
            "tensors.utils.reductions",
            "tensors.utils.summation",
        ):
            with self.subTest(module=name):
                reached = _closure(name)
                offenders = sorted(
                    module for module in reached if module.startswith(self.FORBIDDEN)
                )
                self.assertEqual(
                    offenders,
                    [],
                    f"{name} transitively depends on {offenders}",
                )


if __name__ == "__main__":
    unittest.main()
