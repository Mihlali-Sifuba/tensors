"""The package facades decide the supported API, not leading underscores.

`tensors.backend` and `tensors` are the facades. A name they re-export is
supported; a name they do not is internal, whether or not it starts with an
underscore. Implementation modules therefore name a function for the operation
it performs, and keep underscores for state no caller may touch.

These tests pin that boundary so a later rename cannot widen the supported API
by accident, and so removing an underscore from a helper does not publish it.
"""

import unittest

import tensors as ts
import tensors.backend as backend
from tensors.backend import config

#: Selection operations the facades intend to support.
SUPPORTED_SELECTION = (
    "available_backends",
    "get_backend",
    "set_backend",
    "use_backend",
)

#: Errors the facades intend to support.
SUPPORTED_ERRORS = (
    "BackendMismatchError",
    "BackendOperationUnsupportedError",
    "BackendUnavailableError",
)

#: Operations that live in ``config`` but are not part of the supported API.
#: They carry ordinary names because they are ordinary operations; the facades
#: are what keep them internal.
INTERNAL_OPERATIONS = (
    "numpy_available",
    "cuda_available",
    "resolve_backend",
    "environment_default",
)

#: Selection state. Not an operation any caller may perform, so it stays
#: underscore-private and is mutated only through set_backend/use_backend.
PRIVATE_STATE = (
    "_process_backend",
    "_backend_override",
    "_backend_lock",
    "_VALID_BACKENDS",
)


class SupportedApiTests(unittest.TestCase):
    def test_the_selection_api_is_reachable_from_both_facades(self):
        for name in SUPPORTED_SELECTION + SUPPORTED_ERRORS:
            with self.subTest(name=name):
                self.assertTrue(hasattr(backend, name), f"tensors.backend.{name}")
                self.assertTrue(hasattr(ts, name), f"tensors.{name}")

    def test_the_backend_facade_declares_the_selection_api(self):
        for name in SUPPORTED_SELECTION + SUPPORTED_ERRORS:
            with self.subTest(name=name):
                self.assertIn(name, backend.__all__)

    def test_the_package_facade_declares_the_selection_api(self):
        for name in SUPPORTED_SELECTION:
            with self.subTest(name=name):
                self.assertIn(name, ts.__all__)

    def test_both_facades_expose_the_same_selection_objects(self):
        for name in SUPPORTED_SELECTION:
            with self.subTest(name=name):
                self.assertIs(getattr(ts, name), getattr(backend, name))
                self.assertIs(getattr(backend, name), getattr(config, name))


class InternalOperationTests(unittest.TestCase):
    def test_internal_operations_carry_ordinary_names(self):
        """A helper is named for what it does, not for how public it is."""
        for name in INTERNAL_OPERATIONS:
            with self.subTest(name=name):
                self.assertTrue(hasattr(config, name))
                self.assertFalse(
                    name.startswith("_"),
                    "an ordinary operation should not be named with an underscore",
                )

    def test_internal_operations_are_not_published_by_either_facade(self):
        for name in INTERNAL_OPERATIONS:
            with self.subTest(name=name):
                self.assertFalse(
                    hasattr(backend, name),
                    f"tensors.backend published {name}; the facade decides the API",
                )
                self.assertFalse(hasattr(ts, name), f"tensors published {name}")
                self.assertNotIn(name, backend.__all__)
                self.assertNotIn(name, ts.__all__)

    def test_internal_operations_stay_reachable_through_their_module(self):
        """Internal does not mean hidden: the owning module still offers them."""
        self.assertIsInstance(config.numpy_available(), bool)
        self.assertIsInstance(config.cuda_available(), bool)
        self.assertEqual(config.resolve_backend("python"), "python")
        self.assertIn(config.environment_default(), ts.available_backends())


class PrivateStateTests(unittest.TestCase):
    def test_selection_state_stays_private(self):
        for name in PRIVATE_STATE:
            with self.subTest(name=name):
                self.assertTrue(hasattr(config, name), f"config.{name} should exist")
                self.assertTrue(name.startswith("_"), "state stays underscore-private")
                self.assertFalse(hasattr(backend, name))
                self.assertFalse(hasattr(ts, name))

    def test_selection_behaviour_is_unchanged(self):
        previous = ts.get_backend()
        try:
            for name in ts.available_backends():
                with self.subTest(backend=name):
                    with ts.use_backend(name):
                        self.assertEqual(ts.get_backend(), name)
                    self.assertEqual(ts.get_backend(), previous)
            ts.set_backend("python")
            self.assertEqual(ts.get_backend(), "python")
            with self.assertRaises(ValueError):
                ts.set_backend("nonsense")
        finally:
            ts.set_backend(previous)

    def test_auto_still_resolves_to_a_concrete_backend(self):
        previous = ts.get_backend()
        try:
            with ts.use_backend("auto"):
                self.assertIn(ts.get_backend(), ("numpy", "python"))
        finally:
            ts.set_backend(previous)


if __name__ == "__main__":
    unittest.main()
