import threading
import unittest
from unittest.mock import patch

import tensors as ts
import tensors.backend as backend_state

from ._support import BackendTestCase


class BackendSelectionTests(BackendTestCase):
    """Process default, environment resolution, and scoped overrides."""

    def test_python_backend_is_always_available(self):
        self.assertIn("python", ts.available_backends())
    def test_set_backend_changes_process_default(self):
        ts.set_backend("python")

        self.assertEqual(ts.get_backend(), "python")
    def test_auto_backend_falls_back_to_python(self):
        with patch.object(backend_state, "_numpy_available", return_value=False):
            ts.set_backend("auto")

        self.assertEqual(ts.get_backend(), "python")
    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            ts.set_backend("missing")  # type: ignore[arg-type]
    def test_unavailable_numpy_backend_has_install_guidance(self):
        with patch.object(backend_state, "_numpy_available", return_value=False):
            with self.assertRaisesRegex(
                ts.BackendUnavailableError,
                r"ms-tensors\[numpy\]",
            ):
                ts.set_backend("numpy")
    def test_unavailable_cuda_backend_has_install_guidance(self):
        with patch.object(backend_state, "_cuda_available", return_value=False):
            with self.assertRaisesRegex(
                ts.BackendUnavailableError,
                r"ms-tensors\[cuda1[23]\]",
            ):
                ts.set_backend("cuda")
    @unittest.skipUnless(
        "numpy" in ts.available_backends(),
        "NumPy is not installed",
    )
    def test_context_override_is_nested_and_restored(self):
        ts.set_backend("python")

        with ts.use_backend("numpy"):
            self.assertEqual(ts.get_backend(), "numpy")
            with ts.use_backend("python"):
                self.assertEqual(ts.get_backend(), "python")
            self.assertEqual(ts.get_backend(), "numpy")

        self.assertEqual(ts.get_backend(), "python")
    @unittest.skipUnless(
        "numpy" in ts.available_backends(),
        "NumPy is not installed",
    )
    def test_context_override_is_restored_after_an_error(self):
        ts.set_backend("python")

        with self.assertRaisesRegex(RuntimeError, "stop"):
            with ts.use_backend("numpy"):
                raise RuntimeError("stop")

        self.assertEqual(ts.get_backend(), "python")
    @unittest.skipUnless(
        "numpy" in ts.available_backends(),
        "NumPy is not installed",
    )
    def test_context_override_does_not_replace_worker_default(self):
        ts.set_backend("python")
        observed = []

        with ts.use_backend("numpy"):
            worker = threading.Thread(
                target=lambda: observed.append(ts.get_backend()),
            )
            worker.start()
            worker.join()
            self.assertEqual(ts.get_backend(), "numpy")

        self.assertEqual(observed, ["python"])


if __name__ == "__main__":
    unittest.main()
