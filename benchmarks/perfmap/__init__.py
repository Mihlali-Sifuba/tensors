"""A layered performance map of the tensors package.

This package measures the same computations at every layer of the execution
stack so that overhead can be attributed to a layer rather than merely
observed. It is measurement only: it imports the package under test and
never modifies it.
"""

__all__ = ["analysis", "harness", "memory", "meta", "registry", "report",
           "timing", "workloads"]
