"""The semantic operation hierarchy.

Every computation the library exposes is defined here, grouped by what the
operation *means* rather than by how a backend executes it or by which Python
syntax reaches it. An operation module owns its input and configuration
validation, its output shape and dtype, the dispatch call that evaluates it,
the storage it wraps, and its reverse-mode and higher-order derivative rules.

The layers below it have narrower jobs: :mod:`tensors.backend.dispatch`
selects an execution path, the backend kernel packages perform the arithmetic,
and :mod:`tensors.utils` holds backend-neutral primitives.

This package deliberately exports only the operation base class. Importing a
concrete operation from here would pull in every domain, and the domains reach
back into :mod:`tensors.graph`, which reaches back into the operations it
builds nodes for. Import a concrete operation from its own domain instead::

    from tensors.operations.arithmetic import Add
    from tensors.operations.linalg import matmul

``tensors.math``, ``tensors.linalg`` and ``tensors.ops`` remain as convenience
namespaces over those domains. This package is the canonical location.
"""

from __future__ import annotations

from tensors.operations.base import Operation, UNARY_DEMAND

__all__ = [
    "Operation",
    "UNARY_DEMAND",
]
