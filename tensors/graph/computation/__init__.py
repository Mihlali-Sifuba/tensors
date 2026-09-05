"""The executable, differentiable form of a computational graph.

The graph package records structure; this package executes it. A
:class:`Computation` plans a recorded graph into instructions and runs them
forwards and in reverse, ``autograd`` is the functional interface through
which callers request differentiation, ``derivatives`` builds Jacobians and
Hessians from repeated reverse passes, ``gradcheck`` verifies them against
finite differences, and ``fusion`` accelerates compatible instruction runs.

This module is the internal façade the rest of ``tensors.graph`` imports from.
The execution machinery behind these names — ``Instruction``, the fusion
helpers, and the cached-plan lookup — stays inside the package.
"""

from .computation import Computation
from .autograd import backward, grad
from .derivatives import hessian, jacobian
from .gradcheck import GradcheckError, gradcheck

__all__ = [
    "Computation", "GradcheckError", "backward", "grad", "gradcheck",
    "hessian", "jacobian",
]
