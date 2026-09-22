"""The contract every concrete mathematical operation implements.

An :class:`Operation` instance describes exactly one concrete invocation. It
owns the immutable configuration that defines its mathematical transformation
(a reduction axis, a cast dtype, a slice key) and nothing else.

This is the operations subsystem's own abstraction: the graph package does not
define what an operation is, it references one.
:class:`~tensors.graph.node.OperationNode` holds an operation as its graph
form and :class:`~tensors.graph.computation.computation.Instruction` holds one
as its execution form.

An operation defines *how* a local derivative is calculated.
:class:`~tensors.graph.computation.Computation` decides *which* local
derivatives a particular reverse pass requires and passes that demand to
``backward`` and ``backward_graph`` as ``needs_input_grad``. Reverse demand is
execution state: it depends on the call being made, never on the recorded
graph, and it is never stored on an operation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

from tensors.tensor import Tensor


class Operation(ABC):
    """One concrete mathematical invocation recorded in a computational graph.

    Subclasses implement ``forward`` and ``backward``. ``backward`` is the
    one derivative an operation defines: it is written against operations
    rather than against Tensors, so the same method calculates a numerical
    VJP and records a differentiable one.

    ``backward_graph`` below is the superseded second derivative method. It
    is no longer called, and the operations that still define it have not
    been migrated yet.

    An operation is immutable once constructed: its recorded mathematical
    meaning must not change while a graph still refers to it. Subclasses that
    take configuration declare it in ``__slots__`` and assign it with
    ``object.__setattr__`` inside ``__init__``. Configuration describes the
    mathematics only; which derivatives are wanted is not configuration.
    """

    __slots__ = ()

    #: Short label used for graph inspection and error messages.
    name: ClassVar[str] = "operation"

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject mutation so a recorded invocation keeps its meaning."""
        raise AttributeError(
            f"{type(self).__name__} is immutable; construct a new operation "
            "instead of reconfiguring a recorded one"
        )

    def __delattr__(self, name: str) -> None:
        """Reject attribute removal for the same reason as assignment."""
        raise AttributeError(
            f"{type(self).__name__} is immutable; construct a new operation "
            "instead of reconfiguring a recorded one"
        )

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        """Calculate this invocation's eager output from its input tensors."""

    @abstractmethod
    def backward(
        self,
        gradient: Any,
        *inputs: Any,
        needs_input_grad: tuple[bool, ...],
    ) -> Sequence[Any | None]:
        """Return the requested VJP contributions.

        The operands say which reverse pass this is. An ordinary one hands
        over Tensors and wants values back; one building a derivative graph
        hands over Variables and wants the same statements recorded. An
        operation written against operations rather than against Tensors
        serves both without branching on the difference.

        ``needs_input_grad[i]`` states whether this reverse pass requires the
        VJP for input ``i``. The result holds one entry per input:

        - a value of the kind the operands were, where the VJP was requested,
          including a real zero when the derivative was requested and its
          value is mathematically zero;
        - ``None`` where the VJP was not requested.

        Skip the work for an unrequested input rather than calculating a value
        the caller discards, and only apply a derivative-specific domain check
        when the derivative it guards was requested.
        """

    def backward_graph(
        self,
        gradient: Any,
        *inputs: Any,
        needs_input_grad: tuple[bool, ...],
    ) -> Sequence[Any | None]:
        """Superseded: build the requested differentiable VJPs.

        :meth:`backward` now serves both reverse modes, so nothing calls this.
        It remains only for the operations that have not been migrated to a
        single derivative yet, and goes away with the last of them.
        """
        raise NotImplementedError(
            f"Higher-order derivatives are not implemented for {self.name}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


#: The demand a single-input operation always receives. ``Computation`` skips
#: an instruction whose inputs are all unrequested, so a unary VJP is only
#: ever invoked when its one input is wanted.
UNARY_DEMAND: tuple[bool, ...] = (True,)


__all__ = ["Operation", "UNARY_DEMAND"]
