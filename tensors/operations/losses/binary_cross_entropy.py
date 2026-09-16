"""Numerically stable binary cross-entropy."""

from __future__ import annotations
import math
from typing import TYPE_CHECKING, Any, List, Optional, overload
from tensors._typing import TensorData, TensorLike, TensorResult
from tensors.backend import (
    execute_binary_cross_entropy,
    execute_binary_cross_entropy_gradient,
)
from tensors.dtype import result_dtype
from tensors.operations._gradient_shaping import sum_to_shape, sum_to_shape_graph
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.graph.expression import as_tensor_operand
from tensors.utils.broadcasting import broadcast_tensors
from tensors.operations.losses.cross_entropy import Reduction, _validate_reduction

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode
    from tensors.variable import Variable


def _validate_targets(target: Tensor) -> None:
    if any((not 0.0 <= float(value) <= 1.0 for value in target._data)):
        raise ValueError("binary cross-entropy targets must be between 0 and 1")


def _validate_from_logits(from_logits: bool) -> None:
    if not isinstance(from_logits, bool):
        raise TypeError("from_logits must be a bool")


def _probability_loss(probability: float, target: float) -> float:
    if probability == 0.0:
        return 0.0 if target == 0.0 else math.inf
    if probability == 1.0:
        return 0.0 if target == 1.0 else math.inf
    return -target * math.log(probability) - (1.0 - target) * math.log1p(-probability)


def _probability_gradient(probability: float, target: float) -> float:
    if probability == 0.0:
        return 1.0 if target == 0.0 else -math.inf
    if probability == 1.0:
        return -1.0 if target == 1.0 else math.inf
    return (probability - target) / (probability * (1.0 - probability))


def _target_gradient(probability: float) -> float:
    if probability == 0.0:
        return math.inf
    if probability == 1.0:
        return -math.inf
    return math.log1p(-probability) - math.log(probability)


class BinaryCrossEntropy(Operation):
    """Binary cross-entropy for probabilities or raw logits."""

    __slots__ = ("from_logits", "reduction")
    name = "binary_cross_entropy"

    def __init__(
        self, *, from_logits: bool = False, reduction: Reduction = "mean"
    ) -> None:
        object.__setattr__(self, "from_logits", from_logits)
        object.__setattr__(self, "reduction", reduction)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        from_logits = self.from_logits
        reduction = self.reduction
        _validate_reduction(reduction)
        _validate_from_logits(from_logits)
        prediction, target = broadcast_tensors(prediction, target)
        _validate_targets(target)
        dtype = result_dtype(prediction.dtype, target, division=True)
        output_shape = prediction.shape if reduction == "none" else (1,)
        storage = execute_binary_cross_entropy(
            prediction,
            target,
            from_logits=from_logits,
            reduction=reduction,
            dtype=dtype,
            output_shape=output_shape,
        )
        return Tensor._from_owned_storage(storage, dtype=dtype, shape=output_shape)

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Optional[Tensor]]:
        prediction, target = inputs
        need_prediction, need_target = needs_input_grad
        from_logits = self.from_logits
        _validate_from_logits(from_logits)
        reduction = self.reduction
        if not isinstance(reduction, str):
            raise TypeError("reduction must be a string")
        expanded_prediction, expanded_target = broadcast_tensors(prediction, target)
        accelerated = execute_binary_cross_entropy_gradient(
            grad,
            expanded_prediction,
            expanded_target,
            from_logits=from_logits,
            reduction=reduction,
            needs_input_grad=needs_input_grad,
        )
        prediction_storage, target_storage = accelerated
        expanded_shape = expanded_prediction.shape
        return [
            (
                sum_to_shape(
                    Tensor._from_owned_storage(
                        prediction_storage, dtype=grad.dtype, shape=expanded_shape
                    ),
                    prediction.shape,
                )
                if prediction_storage is not None
                else None
            ),
            (
                sum_to_shape(
                    Tensor._from_owned_storage(
                        target_storage, dtype=grad.dtype, shape=expanded_shape
                    ),
                    target.shape,
                )
                if target_storage is not None
                else None
            ),
        ]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build a differentiable binary cross-entropy VJP."""
        from tensors.operations.elementary.log import log
        from tensors.operations.activations.sigmoid import sigmoid

        prediction, target = inputs
        need_prediction, need_target = needs_input_grad
        from_logits = self.from_logits
        _validate_from_logits(from_logits)
        reduction = self.reduction
        shape = prediction.shape.broadcast_with(target.shape)
        size = shape.size
        upstream = grad / size if reduction == "mean" and size else grad
        if from_logits:
            positive_mask = Tensor(
                [
                    1.0 if float(value) >= 0.0 else 0.0
                    for value in prediction.data._data
                ],
                dtype=prediction.dtype,
                shape=prediction.shape,
            )
            negative_mask = Tensor(
                [1.0 - value for value in positive_mask._data],
                dtype=prediction.dtype,
                shape=prediction.shape,
            )
            prediction_derivative = positive_mask * (
                1.0 - target - sigmoid(-prediction)
            ) + negative_mask * (sigmoid(prediction) - target)
            target_derivative = -prediction + target * 0.0
        else:
            expanded_prediction, expanded_target = broadcast_tensors(
                prediction.data, target.data
            )
            boundary_values = list(
                zip(expanded_prediction._data, expanded_target._data)
            )
            if need_prediction:
                if any(
                    (
                        value == 0.0
                        and target_value != 0.0
                        or (value == 1.0 and target_value != 1.0)
                        for value, target_value in boundary_values
                    )
                ):
                    raise ValueError(
                        "Higher-order binary cross-entropy gradients require a finite first derivative"
                    )
                zero_mask = Tensor(
                    [
                        1.0 if value == 0.0 and target_value == 0.0 else 0.0
                        for value, target_value in boundary_values
                    ],
                    dtype=prediction.dtype,
                    shape=expanded_prediction.shape,
                )
                one_mask = Tensor(
                    [
                        1.0 if value == 1.0 and target_value == 1.0 else 0.0
                        for value, target_value in boundary_values
                    ],
                    dtype=prediction.dtype,
                    shape=expanded_prediction.shape,
                )
                prediction_derivative = -target / (prediction + zero_mask) + (
                    1.0 - target
                ) / (1.0 - prediction + one_mask)
            else:
                prediction_derivative = None
            if need_target:
                if any((value in {0.0, 1.0} for value, _ in boundary_values)):
                    raise ValueError(
                        "Higher-order gradients with respect to binary cross-entropy targets require probabilities strictly between 0 and 1"
                    )
                target_derivative = (
                    log(1.0 - prediction) - log(prediction) + target * 0.0
                )
            else:
                target_derivative = None
        return [
            (
                sum_to_shape_graph(upstream * prediction_derivative, prediction.shape)
                if need_prediction
                else None
            ),
            (
                sum_to_shape_graph(upstream * target_derivative, target.shape)
                if need_target
                else None
            ),
        ]


@overload
def binary_cross_entropy(
    prediction: VariableNode,
    target: TensorLike | VariableNode,
    *,
    from_logits: bool = False,
    reduction: Reduction = "mean",
) -> VariableNode: ...


@overload
def binary_cross_entropy(
    prediction: TensorLike,
    target: VariableNode,
    *,
    from_logits: bool = False,
    reduction: Reduction = "mean",
) -> VariableNode: ...


@overload
def binary_cross_entropy(
    prediction: Variable,
    target: TensorLike,
    *,
    from_logits: bool = False,
    reduction: Reduction = "mean",
) -> Variable: ...


@overload
def binary_cross_entropy(
    prediction: TensorLike,
    target: Variable,
    *,
    from_logits: bool = False,
    reduction: Reduction = "mean",
) -> Variable: ...


@overload
def binary_cross_entropy(
    prediction: TensorData,
    target: TensorData,
    *,
    from_logits: bool = False,
    reduction: Reduction = "mean",
) -> Tensor: ...


def binary_cross_entropy(
    prediction: TensorLike | VariableNode,
    target: TensorLike | VariableNode,
    *,
    from_logits: bool = False,
    reduction: Reduction = "mean",
) -> TensorResult | VariableNode:
    """Compute binary cross-entropy with optional stable logits input.

    A vertex in either position is answered first, because it names a
    value that does not exist and the coercions below read one. The
    prediction and the target are recorded as the first and second
    operands, and the operation keeps the ``from_logits`` and
    ``reduction`` configuration every application uses.
    """
    from tensors.graph.expression import apply_operation, as_graph_operand
    from tensors.graph.node import VariableNode
    from tensors.variable import Variable

    if isinstance(prediction, VariableNode) or isinstance(target, VariableNode):
        return apply_operation(
            BinaryCrossEntropy(from_logits=from_logits, reduction=reduction),
            (as_graph_operand(prediction), as_graph_operand(target)),
        )
    prediction_is_variable = isinstance(prediction, Variable)
    target_is_variable = isinstance(target, Variable)
    prediction_tensor = (
        prediction.data if prediction_is_variable else as_tensor_operand(prediction)
    )
    target_tensor = target.data if target_is_variable else as_tensor_operand(target)
    if prediction_is_variable or target_is_variable:
        prediction_variable = (
            prediction
            if isinstance(prediction, Variable)
            else Variable(prediction_tensor, requires_grad=False)
        )
        target_variable = (
            target
            if isinstance(target, Variable)
            else Variable(target_tensor, requires_grad=False)
        )
        operation = BinaryCrossEntropy(from_logits=from_logits, reduction=reduction)
        return Variable._apply_operation(
            operation, (prediction_variable, target_variable)
        )
    operation = BinaryCrossEntropy(from_logits=from_logits, reduction=reduction)
    return operation.forward(prediction_tensor, target_tensor)


__all__ = ["BinaryCrossEntropy", "binary_cross_entropy"]
