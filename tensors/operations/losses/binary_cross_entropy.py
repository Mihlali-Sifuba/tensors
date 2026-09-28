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
from tensors.operations.gradient_primitives import sum_to_shape
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
        from_logits = self.from_logits
        _validate_from_logits(from_logits)
        reduction = self.reduction
        if not isinstance(reduction, str):
            raise TypeError("reduction must be a string")
        from tensors.graph.expression import apply_operation, is_graph_operand

        if any(is_graph_operand(value) for value in (grad, prediction, target)):
            gradients: List[Optional[Tensor]] = []
            for needed, operand, select_prediction in (
                (needs_input_grad[0], prediction, True),
                (needs_input_grad[1], target, False),
            ):
                if not needed:
                    gradients.append(None)
                    continue
                contribution = apply_operation(
                    BinaryCrossEntropyVJP(
                        from_logits=from_logits,
                        reduction=reduction,
                        select_prediction=select_prediction,
                    ),
                    (grad, prediction, target),
                )
                gradients.append(sum_to_shape(contribution, operand.shape))
            return gradients

        expanded_prediction, expanded_target = broadcast_tensors(prediction, target)
        prediction_storage, target_storage = execute_binary_cross_entropy_gradient(
            grad,
            expanded_prediction,
            expanded_target,
            from_logits=from_logits,
            reduction=reduction,
            needs_input_grad=needs_input_grad,
        )
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


class BinaryCrossEntropyVJP(Operation):
    """One differentiable branch of the fused binary cross-entropy VJP."""

    __slots__ = ("from_logits", "reduction", "select_prediction")
    name = "binary_cross_entropy_vjp"

    def __init__(
        self,
        *,
        from_logits: bool,
        reduction: Reduction,
        select_prediction: bool,
    ) -> None:
        object.__setattr__(self, "from_logits", from_logits)
        object.__setattr__(self, "reduction", reduction)
        object.__setattr__(self, "select_prediction", select_prediction)

    def forward(self, grad: Tensor, prediction: Tensor, target: Tensor) -> Tensor:
        expanded_prediction, expanded_target = broadcast_tensors(prediction, target)
        _validate_targets(expanded_target)
        if (
            (not self.select_prediction)
            and (not self.from_logits)
            and any(not 0.0 < float(value) < 1.0 for value in expanded_prediction._data)
        ):
            raise ValueError(
                "Higher-order binary cross-entropy target derivatives require "
                "probabilities strictly between 0 and 1"
            )
        expected_shape = expanded_prediction.shape if self.reduction == "none" else (1,)
        if grad.shape != expected_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape "
                f"{expected_shape}"
            )
        storages = execute_binary_cross_entropy_gradient(
            grad,
            expanded_prediction,
            expanded_target,
            from_logits=self.from_logits,
            reduction=self.reduction,
            needs_input_grad=(
                self.select_prediction,
                not self.select_prediction,
            ),
        )
        storage = storages[0] if self.select_prediction else storages[1]
        if storage is None:
            raise RuntimeError("binary cross-entropy VJP omitted its requested branch")
        return Tensor._from_owned_storage(
            storage, dtype=grad.dtype, shape=expanded_prediction.shape
        )

    def backward(
        self, outer_grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Optional[Tensor]]:
        from tensors.creation import ones, zeros
        from tensors.operations.activations.sigmoid import sigmoid
        from tensors.utils.broadcasting import broadcast_to

        grad, prediction, target = inputs
        expanded_prediction, expanded_target = broadcast_tensors(prediction, target)
        shape = expanded_prediction.shape
        if self.reduction == "none":
            expanded_grad = grad
        else:
            expanded_grad = broadcast_to(grad, shape)
            if self.reduction == "mean" and expanded_prediction.size:
                expanded_grad = expanded_grad / expanded_prediction.size

        need_grad, need_prediction, need_target = needs_input_grad
        grad_partial = None
        if need_grad:
            unit = (
                ones(grad.shape, dtype=grad.dtype)
                if self.reduction == "none"
                else Tensor([1.0], dtype=grad.dtype)
            )
            local = self.forward(unit, prediction, target)
            grad_partial = sum_to_shape(outer_grad * local, grad.shape)

        prediction_partial = None
        target_partial = None
        if self.select_prediction:
            if need_prediction:
                if self.from_logits:
                    probability = sigmoid(expanded_prediction)
                    second = probability * (1.0 - probability)
                    prediction_partial = sum_to_shape(
                        outer_grad * expanded_grad * second,
                        prediction.shape,
                    )
                else:
                    values = []
                    for raw_probability, raw_target in zip(
                        expanded_prediction._data, expanded_target._data
                    ):
                        probability = float(raw_probability)
                        target_value = float(raw_target)
                        if probability == 0.0 and target_value == 0.0:
                            values.append(1.0)
                        elif probability == 1.0 and target_value == 1.0:
                            values.append(1.0)
                        else:
                            values.append(
                                target_value / (probability * probability)
                                + (1.0 - target_value)
                                / ((1.0 - probability) * (1.0 - probability))
                            )
                    second = Tensor(values, dtype=outer_grad.dtype, shape=shape)
                    prediction_partial = sum_to_shape(
                        outer_grad * expanded_grad * second,
                        prediction.shape,
                    )
            if need_target:
                if self.from_logits:
                    mixed = broadcast_to(Tensor([-1.0], dtype=outer_grad.dtype), shape)
                else:
                    values = []
                    for raw_probability in expanded_prediction._data:
                        probability = float(raw_probability)
                        values.append(
                            -math.inf
                            if probability in {0.0, 1.0}
                            else -1.0 / (probability * (1.0 - probability))
                        )
                    mixed = Tensor(values, dtype=outer_grad.dtype, shape=shape)
                target_partial = sum_to_shape(
                    outer_grad * expanded_grad * mixed,
                    target.shape,
                )
        else:
            if need_prediction:
                if self.from_logits:
                    mixed = broadcast_to(Tensor([-1.0], dtype=outer_grad.dtype), shape)
                else:
                    values = []
                    for raw_probability in expanded_prediction._data:
                        probability = float(raw_probability)
                        values.append(
                            -math.inf
                            if probability in {0.0, 1.0}
                            else -1.0 / (probability * (1.0 - probability))
                        )
                    mixed = Tensor(values, dtype=outer_grad.dtype, shape=shape)
                prediction_partial = sum_to_shape(
                    outer_grad * expanded_grad * mixed,
                    prediction.shape,
                )
            if need_target:
                target_partial = zeros(target.shape, dtype=outer_grad.dtype)
        return [grad_partial, prediction_partial, target_partial]


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
