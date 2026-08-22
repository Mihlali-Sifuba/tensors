"""Representative benchmarks for backend-dispatched kernel families."""

from __future__ import annotations

import math
from collections.abc import Callable

import tensors as ts

from .runner import BenchmarkCase


def _optimizer_case(
    name: str,
    optimizer_type: Callable[..., ts.optim.Optimizer],
) -> BenchmarkCase:
    size = 10_000
    parameter = ts.Variable(ts.full((size,), 1.0))
    parameter.grad = ts.full((size,), 0.5)
    optimizer = optimizer_type([parameter], learning_rate=1e-3)

    def run() -> None:
        optimizer.step()

    def validate() -> None:
        run()
        assert math.isfinite(float(parameter.data[0]))
        assert parameter.data[-1] == parameter.data[0]

    return BenchmarkCase(
        name=f"optimizer.{name}_step/{size}",
        run=run,
        validate=validate,
        work_items=size,
        description=f"one fused {name} update over {size} parameters",
    )


def cases() -> list[BenchmarkCase]:
    """Return one stable public-API case per major dispatch family."""
    unary_size = 10_000
    unary_input = ts.linspace(-2.0, 2.0, unary_size)

    def unary_exp() -> ts.Tensor:
        return ts.exp(unary_input)

    def validate_unary_exp() -> None:
        result = unary_exp()
        assert result.shape == unary_input.shape
        assert math.isclose(float(result[0]), math.exp(-2.0))

    normalization_shape = (128, 64)
    logits = ts.Tensor(
        [float(index % 17) / 8.0 for index in range(128 * 64)],
        shape=normalization_shape,
    )

    def normalize() -> ts.Tensor:
        return ts.softmax(logits, axis=1)

    def validate_normalize() -> None:
        result = normalize()
        assert result.shape == normalization_shape
        assert math.isclose(sum(result._data[:64]), 1.0)

    targets = ts.Tensor([index % 64 for index in range(128)], dtype=ts.int64)

    def classify() -> ts.Tensor:
        return ts.cross_entropy(logits, targets)

    def validate_classify() -> None:
        result = classify()
        assert result.shape == (1,)
        assert math.isfinite(float(result.item()))

    reduction_input = ts.Tensor(
        [float(index % 31) for index in range(10_000)],
    )

    def reduce_std() -> ts.Tensor:
        return ts.std(reduction_input)

    def validate_reduce_std() -> None:
        result = reduce_std()
        assert result.shape == (1,)
        assert float(result.item()) > 0.0

    selection_size = 10_000
    condition = ts.Tensor(
        [index % 2 for index in range(selection_size)],
        dtype=ts.uint8,
    )
    selection_left = ts.full((selection_size,), 1.0)
    selection_right = ts.full((selection_size,), 2.0)

    def select() -> ts.Tensor:
        return ts.where(condition, selection_left, selection_right)

    def validate_select() -> None:
        result = select()
        assert result.shape == (selection_size,)
        assert result[0] == 2.0
        assert result[1] == 1.0

    matrix = ts.full((128, 128), 1.0)

    def transpose() -> ts.Tensor:
        return ts.transpose(matrix)

    def validate_transpose() -> None:
        result = transpose()
        assert result.shape == matrix.shape
        assert result[0, 0] == 1.0

    concat_left = ts.full((64, 64), 1.0)
    concat_right = ts.full((64, 64), 2.0)

    def concatenate() -> ts.Tensor:
        return ts.concat([concat_left, concat_right], axis=1)

    def validate_concatenate() -> None:
        result = concatenate()
        assert result.shape == (64, 128)
        assert result[0, -1] == 2.0

    def create_full() -> ts.Tensor:
        return ts.full((10_000,), 1.25)

    def validate_create_full() -> None:
        result = create_full()
        assert result.shape == (10_000,)
        assert result[-1] == 1.25

    benchmarks = [
        BenchmarkCase(
            name="unary.exp/10000",
            run=unary_exp,
            validate=validate_unary_exp,
            work_items=unary_size,
            description="elementwise exponential over 10000 values",
        ),
        BenchmarkCase(
            name="normalization.softmax/128x64",
            run=normalize,
            validate=validate_normalize,
            work_items=128 * 64,
            description="row-wise softmax over a 128 by 64 matrix",
        ),
        BenchmarkCase(
            name="loss.cross_entropy/128x64",
            run=classify,
            validate=validate_classify,
            work_items=128 * 64,
            description="dense multiclass loss from class-index targets",
        ),
        BenchmarkCase(
            name="reduction.std/10000",
            run=reduce_std,
            validate=validate_reduce_std,
            work_items=10_000,
            description="population standard deviation over 10000 values",
        ),
        BenchmarkCase(
            name="selection.where/10000",
            run=select,
            validate=validate_select,
            work_items=selection_size,
            description="mask selection over 10000 values",
        ),
        BenchmarkCase(
            name="layout.transpose/128x128",
            run=transpose,
            validate=validate_transpose,
            work_items=matrix.size,
            description="materialized transpose of a square matrix",
        ),
        BenchmarkCase(
            name="layout.concat/64x128",
            run=concatenate,
            validate=validate_concatenate,
            work_items=64 * 128,
            description="concatenation of two matrices along columns",
        ),
        BenchmarkCase(
            name="creation.full/10000",
            run=create_full,
            validate=validate_create_full,
            work_items=10_000,
            description="constant tensor construction",
        ),
    ]
    benchmarks.extend((
        _optimizer_case("sgd", ts.optim.SGD),
        _optimizer_case("adam", ts.optim.Adam),
        _optimizer_case("rmsprop", ts.optim.RMSprop),
    ))
    return benchmarks
