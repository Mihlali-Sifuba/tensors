"""Benchmarks for convolution forward and reverse execution."""

from __future__ import annotations

import math

import tensors as ts

from .runner import BenchmarkCase


def _ramp(shape: tuple[int, ...], scale: float = 1.0) -> ts.Tensor:
    size = ts.Shape.from_iterable(shape).size
    return ts.Tensor(
        [scale * (((index * 37) % 19) - 9) / 7.0 for index in range(size)],
        shape=shape,
    )


def cases() -> list[BenchmarkCase]:
    """Return representative 1D, 2D, 3D, and backward convolution cases."""
    signal = _ramp((4, 8, 128), 0.5)
    signal_kernel = _ramp((12, 8, 5), 0.25)

    def conv1d() -> ts.Tensor:
        return ts.conv1d(signal, signal_kernel, padding=2)

    image = _ramp((2, 4, 16, 16), 0.5)
    image_kernel = _ramp((6, 4, 3, 3), 0.25)

    def conv2d() -> ts.Tensor:
        return ts.conv2d(image, image_kernel, padding=1)

    volume = _ramp((1, 2, 6, 6, 6), 0.5)
    volume_kernel = _ramp((3, 2, 3, 3, 3), 0.25)

    def conv3d() -> ts.Tensor:
        return ts.conv3d(volume, volume_kernel, padding=1)

    backward_input = ts.Variable(image)
    backward_kernel = ts.Variable(image_kernel)
    backward_output = ts.conv2d(
        backward_input,
        backward_kernel,
        padding=1,
    )

    def conv2d_backward() -> None:
        ts.backward(backward_output)

    def reset_backward() -> None:
        backward_input.grad = None
        backward_kernel.grad = None

    def validate_forward(run, expected_shape: tuple[int, ...]) -> None:
        result = run()
        assert result.shape == expected_shape
        assert all(math.isfinite(float(value)) for value in result._data)

    def validate_backward() -> None:
        reset_backward()
        conv2d_backward()
        assert backward_input.grad is not None
        assert backward_kernel.grad is not None
        assert backward_input.grad.shape == backward_input.shape
        assert backward_kernel.grad.shape == backward_kernel.shape

    return [
        BenchmarkCase(
            name="convolution.conv1d/4x8x128-k5",
            run=conv1d,
            validate=lambda: validate_forward(conv1d, (4, 12, 128)),
            work_items=4 * 12 * 128 * 8 * 5,
            description="batched 1D convolution with same-size padding",
        ),
        BenchmarkCase(
            name="convolution.conv2d/2x4x16x16-k3",
            run=conv2d,
            validate=lambda: validate_forward(conv2d, (2, 6, 16, 16)),
            work_items=2 * 6 * 16 * 16 * 4 * 3 * 3,
            description="batched 2D convolution with same-size padding",
        ),
        BenchmarkCase(
            name="convolution.conv3d/1x2x6x6x6-k3",
            run=conv3d,
            validate=lambda: validate_forward(conv3d, (1, 3, 6, 6, 6)),
            work_items=3 * 6 * 6 * 6 * 2 * 3 * 3 * 3,
            description="batched 3D convolution with same-size padding",
        ),
        BenchmarkCase(
            name="convolution.backward_conv2d/2x4x16x16-k3",
            run=conv2d_backward,
            validate=validate_backward,
            work_items=2 * 6 * 16 * 16 * 4 * 3 * 3,
            layer="autograd",
            description="input and kernel VJPs for a batched 2D convolution",
            reset=reset_backward,
        ),
    ]