"""Convolution VJPs follow their specified term sequences, on every backend.

docs/summation-semantics.md section 11 fixes each gradient element as the
pairwise sum of a fixed sequence of products, with an exact zero term for a
combination that contributes nothing:

- input gradient ``(b, c, x)``: kernel offsets in row-major order and, within
  an offset, the group's output channels;
- kernel gradient ``(o, c, u)``: batch and, within it, output positions in
  row-major order;
- bias gradient ``o``: the upstream over batch and output positions.

The oracle below restates those sequences in plain Python from that text and
reduces them with the independent pairwise oracle. Every backend is compared
against it bit for bit, never against another backend.
"""

import ast
import itertools
import math
import random
import struct
import unittest
from pathlib import Path

import tensors as ts
from tensors.backend import dispatch as backend_dispatch
from tests._pairwise_oracle import pairwise_dot, pairwise_sum, round32


def _strides(shape):
    return [math.prod(shape[axis + 1 :]) for axis in range(len(shape))]


def _index(coordinate, shape):
    return sum(c * s for c, s in zip(coordinate, _strides(shape)))


def oracle(grad, inputs, kernel, geometry, dtype_name):
    """Input, kernel and bias VJPs by their specified term sequences."""
    batch, in_channels, spatial, out_channels, kernel_spatial, output_spatial = (
        geometry["batch"],
        geometry["in_channels"],
        geometry["spatial"],
        geometry["out_channels"],
        geometry["kernel_spatial"],
        geometry["output_spatial"],
    )
    stride, padding, dilation, groups = (
        geometry["stride"],
        geometry["padding"],
        geometry["dilation"],
        geometry["groups"],
    )
    rank = len(spatial)
    group_channels = in_channels // groups
    group_outputs = out_channels // groups
    input_shape = (batch, in_channels) + spatial
    kernel_shape = (out_channels, group_channels) + kernel_spatial
    grad_shape = (batch, out_channels) + output_spatial
    offsets = list(itertools.product(*map(range, kernel_spatial)))
    positions = list(itertools.product(*map(range, output_spatial)))

    input_gradient = []
    for b, c, *x in itertools.product(range(batch), range(in_channels), *map(range, spatial)):
        group, local = divmod(c, group_channels)
        left, right = [], []
        for u in offsets:
            numerators = [x[d] + padding[d] - u[d] * dilation[d] for d in range(rank)]
            valid = all(n % stride[d] == 0 for d, n in enumerate(numerators))
            p = [n // stride[d] for d, n in enumerate(numerators)]
            valid = valid and all(0 <= p[d] < output_spatial[d] for d in range(rank))
            for j in range(group_outputs):
                o = group * group_outputs + j
                if valid:
                    left.append(grad[_index((b, o, *p), grad_shape)])
                    right.append(kernel[_index((o, local, *u), kernel_shape)])
                else:
                    left.append(0.0)
                    right.append(0.0)
        input_gradient.append(pairwise_dot(left, right, dtype_name))

    kernel_gradient = []
    for o, local, *u in itertools.product(
        range(out_channels), range(group_channels), *map(range, kernel_spatial)
    ):
        c = (o // group_outputs) * group_channels + local
        left, right = [], []
        for b in range(batch):
            for p in positions:
                source = [p[d] * stride[d] - padding[d] + u[d] * dilation[d] for d in range(rank)]
                if all(0 <= source[d] < spatial[d] for d in range(rank)):
                    left.append(grad[_index((b, o, *p), grad_shape)])
                    right.append(inputs[_index((b, c, *source), input_shape)])
                else:
                    left.append(0.0)
                    right.append(0.0)
        kernel_gradient.append(pairwise_dot(left, right, dtype_name))

    bias_gradient = [
        pairwise_sum(
            [grad[_index((b, o, *p), grad_shape)] for b in range(batch) for p in positions],
            dtype_name,
        )
        for o in range(out_channels)
    ]
    return input_gradient, kernel_gradient, bias_gradient


def _bits(value, dtype):
    if math.isnan(value):
        return b"nan"
    return struct.pack("<f" if dtype is ts.float32 else "<d", value)


GEOMETRIES = (
    # rank, spatial, kernel, groups, in, out, stride, padding, dilation, batched
    (1, (7,), (3,), 1, 1, 1, 1, 0, 1, True),
    (1, (8,), (2,), 1, 3, 2, 2, 1, 1, False),
    (1, (9,), (3,), 2, 4, 6, 1, 2, 2, True),
    (2, (5, 6), (3, 2), 1, 2, 3, 1, 1, 1, True),
    (2, (6, 5), (2, 2), 3, 3, 3, 2, 1, 1, False),
    (2, (7, 7), (3, 3), 2, 4, 2, 3, 1, 2, True),
    (3, (3, 4, 3), (2, 2, 2), 1, 2, 2, 1, 1, 1, True),
    (3, (4, 3, 4), (2, 1, 2), 2, 2, 4, 2, 0, 1, False),
)


class ConvolutionGradientTermTests(unittest.TestCase):
    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)

    def _values(self, rng, count, dtype, special):
        values = []
        for _ in range(count):
            if special and rng.random() < 0.2:
                values.append(rng.choice((0.0, -0.0, 1e-40, -2e-39, 1.5e-45)))
            else:
                values.append(rng.uniform(-2.0, 2.0) * 10.0 ** rng.randint(-2, 2))
        return [round32(v) for v in values] if dtype is ts.float32 else values

    def test_every_backend_matches_the_specified_sequences(self):
        rng = random.Random(4)
        for geometry_values in GEOMETRIES:
            rank, spatial, kernel_spatial, groups, in_ch, out_ch, s, p, d, batched = geometry_values
            output_spatial = tuple(
                (n + 2 * p - d * (k - 1) - 1) // s + 1 for n, k in zip(spatial, kernel_spatial)
            )
            batch = 2 if batched else 1
            geometry = dict(
                batch=batch,
                in_channels=in_ch,
                spatial=spatial,
                out_channels=out_ch,
                kernel_spatial=kernel_spatial,
                output_spatial=output_spatial,
                stride=(s,) * rank,
                padding=(p,) * rank,
                dilation=(d,) * rank,
                groups=groups,
            )
            for dtype in (ts.float32, ts.float64):
                special = dtype is ts.float32
                inputs = self._values(rng, batch * in_ch * math.prod(spatial), dtype, special)
                kernel = self._values(rng, out_ch * (in_ch // groups) * math.prod(kernel_spatial), dtype, special)
                grad = self._values(rng, batch * out_ch * math.prod(output_spatial), dtype, special)
                expected = oracle(grad, inputs, kernel, geometry, dtype.name)
                input_shape = ((batch,) if batched else ()) + (in_ch,) + spatial
                kernel_shape = (out_ch, in_ch // groups) + kernel_spatial
                grad_shape = ((batch,) if batched else ()) + (out_ch,) + output_spatial
                for backend in ts.available_backends():
                    for needs in (
                        (True, True, True),
                        (True, False, False),
                        (False, True, False),
                        (False, False, True),
                    ):
                        with self.subTest(
                            geometry=geometry_values, dtype=dtype.name, backend=backend, needs=needs
                        ), ts.use_backend(backend):
                            storages = backend_dispatch.execute_convolution_gradient(
                                ts.Tensor(grad, dtype=dtype, shape=grad_shape),
                                ts.Tensor(inputs, dtype=dtype, shape=input_shape),
                                ts.Tensor(kernel, dtype=dtype, shape=kernel_shape),
                                stride=(s,) * rank,
                                padding=(p,) * rank,
                                dilation=(d,) * rank,
                                groups=groups,
                                include_bias=True,
                                needs_input_grad=needs,
                            )
                            for wanted, storage, values in zip(needs, storages, expected):
                                if not wanted:
                                    self.assertIsNone(storage)
                                    continue
                                self.assertEqual(storage.kind, backend)
                                produced = ts.Tensor.from_backend_storage(
                                    storage, dtype=dtype, shape=(storage.size,)
                                ).tolist()
                                self.assertEqual(
                                    [_bits(v, dtype) for v in produced],
                                    [_bits(v, dtype) for v in values],
                                )

    def test_signed_zero_and_float32_subnormal_gradients(self):
        """A zero result is +0.0; a subnormal product survives the reduction."""
        tiny = struct.unpack("<f", struct.pack("<I", 7))[0]
        for backend in ts.available_backends():
            with self.subTest(backend=backend), ts.use_backend(backend):
                inputs = ts.Tensor([-0.0, tiny, 1.0], dtype=ts.float32, shape=(1, 1, 3))
                kernel = ts.Tensor([1.0], dtype=ts.float32, shape=(1, 1, 1))
                grad = ts.Tensor([-0.0, 1.0, 0.0], dtype=ts.float32, shape=(1, 1, 3))
                _, kernel_grad, bias_grad = backend_dispatch.execute_convolution_gradient(
                    grad,
                    inputs,
                    kernel,
                    stride=(1,),
                    padding=(0,),
                    dilation=(1,),
                    groups=1,
                    include_bias=True,
                    needs_input_grad=(False, True, True),
                )
                produced = ts.Tensor.from_backend_storage(
                    kernel_grad, dtype=ts.float32, shape=(1,)
                ).item()
                self.assertEqual(produced, tiny)
                bias = ts.Tensor.from_backend_storage(
                    bias_grad, dtype=ts.float32, shape=(1,)
                ).item()
                self.assertEqual(bias, 1.0)
                zero = ts.Tensor.from_backend_storage(
                    backend_dispatch.execute_convolution_gradient(
                        ts.Tensor([-0.0], dtype=ts.float32, shape=(1, 1, 1)),
                        ts.Tensor([1.0], dtype=ts.float32, shape=(1, 1, 1)),
                        kernel,
                        stride=(1,),
                        padding=(0,),
                        dilation=(1,),
                        groups=1,
                        include_bias=False,
                        needs_input_grad=(True, False),
                    )[0],
                    dtype=ts.float32,
                    shape=(1,),
                ).item()
                self.assertEqual(math.copysign(1.0, zero), 1.0)


class VectorisedTermConstructionTests(unittest.TestCase):
    """The array backends build terms per tile, not per gradient element."""

    def test_the_vjp_helpers_loop_only_over_tiles(self):
        for backend in ("numpy", "cuda"):
            path = (
                Path(ts.__file__).parent
                / "backend"
                / backend
                / "kernels"
                / "convolution"
                / "convolution_gradient.py"
            )
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            helpers = {
                node.name: node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name in {"_input_vjp", "_kernel_vjp"}
            }
            self.assertEqual(set(helpers), {"_input_vjp", "_kernel_vjp"}, backend)
            for name, node in helpers.items():
                with self.subTest(backend=backend, helper=name):
                    loops = [n for n in ast.walk(node) if isinstance(n, (ast.For, ast.While))]
                    self.assertEqual(len(loops), 1)
                    target = loops[0].target
                    self.assertIsInstance(target, ast.Name)
                    self.assertEqual(target.id, "start")
            gradient = next(
                node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "convolution_gradient"
            )
            self.assertNotIn("islice", ast.unparse(gradient))


if __name__ == "__main__":
    unittest.main()
