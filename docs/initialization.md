# Parameter initialization and random generation

MS-Tensors exposes parameter initializers through `ts.init` and random tensor
generation through `ts.random`. Both namespaces return ordinary `Tensor`
objects; they do not introduce layer, module, or stateful initializer classes.

```python
import tensors as ts

ts.random.seed(42)
weight = ts.Variable(
    ts.init.he_normal((128, 64)),
    requires_grad=True,
)
```

## Fan definitions

For a matrix shape `(fan_in, fan_out)`, the first dimension is the number of
input units and the second is the number of output units. This matches the
usual `input @ weight` expression.

For tensors with more than two dimensions, MS-Tensors uses a channel-last
kernel convention:

```text
(receptive_field..., in_channels, out_channels)
```

If `R` is the product of the receptive-field dimensions, then:

```text
fan_in  = R * in_channels
fan_out = R * out_channels
fan_avg = (fan_in + fan_out) / 2
```

Rank-zero and rank-one shapes are ambiguous and are rejected by fan-based
initializers. A zero dimension would produce a zero fan, so it is rejected
rather than silently clamped.

## Mathematical definitions

`variance_scaling(shape, scale, mode, distribution)` targets variance
`scale / fan`, where `fan` is selected by `mode`.

- `uniform` samples from `[-sqrt(3 * scale / fan), +sqrt(3 * scale / fan))`.
- `normal` samples from a zero-mean normal with standard deviation
  `sqrt(scale / fan)`.
- `truncated_normal` truncates at two source standard deviations and corrects
  the source deviation so the retained samples still have variance
  `scale / fan`.

The named initializers are specializations:

| Initializer | Target variance | Typical use |
| --- | --- | --- |
| Xavier uniform/normal | `2 / (fan_in + fan_out)` | symmetric or tanh-like activations |
| He uniform/normal | `2 / fan_in` | ReLU-family activations |
| LeCun uniform/normal | `1 / fan_in` | SELU or scaled exponential activations |

`truncated_normal` samples from a source `N(mean, stddev**2)` subject to
inclusive absolute bounds. Its default bounds are `mean ± 2 * stddev`; the
requested deviation describes the source distribution before truncation.

`orthogonal` views a shape as `(shape[0], product(shape[1:]))`, constructs an
orthonormal basis for the smaller matrix dimension, and multiplies it by
`gain`. Shapes must have at least two positive dimensions.

## RNG state and reproducibility

`ts.random.seed(value)` resets independent MS-Tensors-owned streams for the
Python, NumPy, and CUDA backends. It does not call the module-global seed
functions from Python, NumPy, or CuPy and therefore does not disturb
application-owned random state.

Calling a generator advances the active backend's stream. Repeating a seed and
the same sequence of calls on the same backend reproduces the same tensors.
Streams are independent across backends; exact sample values are not promised
to match between different provider algorithms.

```python
ts.random.seed(42)
first = ts.random.uniform((4, 4))
second = ts.random.normal((4, 4))

ts.random.seed(42)
assert ts.random.uniform((4, 4)).tolist() == first.tolist()
assert ts.random.normal((4, 4)).tolist() == second.tolist()
```

The public random API contains `seed`, `uniform`, `normal`, and `randint`.
`randint` uses a half-open interval and integer dtype; the floating generators
require a floating dtype.

## Backend behavior

The active numerical backend also owns random generation:

```text
initializer
    -> MS-Tensors RNG
        -> Python random.Random
        -> NumPy Generator
        -> CuPy RandomState
```

Python samples are stored in `array.array`, NumPy samples in `numpy.ndarray`,
and CUDA samples in device-resident `cupy.ndarray`. Truncated-normal rejection
sampling and orthogonal QR decomposition also run through the active provider.
CUDA values therefore remain on the device unless a host-facing operation such
as `tolist()` is requested.
