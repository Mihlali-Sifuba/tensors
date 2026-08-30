# Tensor Memory Model

## Status

Current metadata foundation. MS-Tensors represents tensor layout explicitly,
while public construction, reshape, transpose, slicing, and broadcasting retain
their existing owning/materializing behavior. Shared-storage views are not part
of this change.

## Metadata

Every Tensor has four fundamental layout components:

```text
Tensor
├── shape: Shape
├── strides: Strides
├── offset: int
└── storage: Storage
```

- `Shape` contains the logical extent of every axis.
- `Strides` contains the physical storage movement produced by advancing one
  logical position along each axis.
- `offset` is the physical storage position for logical coordinate
  `(0, ..., 0)`.
- `Storage` owns a flat backend-native Python, NumPy, or CUDA buffer.

For logical coordinate `(i_0, ..., i_(n-1))`, the physical storage position
is

$$
k = o + \sum_{d=0}^{n-1} i_d s_d,
$$

where `o` is the offset and `s_d` is the stride for axis `d`.
Coordinate validation belongs to the Shape and indexing semantics. Strides only
represent movement and may therefore be positive, zero, or negative.

## Shape

`Shape` is an immutable tuple-like value object:

```python
import tensors as ts

shape = ts.Shape(2, 3, 4)

shape.rank       # 3
shape.size       # 24
shape[0]         # 2
tuple(shape)     # (2, 3, 4)
```

Dimensions are non-negative integers. A scalar uses `Shape()`, has rank zero,
and contains one logical value. Zero-sized dimensions are valid; for example,
`Shape(2, 0, 4)` has size zero.

`Shape` is exposed at the root because it is the precise public type returned
by `tensor.shape`. It remains interoperable with tuples, so existing equality,
iteration, indexing, slicing, and concatenation behavior is preserved.

## Strides and offset

`Strides` is also immutable and tuple-like:

```python
shape = ts.Shape(2, 3, 4)
strides = ts.Strides.contiguous(shape)

strides           # (12, 4, 1)
```

Canonical layout is row-major. Scalars have `Strides()`. Empty shapes retain
the normal trailing-product rule, so `Shape(2, 0, 3)` has contiguous strides
`Strides(0, 3, 1)`.

Zero and negative values are valid stride metadata:

```text
shape   = Shape(4, 3)
strides = Strides(0, 1)
```

can describe a future zero-stride broadcast view, while

```text
shape   = Shape(3)
strides = Strides(-1)
offset  = 2
```

can describe reversed traversal. This metadata is representable now; public
zero-copy broadcast and reversed views are not yet exposed.

`Strides` is available from the root because `tensor.strides` has that
precise public type and advanced users need to reason about layout. Ordinary
Tensor construction calculates it automatically.

## Contiguity

`tensor.is_contiguous` compares the tensor's actual strides with
`Strides.contiguous(tensor.shape)`. It does not infer contiguity from the
backend provider.

`tensor.contiguous()` returns an already-contiguous tensor unchanged. For a
non-contiguous internal layout, it gathers values in logical row-major order
into independent compact storage while preserving shape, dtype, and backend
residency.

NumPy and CUDA kernels currently consume compact provider arrays. Before that
boundary, Tensor metadata is applied and a non-compact layout is gathered in
the same backend. Reference kernels consume logical row-major values. Ordinary
public tensors already satisfy the compact-storage invariant, so current
backend behavior and transfer costs are unchanged.

## Ownership and current limits

This foundation does not introduce shared-storage aliasing. The internal
explicit-metadata constructor copies storage, and existing public layout
operations continue to produce independently owned tensors. Mutation
versioning, backend representation-cache invalidation, and stale-autograd
detection therefore retain their existing semantics.

The metadata model prepares the package for deliberate follow-up work on:

- zero-copy transpose;
- zero-copy slicing;
- zero-stride broadcasting;
- negative-stride reversal;
- shared-storage lifetime and ownership;
- mutation propagation and Tensor versioning across aliases;
- stale-autograd detection for views; and
- backend cache behavior for shared storage.

Those capabilities require a separate view/aliasing design and are not claimed
as implemented here.
