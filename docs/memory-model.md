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

## Index spaces

The indexing model uses three precise terms:

- A **coordinate** is a logical n-dimensional position such as `(1, 2)`.
- A **logical linear index** is the canonical row-major position of that
  coordinate in logical Tensor order. It depends only on `Shape` and canonical
  contiguous strides.
- A **storage index** is the physical position in the underlying flat
  `Storage`. It depends on `Shape`, `Strides`, and `offset`.

For `Shape(2, 3)`, coordinate `(1, 2)` has logical linear index `5`. A
non-canonical layout may map that same coordinate to a different storage
index. The helpers in `utils/coordinates.py` convert between coordinates and
logical linear indices only; `utils/indexing.py` performs Tensor index
normalization and physical storage addressing.

## Shape

`Shape` is an immutable tuple-like value object:

```python
import tensors as ts

shape = ts.Shape(2, 3, 4)

shape.rank       # 3
shape.size       # 24
shape[0]         # 2
tuple(shape)     # (2, 3, 4)

sliced = shape[1:]
type(sliced) is ts.Shape  # True
sliced == (3, 4)          # True

broadcast = shape.broadcast_with((1, 3, 1))
type(broadcast) is ts.Shape  # True
broadcast == (2, 3, 4)       # True
```

Dimensions are non-negative integers. A scalar uses `Shape()`, has rank zero,
and contains one logical value. Zero-sized dimensions are valid; for example,
`Shape(2, 0, 4)` has size zero.

`Shape` is exposed at the root because it is the precise public type returned
by `tensor.shape`. It remains interoperable with tuples: equality and iteration
retain normal tuple behavior, integer indexing returns an `int`, and slicing
of its logical dimensions preserves the `Shape` type. Tuple concatenation still
follows tuple semantics and returns a plain tuple. This tuple-like
`Shape(2, 3, 4)[1:]` operation is distinct from Tensor/Python indexing such as
`tensor[1:, :, 2]`, which belongs to the slicing utilities.

Pure broadcast compatibility belongs to `Shape`.
`shape.broadcast_with(other)` is the authoritative `Shape × Shape → Shape`
operation. It accepts another `Shape` or an iterable of dimensions, returns a
new `Shape`, and does not mutate either input.

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

`tensor.is_contiguous` determines whether the logical elements occupy
consecutive storage positions in row-major order. Singleton dimensions are
ignored because their strides are never used to select a different element.
For example, `shape=(1, 3)` with `strides=(100, 1)` is contiguous even though
its strides are not canonical.

Every valid zero-sized layout is also contiguous regardless of its strides.
A tensor with `shape=(2, 0, 3)` has no logical elements, so there are no
addressed storage positions that could contain an observable gap. Contiguity
is derived from Tensor metadata rather than inferred from the backend
provider.

Contiguous layout and compact storage are distinct. Contiguity describes the
spacing between logical elements and does not require `offset == 0`. Compact
storage is the stricter internal condition that the layout is contiguous,
starts at offset zero, and exactly fills its owned storage.

For example, `shape=(2,)`, `strides=(1,)`, and `offset=1` over storage
`[10, 20, 30]` is contiguous but not compact. Its logical values are the
adjacent storage values `[20, 30]`.

`tensor.contiguous()` means "ensure a contiguous logical layout." It returns
an already-contiguous tensor unchanged, including one with a non-zero offset;
it does not guarantee compact storage or offset zero. For a non-contiguous
internal layout, it gathers values in logical row-major order into independent
compact storage while preserving shape, dtype, and backend residency.

NumPy and CUDA kernels currently consume compact provider arrays. Before that
boundary, Tensor metadata is applied and a non-compact layout is gathered in
the same backend. Reference kernels consume logical row-major values. Ordinary
public tensors already satisfy the compact-storage invariant, so current
backend behavior and transfer costs are unchanged.

## Responsibility boundaries

- `Shape` owns logical dimensions, including rank, size, tuple-like slicing of
  its dimension values, and pure broadcast-shape inference.
- `Strides` owns physical movement metadata and canonical contiguous-stride
  construction.
- `offset` identifies the logical tensor's origin within storage.
- Coordinate utilities convert between logical coordinates and canonical
  row-major logical linear indices using only `Shape`.
- Indexing utilities normalize Tensor indices and combine coordinates,
  `Shape`, `Strides`, and `offset` to produce physical storage indices.
- Slicing utilities own Tensor/Python indexing and slicing semantics, such as
  `tensor[1:, :, 2]`, and return `Shape` metadata for the resulting logical
  dimensions.
- Broadcasting utilities own tensor/value expansion and delegate pure
  shape compatibility to `Shape.broadcast_with`.

Logical coordinate conversion intentionally remains outside `Shape`, while
physical addressing remains outside `Shape` and `Strides` because it combines
both metadata objects with `offset`. Tensor slicing and broadcasting remain
materializing operations; this responsibility cleanup does not introduce
views or aliasing.

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
