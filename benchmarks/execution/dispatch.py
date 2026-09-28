"""Fixed framework cost, isolated from numerical work.

Everything here runs on inputs small enough that the arithmetic is free, so
what is left is the machinery: selecting a backend, resolving a kernel,
deciding a dtype, agreeing a shape, wrapping a result, and recording a graph
vertex. These are the numbers that explain why a small operation costs what
it does, and they are the floor no size curve drops below.

Some rungs are internal by necessity. There is no public way to ask for
"dtype resolution alone", so the internal function is called directly and
the case says so.
"""

from __future__ import annotations
import importlib
from typing import Any
import tensors as ts
from tensors.backend import execute_add, loading
from tensors.backend.policy import _array_work_is_large_enough, _shape_size
from tensors.backend.python.storage import PythonStorage
from tensors.dtype import result_dtype
from tensors.graph import Computation
from tensors.graph.node import OperationNode, VariableNode
from tensors.graph.edge import Edge
from tensors.graph.state import get_graph_state
from tensors.ops import Add, Mul
from tensors.shape import Shape
from tensors.strides import Strides
from tensors.utils.broadcasting import broadcast_to
from benchmarks.case import Case, Group, Unsupported
from benchmarks.inputs import ACCELERATED, tensor


def _selection_cases(backend: str) -> list[Case]:
    """Measure backend selection and kernel resolution."""
    cases: list[Case] = []
    common: dict[str, Any] = {
        "family": "framework/selection",
        "elements": 0,
        "tags": {"curve": "framework-selection"},
    }
    cases.append(
        Case(
            name="framework.get_backend",
            run=ts.get_backend,
            layer="dispatch",
            validate=ts.get_backend,
            description="read the backend active in this execution context",
            **common,
        )
    )
    cases.append(
        Case(
            name="framework.available_backends",
            run=ts.available_backends,
            layer="dispatch",
            validate=ts.available_backends,
            description="probe which backends are installed, which imports CuPy and counts devices",
            **common,
        )
    )
    cases.append(
        Case(
            name="framework.use_backend_scope",
            run=lambda: _enter_and_exit(backend),
            layer="dispatch",
            validate=lambda: _enter_and_exit(backend),
            description="enter and leave a scoped backend context, which clears the kernel-lookup cache on both boundaries",
            **common,
        )
    )
    if backend in ACCELERATED:
        cases.append(
            Case(
                name="framework.kernel_lookup_cached",
                run=lambda: loading.load_backend(backend).add,
                layer="dispatch",
                validate=lambda: loading.load_backend(backend).add,
                description="resolve a kernel that the cache already holds",
                backends=ACCELERATED,
                **common,
            )
        )

        def cold_lookup() -> Any:
            loading.load_backend.cache_clear()
            return loading.load_backend(backend).add

        cases.append(
            Case(
                name="framework.kernel_lookup_cold",
                run=cold_lookup,
                layer="dispatch",
                validate=cold_lookup,
                description="construct a provider with an empty provider cache and retrieve its bound addition kernel",
                backends=ACCELERATED,
                **common,
            )
        )
        cases.append(
            Case(
                name="framework.policy_decision",
                run=lambda: _array_work_is_large_enough(_shape_size((64,)), 32),
                layer="dispatch",
                validate=lambda: _array_work_is_large_enough(_shape_size((64,)), 32),
                description="the workload-size policy that decides whether to accelerate an operation",
                backends=ACCELERATED,
                **common,
            )
        )
    return cases


def _enter_and_exit(backend: str) -> None:
    """Enter and leave a scoped backend context."""
    with ts.use_backend(backend):
        pass


def _add_scalars(left: object, right: object) -> object:
    """Add two host values, as a reference elementwise kernel does."""
    return left + right


def _metadata_cases(backend: str) -> list[Case]:
    """Measure dtype, shape, and broadcast resolution."""
    common: dict[str, Any] = {
        "family": "framework/metadata",
        "elements": 0,
        "tags": {"curve": "framework-metadata"},
    }
    scalar = tensor((1,), dtype_name="float64", kind="constant")
    matrix = tensor((32, 32), dtype_name="float64", kind="constant")
    return [
        Case(
            name="framework.result_dtype_tensor",
            run=lambda: result_dtype(ts.float64, scalar),
            layer="dispatch",
            validate=lambda: result_dtype(ts.float64, scalar),
            description="resolve the result dtype of a tensor operand",
            **common,
        ),
        Case(
            name="framework.result_dtype_scalar",
            run=lambda: result_dtype(ts.float64, 2.0),
            layer="dispatch",
            validate=lambda: result_dtype(ts.float64, 2.0),
            description="resolve the result dtype of a Python scalar operand",
            **common,
        ),
        Case(
            name="framework.shape_construction",
            run=lambda: Shape(32, 32),
            layer="dispatch",
            validate=lambda: Shape(32, 32),
            description="construct a validated Shape",
            **common,
        ),
        Case(
            name="framework.shape_size",
            run=lambda: Shape(32, 32).size,
            layer="dispatch",
            validate=lambda: Shape(32, 32).size,
            description="calculate a shape's element count",
            **common,
        ),
        Case(
            name="framework.strides_contiguous",
            run=lambda: Strides.contiguous(Shape(32, 32)),
            layer="dispatch",
            validate=lambda: Strides.contiguous(Shape(32, 32)),
            description="derive contiguous strides for a shape",
            **common,
        ),
        Case(
            name="framework.broadcast_shapes_equal",
            run=lambda: Shape(32, 32).broadcast_with(Shape(32, 32)),
            layer="dispatch",
            validate=lambda: Shape(32, 32).broadcast_with(Shape(32, 32)),
            description="agree a result shape between identical shapes",
            **common,
        ),
        Case(
            name="framework.broadcast_shapes_expand",
            run=lambda: Shape(32, 32).broadcast_with(Shape(32)),
            layer="dispatch",
            validate=lambda: Shape(32, 32).broadcast_with(Shape(32)),
            description="agree a result shape that expands a rank",
            **common,
        ),
        Case(
            name="framework.broadcast_to_equal_shape",
            run=lambda: broadcast_to(scalar, Shape(1)),
            layer="dispatch",
            validate=lambda: broadcast_to(scalar, Shape(1)),
            description="broadcasting to a shape the tensor already has, which returns it unchanged",
            **common,
        ),
        Case(
            name="framework.broadcast_then_operate",
            run=lambda: [
                _add_scalars(x, y)
                for x, y in zip(
                    broadcast_to(scalar, Shape(1))._data,
                    broadcast_to(scalar, Shape(1))._data,
                )
            ],
            layer="dispatch",
            validate=lambda: [
                _add_scalars(x, y)
                for x, y in zip(
                    broadcast_to(scalar, Shape(1))._data,
                    broadcast_to(scalar, Shape(1))._data,
                )
            ],
            description="broadcasting both operands and then applying the operation, as the reference kernels now do",
            **common,
        ),
        Case(
            name="framework.is_contiguous",
            run=lambda: matrix.is_contiguous,
            layer="dispatch",
            validate=lambda: matrix.is_contiguous,
            description="check a tensor's logical contiguity",
            **common,
        ),
    ]


def _construction_cases(backend: str) -> list[Case]:
    """Measure Tensor construction around an already-computed result."""
    common: dict[str, Any] = {
        "family": "framework/construction",
        "elements": 1,
        "tags": {"curve": "framework-construction"},
    }
    cases: list[Case] = []
    shape = (64,)
    if backend in ACCELERATED:
        kernels = loading.load_backend(backend)
        conversion = importlib.import_module(
            f"tensors.backend.{backend}.conversion"
        )
        array_module = importlib.import_module(
            "numpy" if backend == "numpy" else "cupy"
        )
        native = array_module.dtype(ts.float64.name)
        left = tensor(shape, dtype_name="float64", kind="ramp")
        right = tensor(shape, dtype_name="float64", kind="constant", value=2.0)
        prepared = (
            conversion.tensor_to_logical_array(left).astype(native, copy=False),
            conversion.tensor_to_logical_array(right).astype(native, copy=False),
        )
        storage = kernels.add(*prepared, dtype=ts.float64, output_shape=shape)
        if storage is None:
            raise Unsupported(
                "the binary kernel declined this configuration, so there is no native result to wrap"
            )
        cases.append(
            Case(
                name="framework.tensor_from_owned_storage",
                run=lambda: ts.Tensor._from_owned_storage(
                    storage, dtype=ts.float64, shape=Shape(*shape)
                ),
                layer="dispatch",
                validate=lambda: ts.Tensor._from_owned_storage(
                    storage, dtype=ts.float64, shape=Shape(*shape)
                ),
                description="wrap an already-computed provider result in a Tensor, which is what every accelerated operation ends with",
                backends=ACCELERATED,
                **common,
            )
        )
        tiny_left = tensor((4,), dtype_name="float64", kind="ramp")
        tiny_right = tensor((4,), dtype_name="float64", kind="constant")

        def rejected() -> Any:
            return execute_add(
                tiny_left,
                tiny_right,
                dtype=ts.float64,
                output_shape=(4,),
            )

        def validate_rejected() -> None:
            if rejected() is not None:
                raise Unsupported(
                    "this backend's workload policy accepted the operation instead of declining it, so there is no rejection to measure; the CUDA policy keeps supported work on the device at every size"
                )

        cases.append(
            Case(
                name="framework.dispatch_returning_none",
                run=rejected,
                layer="dispatch",
                validate=validate_rejected,
                description="a dispatch call the workload policy rejects, so it returns without resolving or running a kernel",
                backends=ACCELERATED,
                **common,
            )
        )
    values = [1.0] * 64
    cases.append(
        Case(
            name="framework.python_storage_from_values",
            run=lambda: PythonStorage.from_values(values, ts.float64),
            layer="dispatch",
            validate=lambda: PythonStorage.from_values(values, ts.float64),
            description="build host storage from Python values",
            **common,
        )
    )
    cases.append(
        Case(
            name="framework.tensor_empty",
            run=lambda: ts.Tensor([0.0]),
            layer="dispatch",
            validate=lambda: ts.Tensor([0.0]),
            description="construct the smallest possible Tensor",
            **common,
        )
    )
    return cases


def _graph_object_cases(backend: str) -> list[Case]:
    """Measure the graph and operation objects an eager call creates."""
    common: dict[str, Any] = {
        "family": "framework/graph-objects",
        "elements": 1,
        "gc_enabled": True,
        "tags": {"curve": "framework-graph-objects"},
    }
    scalar = tensor((1,), dtype_name="float64", kind="constant")
    left_node = VariableNode()
    right_node = VariableNode()
    operation_node = OperationNode(Add())
    cases = [
        Case(
            name="framework.operation_construction",
            run=Add,
            layer="dispatch",
            validate=Add,
            description="construct an immutable Operation object",
            **common,
        ),
        Case(
            name="framework.variable_node_construction",
            run=VariableNode,
            layer="dispatch",
            validate=VariableNode,
            description="construct an unbound graph value vertex",
            **common,
        ),
        Case(
            name="framework.operation_node_construction",
            run=lambda: OperationNode(Mul()),
            layer="dispatch",
            validate=lambda: OperationNode(Mul()),
            description="construct an operation vertex",
            **common,
        ),
        Case(
            name="framework.edge_construction",
            run=lambda: Edge(left_node, operation_node, label="input_0"),
            layer="dispatch",
            validate=lambda: Edge(left_node, operation_node, label="input_0"),
            description="record one edge between two vertices",
            **common,
        ),
        Case(
            name="framework.variable_construction",
            run=lambda: ts.Variable(scalar, requires_grad=False),
            layer="variable",
            validate=lambda: ts.Variable(scalar, requires_grad=False),
            description="construct a Variable, which also records its graph vertex",
            **common,
        ),
        Case(
            name="framework.record_operation",
            run=lambda: get_graph_state().record_operation(
                Add(), (left_node, right_node)
            ),
            layer="dispatch",
            validate=lambda: get_graph_state().record_operation(
                Add(), (left_node, right_node)
            ),
            description="record one operation invocation: a result vertex, an operation vertex, and the edges ordering the operands",
            **common,
        ),
    ]
    left = ts.Variable(scalar, requires_grad=False)
    right = ts.Variable(scalar, requires_grad=False)
    result_node = get_graph_state().record_operation(Add(), (left.node, right.node))
    operands = (left.node, right.node)
    cases.append(
        Case(
            name="framework.fragment_compile",
            run=lambda: Computation.from_nodes((result_node,), boundaries=operands),
            layer="graph-compile",
            validate=lambda: Computation.from_nodes(
                (result_node,), boundaries=operands
            ),
            description="compile the one-instruction fragment an eager operation builds, without running it",
            **common,
        )
    )
    eager_left = ts.Variable(scalar, requires_grad=False)
    eager_right = ts.Variable(scalar, requires_grad=False)
    cases.append(
        Case(
            name="framework.eager_operation_total",
            run=lambda: eager_left + eager_right,
            layer="variable",
            validate=lambda: eager_left + eager_right,
            description="one complete eager Variable operation: record, compile, and execute a one-instruction program",
            memory=True,
            **common,
        )
    )
    return cases


def groups() -> list[Group]:
    """Return the framework-overhead groups."""
    return [
        Group(name="framework/selection", factory=_selection_cases, suite="framework"),
        Group(name="framework/metadata", factory=_metadata_cases, suite="framework"),
        Group(
            name="framework/construction",
            factory=_construction_cases,
            suite="framework",
        ),
        Group(
            name="framework/graph-objects",
            factory=_graph_object_cases,
            suite="framework",
        ),
    ]


__all__ = ["groups"]
