"""Inspect and replay the recorded computation for z = x + y."""

import tensors as ts


x = ts.Variable([2.0], name="x")
y = ts.Variable([3.0], name="y")
z = x + y

computation = ts.graph.Computation(z)

print("x Variable:", x)
print("x Node:", x.node)
print("y Variable:", y)
print("y Node:", y.node)
print("z Variable:", z)
print("z Node:", z.node)
print("Computation output:", computation.output)
print("Computation nodes:", computation.nodes)

# Change the leaf values, then replay the already-recorded add operation.
x.data = ts.Tensor([10.0])
y.data = ts.Tensor([20.0])

print("z before recomputation:", z.data.item())
print("recomputed z:", computation.forward().item())
