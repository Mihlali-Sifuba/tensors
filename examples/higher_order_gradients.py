"""Higher-order differentiation with explicit gradient graphs."""

import tensors as ts

# A scalar polynomial: y = x³.
x = ts.Variable([3.0])
y = ts.sum(x ** 3.0)

first = ts.grad(y, x, create_graph=True)
second = ts.grad(first, x, create_graph=True)
third = ts.grad(second, x)

print("y:", y.data.item())
print("dy/dx:", first.data.item())       # 27
print("d²y/dx²:", second.data.item())    # 18
print("d³y/dx³:", third.item())          # 6

# Higher derivatives also work through a matrix expression.
inputs = ts.Variable([[2.0]], requires_grad=False)
weight = ts.Variable([[3.0]])
loss = ts.sum((inputs @ weight) ** 2.0)

loss_gradient = ts.grad(loss, weight, create_graph=True)
loss_curvature = ts.grad(loss_gradient, weight)

print("\nmatrix loss:", loss.data.item())
print("d(loss)/d(weight):", loss_gradient.data.item())  # 24
print("d²(loss)/d(weight)²:", loss_curvature.item())     # 8
