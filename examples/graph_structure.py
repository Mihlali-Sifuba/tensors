"""Build and inspect the graph structure for z = x + y.

A recorded graph alternates ``VariableNode -> OperationNode -> VariableNode``,
and every relationship is an ``Edge``.
"""

import tensors as ts
from tensors.graph import Edge, OperationNode, VariableNode
from tensors.graph.state import GraphState
from tensors.ops import Add


# Build the structure directly from vertex and edge objects.
x = ts.Variable([1.0], name="x")
y = ts.Variable([2.0], name="y")
z = ts.Variable([3.0], name="z")

add = OperationNode(Add())
x_to_add = Edge(x.node, add, label="input_0")
y_to_add = Edge(y.node, add, label="input_1")
add_to_z = Edge(add, z.node, label="result")

print("operand vertices:", x.node, y.node)
print("operation vertex:", add)
print("result vertex:", z.node)
print("edges:", x_to_add, y_to_add, add_to_z)
print("operation operands:", add.operands)
print("operation result:", add.result)
print("x outputs:", x.node.outputs)

# Record the same expression eagerly and read the structure back.
left = ts.Variable([1.0], name="left")
right = ts.Variable([2.0], name="right")
result = left + right

producer = result.node.producer
print("\nrecorded operation:", producer.label)
print("recorded operands:", [operand.name for operand in producer.operands])
print("result variable:", producer.result.name)
print("variable owns its node:", result.node.variable is result)

# A vertex is the graph identity of a value, so it can be recorded and
# connected before that value has been calculated.
a = ts.Variable([1.0, 2.0, 3.0], name="a")
b = ts.Variable([4.0, 5.0, 6.0], name="b")

sum_node = OperationNode(Add())
c_node = VariableNode()
Edge(a.node, sum_node, label="input_0")
Edge(b.node, sum_node, label="input_1")
Edge(sum_node, c_node, label="result")

print("\nresult vertex:", c_node)
print("result operands are known before the value:", sum_node.operand_nodes)
print("result is materialized:", c_node.is_bound)

# Executing the operation produces the Tensor the vertex was naming, and the
# runtime Variable is materialized against the vertex that named it.
c = c_node.materialize(sum_node.operation.forward(a.data, b.data), "c")

print("result is materialized:", c_node.is_bound)
print("materialized result:", c.name, c.data.tolist())
print("operation result is that Variable:", sum_node.result is c)
print("the vertex and its Variable name each other:", c.node is c_node)

# A GraphState registers vertices and edges without owning their lifetime,
# and records a whole operation invocation as one structural step.
graph = GraphState()
state_x = graph.add_variable_node()
state_y = graph.add_variable_node()
state_z = graph.record_operation(Add(), (state_x, state_y))

print("\nrecorded operation:", state_z.producer)
print("recorded operands:", state_z.producer.operand_nodes)
print("recorded result holds a value:", state_z.is_bound)

print("\ngraph-state nodes:", graph.nodes)
print("graph-state edges:", graph.edges)

# A scalar operand is an ordinary graph Variable rather than a hidden flag.
scaled = left * 3.0
operands = scaled.node.producer.operands
print("\nscalar operand is a Variable:", isinstance(operands[1], ts.Variable))
print("scalar operand value:", operands[1].data.item())
print("scalar operand requires grad:", operands[1].requires_grad)
