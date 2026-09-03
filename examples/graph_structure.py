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

# A GraphState registers vertices and edges without owning their lifetime.
graph = GraphState()
state_x = graph.add_variable_node(ts.Variable([1.0], name="state_x"))
state_y = graph.add_variable_node(ts.Variable([2.0], name="state_y"))
state_add = graph.add_operation_node(Add())
state_z = graph.add_variable_node(ts.Variable([3.0], name="state_z"))

graph.add_edge(state_x, state_add, label="input_0")
graph.add_edge(state_y, state_add, label="input_1")
graph.add_edge(state_add, state_z, label="result")

print("\ngraph-state nodes:", graph.nodes)
print("graph-state edges:", graph.edges)

# A scalar operand is an ordinary graph Variable rather than a hidden flag.
scaled = left * 3.0
operands = scaled.node.producer.operands
print("\nscalar operand is a Variable:", isinstance(operands[1], ts.Variable))
print("scalar operand value:", operands[1].data.item())
print("scalar operand requires grad:", operands[1].requires_grad)
