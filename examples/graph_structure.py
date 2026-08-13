"""Build and inspect the graph structure for z = x + y."""

import tensors as ts
from tensors.graph.state import GraphState


# Build the graph directly with Node and Edge objects.
x = ts.graph.Node(label="x")
y = ts.graph.Node(label="y")
add = ts.graph.Node(label="add")

x_to_add = ts.graph.Edge(x, add, label="input_0")
y_to_add = ts.graph.Edge(y, add, label="input_1")

print("manual nodes:", x, y, add)
print("manual edges:", x_to_add, y_to_add)
print("add inputs:", add.inputs)
print("x outputs:", x.outputs)
print("y outputs:", y.outputs)

# Build the same structure through a GraphState, which owns its nodes and edges.
graph = GraphState()
state_x = graph.add_node(label="x")
state_y = graph.add_node(label="y")
state_add = graph.add_node(label="add")

graph.add_edge(state_x, state_add, label="input_0")
graph.add_edge(state_y, state_add, label="input_1")

print("graph-state nodes:", graph.nodes)
print("graph-state edges:", graph.edges)
