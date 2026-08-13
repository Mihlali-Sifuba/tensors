# Examples

Runnable examples of the public `tensors` API.

Run an example from the project root:

```powershell
python -m examples.higher_order_gradients
```

## Suggested learning order

1. `computation_forward.py` — inspect the graph for `z = x + y` and replay it with new input values.
2. `graph_structure.py` — build the directed structure for `z = x + y` manually and through `GraphState`.
3. `higher_order_gradients.py` — first, second, and third derivatives through scalar and matrix expressions.
4. `multilayer_perceptron.py` — a two-layer neural network trained with Adam.
5. `mlp_threads.py` — compare independent MLP training on one thread and two threads.
