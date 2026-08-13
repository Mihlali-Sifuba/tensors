"""Compare two MLPs trained sequentially and in separate threads."""

import threading

import tensors as ts
from tensors.graph.state import get_graph_state


class MLP(ts.Graph):
    """A one-hidden-layer network: input -> ReLU hidden layer -> output."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.input_to_hidden = ts.Variable([[0.5, 0.25]], name=f"{name}.W₁")
        self.hidden_bias = ts.Variable([0.1, 0.1], name=f"{name}.b₁")
        self.hidden_to_output = ts.Variable([[0.25], [0.25]], name=f"{name}.W₂")
        self.output_bias = ts.Variable([0.0], name=f"{name}.b₂")

    def forward(self, inputs):
        hidden = ts.relu(inputs @ self.input_to_hidden + self.hidden_bias)
        return hidden @ self.hidden_to_output + self.output_bias


FEATURES = ts.Tensor([[0.0], [1.0], [2.0], [3.0]])
TARGETS = ts.Tensor([[1.0], [3.0], [5.0], [7.0]])


def train(model: MLP) -> dict[str, object]:
    """Train one independent model to learn y = 2x + 1."""
    optimizer = ts.optim.Adam(model.parameters(), learning_rate=0.05)

    for _ in range(300):
        predictions = model(FEATURES)
        loss = ts.mean((predictions - TARGETS) ** 2.0)
        optimizer.zero_grad()
        ts.backward(loss)
        optimizer.step()

    predictions = model(FEATURES)
    loss = ts.mean((predictions - TARGETS) ** 2.0)
    graph_state = get_graph_state()

    return {
        "model": model.input_to_hidden.name.split(".")[0],
        "thread": threading.current_thread().name,
        "thread_id": threading.get_ident(),
        "graph_state": graph_state,
        "loss": loss.data.item(),
        "predictions": predictions.data.tolist(),
    }


def print_result(result: dict[str, object]) -> None:
    """Display the thread and graph state used by one model run."""
    graph_state = result["graph_state"]
    print(f"{result['model']}:")
    print(f"  thread: {result['thread']} ({result['thread_id']})")
    print(f"  GraphState id: {id(graph_state)}")
    print(f"  final loss: {result['loss']}")
    print(f"  predictions: {result['predictions']}")


print("Two MLPs trained sequentially on the main thread")
sequential_a = train(MLP("sequential_a"))
sequential_b = train(MLP("sequential_b"))
print_result(sequential_a)
print_result(sequential_b)


print("\nTwo MLPs trained in separate threads")
threaded_results: list[dict[str, object]] = []
results_lock = threading.Lock()


def train_in_thread(name: str) -> None:
    result = train(MLP(name))
    with results_lock:
        threaded_results.append(result)


workers = [
    threading.Thread(target=train_in_thread, args=("threaded_a",), name="MLP-A"),
    threading.Thread(target=train_in_thread, args=("threaded_b",), name="MLP-B"),
]

for worker in workers:
    worker.start()
for worker in workers:
    worker.join()

for result in sorted(threaded_results, key=lambda item: str(item["model"])):
    print_result(result)
