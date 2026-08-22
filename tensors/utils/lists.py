"""Utilities for validating and transforming nested lists."""

from typing import Any, List, Tuple


def flatten_nested_list(nested_list: List[Any]) -> List[Any]:
    """Flatten a nested list iteratively and reject cyclic input."""
    result: List[Any] = []
    stack: List[Tuple[Any, bool]] = [(nested_list, False)]
    active: set[int] = set()

    while stack:
        item, leaving = stack.pop()
        if isinstance(item, list):
            identity = id(item)
            if leaving:
                active.remove(identity)
                continue
            if identity in active:
                raise ValueError("Cyclic nested lists are not valid tensor data")
            active.add(identity)
            stack.append((item, True))
            for child in reversed(item):
                stack.append((child, False))
        else:
            result.append(item)

    return result


def infer_nested_list_shape(nested_list: List[Any]) -> Tuple[int, ...]:
    """Infer the shape of a nested list and reject ragged or cyclic input."""
    if not isinstance(nested_list, list):
        return ()

    stack: List[List[Any]] = [[nested_list, 0, None]]
    active = {id(nested_list)}

    while stack:
        current, child_index, expected = stack[-1]
        if child_index == len(current):
            child_shape = () if expected is None else expected
            completed_shape = (len(current),) + child_shape
            active.remove(id(current))
            stack.pop()
            if not stack:
                return completed_shape
            parent = stack[-1]
            if parent[2] is None:
                parent[2] = completed_shape
            elif parent[2] != completed_shape:
                raise ValueError(
                    "Ragged nested lists are not valid tensor data: "
                    f"expected child shape {parent[2]}, got {completed_shape}"
                )
            continue

        child = current[child_index]
        stack[-1][1] += 1
        if isinstance(child, list):
            if id(child) in active:
                raise ValueError("Cyclic nested lists are not valid tensor data")
            active.add(id(child))
            stack.append([child, 0, None])
        else:
            if expected is None:
                stack[-1][2] = ()
            elif expected != ():
                raise ValueError(
                    "Ragged nested lists are not valid tensor data: "
                    f"expected child shape {expected}, got ()"
                )

    raise RuntimeError("nested-list shape inference did not complete")


__all__ = ["flatten_nested_list", "infer_nested_list_shape"]
