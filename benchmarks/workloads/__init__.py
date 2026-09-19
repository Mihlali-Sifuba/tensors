"""The operations the library exposes, grouped by what they mean.

A workload module owns one semantic domain and the full ladder used
to measure it: the provider call, the guarded kernel, dispatch, the
public operation, an eager Variable, and a replayed graph, all in one
group over the same operands. Splitting those apart would make the
difference between two of them a difference of input rather than of
overhead, so nothing here is organized by layer or by backend.
"""
