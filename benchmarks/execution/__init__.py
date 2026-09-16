"""The machinery that runs a workload, measured on its own.

These have no semantic operand to speak of: recording a graph,
compiling it, replaying it, differentiating it, fusing a run of
instructions, moving storage between backends, synchronizing a
device, importing the package, and running two graphs at once. What
they cost is framework cost, which is why they are not workloads.
"""
