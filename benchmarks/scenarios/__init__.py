"""Whole workflows, measured end to end.

A scenario deliberately includes every layer at once, which is what
makes it the number a microbenchmark improvement ultimately has to
move. It is kept apart from the diagnostic suites because it cannot
attribute a cost to a layer; it can only say what the total is.
"""
