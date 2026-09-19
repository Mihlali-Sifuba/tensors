"""Models a scenario is built from.

A scenario measures what running a model costs; it should not also be
where the model is defined, because the same model is measured by more
than one of them. These are deliberately plain: enough structure to
exercise the graph, the reverse pass, and an optimizer, and no more.
"""
