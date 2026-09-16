"""Dependency-free performance measurement for the public tensors API.

Every computation is measured at several depths of the execution stack so
that overhead can be attributed to a layer rather than merely observed.
This package is measurement only: it imports the package under test and
never modifies it.
"""
