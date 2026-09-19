"""The Python backend: the dependency-free reference implementation.

``storage`` owns its ``array.array`` buffers and ``kernels`` holds one module
per operation. These kernels define the shape, dtype, error, and
differentiation semantics every other backend is measured against.
"""
