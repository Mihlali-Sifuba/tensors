"""The NumPy backend.

``storage`` owns its ``numpy.ndarray`` buffers, ``conversion`` is the boundary
between tensors and those arrays, and ``kernels`` holds one module per
operation. Kernels are imported on first execution, so selecting another
backend never imports NumPy.
"""
