"""The CUDA backend, executed through CuPy.

``storage`` owns its device-resident ``cupy.ndarray`` buffers, ``conversion``
is the boundary between tensors and those arrays, and ``kernels`` holds one
module per operation. Kernels are imported on first execution, so selecting
another backend never imports CuPy.
"""
