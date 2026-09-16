"""Reference kernels evaluated with ordinary Python arithmetic."""

from tensors.backend.python.kernels.reductions.argmax import argmax as argmax
from tensors.backend.python.kernels.reductions.argmin import argmin as argmin
from tensors.backend.python.kernels.reductions.logsumexp import logsumexp as logsumexp
from tensors.backend.python.kernels.reductions.logsumexp_gradient import (
    logsumexp_gradient as logsumexp_gradient,
)
from tensors.backend.python.kernels.reductions.reduce_max import (
    reduce_max as reduce_max,
)
from tensors.backend.python.kernels.reductions.reduce_max_gradient import (
    reduce_max_gradient as reduce_max_gradient,
)
from tensors.backend.python.kernels.reductions.reduce_mean import (
    reduce_mean as reduce_mean,
)
from tensors.backend.python.kernels.reductions.reduce_mean_gradient import (
    reduce_mean_gradient as reduce_mean_gradient,
)
from tensors.backend.python.kernels.reductions.reduce_min import (
    reduce_min as reduce_min,
)
from tensors.backend.python.kernels.reductions.reduce_min_gradient import (
    reduce_min_gradient as reduce_min_gradient,
)
from tensors.backend.python.kernels.reductions.reduce_norm import (
    reduce_norm as reduce_norm,
)
from tensors.backend.python.kernels.reductions.reduce_prod import (
    reduce_prod as reduce_prod,
)
from tensors.backend.python.kernels.reductions.reduce_prod_gradient import (
    reduce_prod_gradient as reduce_prod_gradient,
)
from tensors.backend.python.kernels.reductions.reduce_std import (
    reduce_std as reduce_std,
)
from tensors.backend.python.kernels.reductions.reduce_std_gradient import (
    reduce_std_gradient as reduce_std_gradient,
)
from tensors.backend.python.kernels.reductions.reduce_sum import (
    reduce_sum as reduce_sum,
)
from tensors.backend.python.kernels.reductions.reduce_sum_gradient import (
    reduce_sum_gradient as reduce_sum_gradient,
)
from tensors.backend.python.kernels.reductions.reduce_variance import (
    reduce_variance as reduce_variance,
)
from tensors.backend.python.kernels.reductions.reduce_variance_gradient import (
    reduce_variance_gradient as reduce_variance_gradient,
)
from tensors.backend.python.kernels.reductions.sum_products_to_shape import (
    sum_products_to_shape as sum_products_to_shape,
)
from tensors.backend.python.kernels.reductions.sum_to_shape import (
    sum_to_shape as sum_to_shape,
)
