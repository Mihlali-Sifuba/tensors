"""Kernels evaluated with CuPy device arrays.

Every kernel lives in its own module under a domain package; this
module re-exports them under the flat names the dispatch layer
resolves by.
"""

from tensors.backend.cuda.kernels.arithmetic.add import add as add
from tensors.backend.cuda.kernels.arithmetic.divide import divide as divide
from tensors.backend.cuda.kernels.arithmetic.multiply import multiply as multiply
from tensors.backend.cuda.kernels.arithmetic.power import power as power
from tensors.backend.cuda.kernels.arithmetic.subtract import subtract as subtract
from tensors.backend.cuda.kernels.convolution.convolution import (
    convolution as convolution,
)
from tensors.backend.cuda.kernels.convolution.convolution_gradient import (
    convolution_gradient as convolution_gradient,
)
from tensors.backend.cuda.kernels.creation.arange import arange as arange
from tensors.backend.cuda.kernels.creation.eye import eye as eye
from tensors.backend.cuda.kernels.creation.full import full as full
from tensors.backend.cuda.kernels.creation.linspace import linspace as linspace
from tensors.backend.cuda.kernels.creation.one_hot_targets import (
    one_hot_targets as one_hot_targets,
)
from tensors.backend.cuda.kernels.elementwise.abs import abs as abs
from tensors.backend.cuda.kernels.elementwise.abs_gradient import (
    abs_gradient as abs_gradient,
)
from tensors.backend.cuda.kernels.elementwise.arccos import arccos as arccos
from tensors.backend.cuda.kernels.elementwise.arccos_gradient import (
    arccos_gradient as arccos_gradient,
)
from tensors.backend.cuda.kernels.elementwise.arccosh import arccosh as arccosh
from tensors.backend.cuda.kernels.elementwise.arccosh_gradient import (
    arccosh_gradient as arccosh_gradient,
)
from tensors.backend.cuda.kernels.elementwise.arcsin import arcsin as arcsin
from tensors.backend.cuda.kernels.elementwise.arcsin_gradient import (
    arcsin_gradient as arcsin_gradient,
)
from tensors.backend.cuda.kernels.elementwise.arcsinh import arcsinh as arcsinh
from tensors.backend.cuda.kernels.elementwise.arcsinh_gradient import (
    arcsinh_gradient as arcsinh_gradient,
)
from tensors.backend.cuda.kernels.elementwise.arctan import arctan as arctan
from tensors.backend.cuda.kernels.elementwise.arctan_gradient import (
    arctan_gradient as arctan_gradient,
)
from tensors.backend.cuda.kernels.elementwise.arctanh import arctanh as arctanh
from tensors.backend.cuda.kernels.elementwise.arctanh_gradient import (
    arctanh_gradient as arctanh_gradient,
)
from tensors.backend.cuda.kernels.elementwise.clip import clip as clip
from tensors.backend.cuda.kernels.elementwise.clip_gradient import (
    clip_gradient as clip_gradient,
)
from tensors.backend.cuda.kernels.elementwise.cos import cos as cos
from tensors.backend.cuda.kernels.elementwise.cos_gradient import (
    cos_gradient as cos_gradient,
)
from tensors.backend.cuda.kernels.elementwise.cosh import cosh as cosh
from tensors.backend.cuda.kernels.elementwise.cosh_gradient import (
    cosh_gradient as cosh_gradient,
)
from tensors.backend.cuda.kernels.elementwise.division_denominator_gradient import (
    division_denominator_gradient as division_denominator_gradient,
)
from tensors.backend.cuda.kernels.elementwise.equal import equal as equal
from tensors.backend.cuda.kernels.elementwise.exp import exp as exp
from tensors.backend.cuda.kernels.elementwise.exp_gradient import (
    exp_gradient as exp_gradient,
)
from tensors.backend.cuda.kernels.elementwise.greater import greater as greater
from tensors.backend.cuda.kernels.elementwise.greater_equal import (
    greater_equal as greater_equal,
)
from tensors.backend.cuda.kernels.elementwise.less import less as less
from tensors.backend.cuda.kernels.elementwise.less_equal import less_equal as less_equal
from tensors.backend.cuda.kernels.elementwise.log import log as log
from tensors.backend.cuda.kernels.elementwise.log_gradient import (
    log_gradient as log_gradient,
)
from tensors.backend.cuda.kernels.elementwise.maximum import maximum as maximum
from tensors.backend.cuda.kernels.elementwise.maximum_gradient import (
    maximum_gradient as maximum_gradient,
)
from tensors.backend.cuda.kernels.elementwise.minimum import minimum as minimum
from tensors.backend.cuda.kernels.elementwise.minimum_gradient import (
    minimum_gradient as minimum_gradient,
)
from tensors.backend.cuda.kernels.elementwise.negate import negate as negate
from tensors.backend.cuda.kernels.elementwise.not_equal import not_equal as not_equal
from tensors.backend.cuda.kernels.elementwise.power_base_gradient import (
    power_base_gradient as power_base_gradient,
)
from tensors.backend.cuda.kernels.elementwise.power_exponent_gradient import (
    power_exponent_gradient as power_exponent_gradient,
)
from tensors.backend.cuda.kernels.elementwise.relu import relu as relu
from tensors.backend.cuda.kernels.elementwise.relu_gradient import (
    relu_gradient as relu_gradient,
)
from tensors.backend.cuda.kernels.elementwise.sigmoid import sigmoid as sigmoid
from tensors.backend.cuda.kernels.elementwise.sigmoid_gradient import (
    sigmoid_gradient as sigmoid_gradient,
)
from tensors.backend.cuda.kernels.elementwise.sign import sign as sign
from tensors.backend.cuda.kernels.elementwise.sign_gradient import (
    sign_gradient as sign_gradient,
)
from tensors.backend.cuda.kernels.elementwise.sin import sin as sin
from tensors.backend.cuda.kernels.elementwise.sin_gradient import (
    sin_gradient as sin_gradient,
)
from tensors.backend.cuda.kernels.elementwise.sinh import sinh as sinh
from tensors.backend.cuda.kernels.elementwise.sinh_gradient import (
    sinh_gradient as sinh_gradient,
)
from tensors.backend.cuda.kernels.elementwise.softplus import softplus as softplus
from tensors.backend.cuda.kernels.elementwise.softplus_gradient import (
    softplus_gradient as softplus_gradient,
)
from tensors.backend.cuda.kernels.elementwise.sqrt import sqrt as sqrt
from tensors.backend.cuda.kernels.elementwise.sqrt_gradient import (
    sqrt_gradient as sqrt_gradient,
)
from tensors.backend.cuda.kernels.elementwise.tan import tan as tan
from tensors.backend.cuda.kernels.elementwise.tan_gradient import (
    tan_gradient as tan_gradient,
)
from tensors.backend.cuda.kernels.elementwise.tanh import tanh as tanh
from tensors.backend.cuda.kernels.elementwise.tanh_gradient import (
    tanh_gradient as tanh_gradient,
)
from tensors.backend.cuda.kernels.elementwise.where import where as where
from tensors.backend.cuda.kernels.elementwise.where_gradient import (
    where_gradient as where_gradient,
)
from tensors.backend.cuda.kernels.fusion.fused_elementwise import (
    fused_elementwise as fused_elementwise,
)
from tensors.backend.cuda.kernels.fusion.fused_elementwise_backward import (
    fused_elementwise_backward as fused_elementwise_backward,
)
from tensors.backend.cuda.kernels.linalg.matmul import matmul as matmul
from tensors.backend.cuda.kernels.linalg.matmul_gradient import (
    matmul_gradient as matmul_gradient,
)
from tensors.backend.cuda.kernels.linalg.outer import outer as outer
from tensors.backend.cuda.kernels.linalg.outer_gradient import (
    outer_gradient as outer_gradient,
)
from tensors.backend.cuda.kernels.manipulation.assign_indices import (
    assign_indices as assign_indices,
)
from tensors.backend.cuda.kernels.manipulation.cast_tensor import (
    cast_tensor as cast_tensor,
)
from tensors.backend.cuda.kernels.manipulation.concat import concat as concat
from tensors.backend.cuda.kernels.manipulation.slice_scatter import (
    slice_scatter as slice_scatter,
)
from tensors.backend.cuda.kernels.manipulation.slice_tensor import (
    slice_tensor as slice_tensor,
)
from tensors.backend.cuda.kernels.manipulation.stack import stack as stack
from tensors.backend.cuda.kernels.manipulation.transpose import transpose as transpose
from tensors.backend.cuda.kernels.nn.binary_cross_entropy import (
    binary_cross_entropy as binary_cross_entropy,
)
from tensors.backend.cuda.kernels.nn.binary_cross_entropy_gradient import (
    binary_cross_entropy_gradient as binary_cross_entropy_gradient,
)
from tensors.backend.cuda.kernels.nn.cross_entropy import cross_entropy as cross_entropy
from tensors.backend.cuda.kernels.nn.cross_entropy_gradient import (
    cross_entropy_gradient as cross_entropy_gradient,
)
from tensors.backend.cuda.kernels.nn.distributions_valid import (
    distributions_valid as distributions_valid,
)
from tensors.backend.cuda.kernels.nn.log_softmax import log_softmax as log_softmax
from tensors.backend.cuda.kernels.nn.log_softmax_gradient import (
    log_softmax_gradient as log_softmax_gradient,
)
from tensors.backend.cuda.kernels.nn.softmax import softmax as softmax
from tensors.backend.cuda.kernels.nn.softmax_gradient import (
    softmax_gradient as softmax_gradient,
)
from tensors.backend.cuda.kernels.optim.adam_update import adam_update as adam_update
from tensors.backend.cuda.kernels.optim.adam_updates import adam_updates as adam_updates
from tensors.backend.cuda.kernels.optim.rmsprop_update import (
    rmsprop_update as rmsprop_update,
)
from tensors.backend.cuda.kernels.optim.rmsprop_updates import (
    rmsprop_updates as rmsprop_updates,
)
from tensors.backend.cuda.kernels.optim.sgd_update import sgd_update as sgd_update
from tensors.backend.cuda.kernels.optim.sgd_updates import sgd_updates as sgd_updates
from tensors.backend.cuda.kernels.reductions.argmax import argmax as argmax
from tensors.backend.cuda.kernels.reductions.argmin import argmin as argmin
from tensors.backend.cuda.kernels.reductions.logsumexp import logsumexp as logsumexp
from tensors.backend.cuda.kernels.reductions.logsumexp_gradient import (
    logsumexp_gradient as logsumexp_gradient,
)
from tensors.backend.cuda.kernels.reductions.reduce_max import reduce_max as reduce_max
from tensors.backend.cuda.kernels.reductions.reduce_max_gradient import (
    reduce_max_gradient as reduce_max_gradient,
)
from tensors.backend.cuda.kernels.reductions.reduce_mean import (
    reduce_mean as reduce_mean,
)
from tensors.backend.cuda.kernels.reductions.reduce_mean_gradient import (
    reduce_mean_gradient as reduce_mean_gradient,
)
from tensors.backend.cuda.kernels.reductions.reduce_min import reduce_min as reduce_min
from tensors.backend.cuda.kernels.reductions.reduce_min_gradient import (
    reduce_min_gradient as reduce_min_gradient,
)
from tensors.backend.cuda.kernels.reductions.reduce_norm import (
    reduce_norm as reduce_norm,
)
from tensors.backend.cuda.kernels.reductions.reduce_prod import (
    reduce_prod as reduce_prod,
)
from tensors.backend.cuda.kernels.reductions.reduce_prod_gradient import (
    reduce_prod_gradient as reduce_prod_gradient,
)
from tensors.backend.cuda.kernels.reductions.reduce_std import reduce_std as reduce_std
from tensors.backend.cuda.kernels.reductions.reduce_std_gradient import (
    reduce_std_gradient as reduce_std_gradient,
)
from tensors.backend.cuda.kernels.reductions.reduce_sum import reduce_sum as reduce_sum
from tensors.backend.cuda.kernels.reductions.reduce_sum_gradient import (
    reduce_sum_gradient as reduce_sum_gradient,
)
from tensors.backend.cuda.kernels.reductions.reduce_variance import (
    reduce_variance as reduce_variance,
)
from tensors.backend.cuda.kernels.reductions.reduce_variance_gradient import (
    reduce_variance_gradient as reduce_variance_gradient,
)
from tensors.backend.cuda.kernels.reductions.sum_products_to_shape import (
    sum_products_to_shape as sum_products_to_shape,
)
from tensors.backend.cuda.kernels.reductions.sum_to_shape import (
    sum_to_shape as sum_to_shape,
)
