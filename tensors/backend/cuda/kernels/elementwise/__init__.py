"""Kernels evaluated with CuPy device arrays."""

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
