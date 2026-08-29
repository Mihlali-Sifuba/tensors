"""Functional parameter initialization."""

from ._special import orthogonal, truncated_normal
from ._variance import (
    he_normal,
    he_uniform,
    lecun_normal,
    lecun_uniform,
    variance_scaling,
    xavier_normal,
    xavier_uniform,
)


__all__ = [
    "he_normal",
    "he_uniform",
    "lecun_normal",
    "lecun_uniform",
    "orthogonal",
    "truncated_normal",
    "variance_scaling",
    "xavier_normal",
    "xavier_uniform",
]
