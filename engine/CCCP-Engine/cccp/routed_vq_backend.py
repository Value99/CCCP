"""Public constructors for the model-independent routed-codebook backend.

The resident implementation was historically placed next to one model
adapter.  Runtime selection must never import that model-named module or make
its choice from an architecture label.  This façade is the only construction
boundary; it accepts a manifest-derived topology and returns the same common
codebook executor for DSV4, GLM, Kimi, Qwen, and future formats.
"""

from __future__ import annotations

from .routed_vq_resident import (
    ResidentRoutedVQPool,
    RoutedVQLayoutPlan,
    build_primary_dense_packed_plan,
    build_routed_vq_layer_plan,
)


__all__ = [
    "ResidentRoutedVQPool",
    "RoutedVQLayoutPlan",
    "build_primary_dense_packed_plan",
    "build_routed_vq_layer_plan",
]
