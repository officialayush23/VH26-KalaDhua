"""AURA training pipeline: cache traces in, model bundles out.

The public surface is deliberately small. Everything the Rust engine depends on
is either the feature vector (``FEATURE_NAMES``) or the bundle contract
(``export``), and both are pinned by ``tests/golden/feature_vectors.json``.
"""

from __future__ import annotations

from .config import Pricing, TrainingConfig, load_config
from .features import FEATURE_GROUPS, FEATURE_NAMES, N_FEATURES, AccessEvent, FeatureBuilder

__all__ = [
    "AccessEvent",
    "FEATURE_GROUPS",
    "FEATURE_NAMES",
    "FeatureBuilder",
    "N_FEATURES",
    "Pricing",
    "TrainingConfig",
    "load_config",
]

__version__ = "0.1.0"
