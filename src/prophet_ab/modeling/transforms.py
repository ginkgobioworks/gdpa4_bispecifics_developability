"""Per-label target transform registry.

Each label in the modeling sweep can optionally be predicted in a transformed
space (e.g., log, logit). Transforms are applied via sklearn's
TransformedTargetRegressor, which wraps the base model and auto-inverts
predictions so that all metrics are computed on the original measurement
scale.

Decision D-2026-04-29-TARGET-TRANSFORMS governs the assignments below.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np


class TransformSpec(NamedTuple):
    name: str
    func: Callable[[np.ndarray], np.ndarray]
    inverse_func: Callable[[np.ndarray], np.ndarray]


_EPS = 1e-6

LOG = TransformSpec(
    name="log",
    func=lambda y: np.log(np.maximum(y, _EPS)),
    inverse_func=np.exp,
)

LOG1P = TransformSpec(
    name="log1p",
    func=np.log1p,
    inverse_func=np.expm1,
)

LOGIT_01 = TransformSpec(
    name="logit_01",
    func=lambda y: np.log(np.clip(y, _EPS, 1 - _EPS) / (1 - np.clip(y, _EPS, 1 - _EPS))),
    inverse_func=lambda z: 1.0 / (1.0 + np.exp(-z)),
)

LOGIT_100 = TransformSpec(
    name="logit_100",
    func=lambda y: np.log(np.clip(y, _EPS, 100 - _EPS) / (100 - np.clip(y, _EPS, 100 - _EPS))),
    inverse_func=lambda z: 100.0 / (1.0 + np.exp(-z)),
)

LABEL_TRANSFORMS: dict[str, TransformSpec] = {
    "label__sehplc_pct_mono__default": LOGIT_100,
    "label__bvp_score_norm__default": LOGIT_01,
    "label__acsins_delta_Lmax__His/Arg, pH 6": LOG1P,
    "label__acsins_delta_Lmax__His/NaCl, pH 6": LOG1P,
    "label__smachplc_retentiontime__default": LOG,
}


def get_transform(label: str) -> TransformSpec | None:
    return LABEL_TRANSFORMS.get(label)
