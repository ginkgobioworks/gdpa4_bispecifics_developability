"""Parse the wide-feature column names produced by `wide.py` back into parts.

Column conventions (see `wide.py`):
- `meas__{value_col}__{condition}__{operator}`
- `is__{source}__{feature}__{operator}`
- `label__{value_col}__{condition}`
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .. import schema
from . import wide


@dataclass(frozen=True)
class ParsedFeature:
    kind: str                       # "meas" or "is"
    source: str                     # in-silico source ("MOE", ...) or assay name
                                    # (== value_col) for measured features
    operator: str                   # mean / min / max / geomean / abs_diff
    base_feature: Optional[str] = None    # in-silico feature name; None for meas
    condition: Optional[str] = None       # measured condition; None for is
    value_col: Optional[str] = None       # measured value_col; None for is


def is_in_silico(col: str) -> bool:
    return col.startswith(wide.IS_PREFIX + wide.SEP)


def is_measured(col: str) -> bool:
    return col.startswith(wide.MEAS_PREFIX + wide.SEP)


def is_arm_measured(col: str) -> bool:
    return col.startswith(wide.ARM_MEAS_PREFIX + wide.SEP)


def is_arm_in_silico(col: str) -> bool:
    return col.startswith(wide.ARM_IS_PREFIX + wide.SEP)


def parse_feature_name(col: str) -> ParsedFeature:
    parts = col.split(wide.SEP)
    if len(parts) != 4:
        raise ValueError(f"unrecognized feature name (need 4 segments): {col!r}")
    kind, a, b, op = parts
    if kind == wide.MEAS_PREFIX:
        return ParsedFeature(
            kind="meas", source=a, operator=op,
            value_col=a, condition=b,
        )
    if kind == wide.IS_PREFIX:
        return ParsedFeature(
            kind="is", source=a, operator=op,
            base_feature=b,
        )
    if kind == wide.ARM_MEAS_PREFIX:
        return ParsedFeature(
            kind="arm_meas", source=a, operator=op,
            value_col=a, condition=b,
        )
    if kind == wide.ARM_IS_PREFIX:
        return ParsedFeature(
            kind="arm_is", source=a, operator=op,
            base_feature=b,
        )
    raise ValueError(f"unrecognized feature kind {kind!r} in {col!r}")


def parse_label_name(col: str) -> tuple[str, str]:
    """Return (value_col, condition) for a `label__{value_col}__{condition}` string."""
    parts = col.split(wide.SEP)
    if len(parts) != 3 or parts[0] != wide.LABEL_PREFIX:
        raise ValueError(f"unrecognized label name: {col!r}")
    return parts[1], parts[2]


# ---------------------------------------------------------------------------
# Display helpers — raw identifiers → human-readable strings for figures
# ---------------------------------------------------------------------------

def display_value_col(vc: str, condition: Optional[str] = None) -> str:
    """'hihplc_normretentiontime' → 'HIC RT (norm)', optionally with condition."""
    name = schema.VALUE_COL_DISPLAY.get(vc, vc)
    if condition is None or condition == "default":
        return name
    cond = schema.CONDITION_DISPLAY.get(condition, condition)
    if not cond or cond == name:
        return name
    return f"{name} @ {cond}"


def display_label(col: str) -> str:
    """'label__acsins_delta_Lmax__1X PBS' → 'AC-SINS ΔLmax @ 1× PBS'."""
    vc, cond = parse_label_name(col)
    return display_value_col(vc, cond)


def display_config(name: str) -> str:
    return schema.CONFIG_DISPLAY.get(name, name)


def display_operator(op: str) -> str:
    return schema.OPERATOR_DISPLAY.get(op, op)


def display_feature(col: str) -> str:
    """Full feature column → readable label for importance plots."""
    p = parse_feature_name(col)
    if p.kind == "arm_meas":
        base = display_value_col(p.value_col, p.condition)
        return f"{base} / Arm {p.operator.upper()}"
    if p.kind == "arm_is":
        return f"{p.source} / {p.base_feature} / Arm {p.operator.upper()}"
    op = display_operator(p.operator)
    if p.kind == "meas":
        base = display_value_col(p.value_col, p.condition)
        return f"{base} / {op}"
    return f"{p.source} / {p.base_feature} / {op}"
