"""Feature-config registry.

Each config is a function `(all_columns, label_col) -> list[str]` that returns
the column names to use as features for a given label. Registered configs
become rows in the experiment sweep.
"""
from __future__ import annotations

from typing import Callable

from ..features import wide

ConfigFn = Callable[[list[str], str], list[str]]
CONFIGS: dict[str, ConfigFn] = {}


def _register(name: str):
    def deco(fn: ConfigFn) -> ConfigFn:
        CONFIGS[name] = fn
        return fn
    return deco


@_register("compositional_baseline")
def cfg_compositional_baseline(cols: list[str], label_col: str) -> list[str]:
    """Single column: the mean operator on the matching measurement (= s03 baseline)."""
    pref = wide.label_to_meas_prefix(label_col)
    target = pref + "mean"
    return [target] if target in cols else []


@_register("corresponding_experimental")
def cfg_corresponding(cols: list[str], label_col: str) -> list[str]:
    """All operators on the matching measurement only."""
    pref = wide.label_to_meas_prefix(label_col)
    return [c for c in cols if c.startswith(pref)]


@_register("all_experimental")
def cfg_all_experimental(cols: list[str], label_col: str) -> list[str]:
    """All measured features under all operators (across every assay)."""
    return [c for c in cols if c.startswith(wide.MEAS_PREFIX + wide.SEP)]


@_register("in_silico_only")
def cfg_in_silico_only(cols: list[str], label_col: str) -> list[str]:
    return [c for c in cols if c.startswith(wide.IS_PREFIX + wide.SEP)]


@_register("in_silico_plus_corresponding")
def cfg_is_plus_corr(cols: list[str], label_col: str) -> list[str]:
    return cfg_in_silico_only(cols, label_col) + cfg_corresponding(cols, label_col)


@_register("in_silico_plus_all_experimental")
def cfg_is_plus_all(cols: list[str], label_col: str) -> list[str]:
    return cfg_in_silico_only(cols, label_col) + cfg_all_experimental(cols, label_col)


# ---------------------------------------------------------------------------
# Arm configs — raw parent values (arm_a, arm_b) instead of operators
# ---------------------------------------------------------------------------


@_register("arm_corresponding")
def cfg_arm_corresponding(cols: list[str], label_col: str) -> list[str]:
    """Arm a + arm b for the matching measurement only (2 columns)."""
    pref = wide.label_to_arm_meas_prefix(label_col)
    return [c for c in cols if c.startswith(pref)]


@_register("arm_all_experimental")
def cfg_arm_all_experimental(cols: list[str], label_col: str) -> list[str]:
    """All arm-measured features (both arms, all assays)."""
    return [c for c in cols if c.startswith(wide.ARM_MEAS_PREFIX + wide.SEP)]


@_register("arm_in_silico_only")
def cfg_arm_in_silico_only(cols: list[str], label_col: str) -> list[str]:
    return [c for c in cols if c.startswith(wide.ARM_IS_PREFIX + wide.SEP)]


@_register("arm_in_silico_plus_corresponding")
def cfg_arm_is_plus_corr(cols: list[str], label_col: str) -> list[str]:
    return cfg_arm_in_silico_only(cols, label_col) + cfg_arm_corresponding(cols, label_col)


@_register("arm_in_silico_plus_all_experimental")
def cfg_arm_is_plus_all(cols: list[str], label_col: str) -> list[str]:
    return cfg_arm_in_silico_only(cols, label_col) + cfg_arm_all_experimental(cols, label_col)
