"""Emit the numerical source data associated with one manuscript figure.

Figure scripts call :func:`emit` after rendering. The extractors remain small,
data-only programs so plotting and source-data calculations share the same
processed inputs without coupling one figure script to another.
"""
from __future__ import annotations

import runpy
from pathlib import Path

from . import paths

_EXTRACTORS = {
    "main_01": "extract_main_fig1.py",
    "main_02": "extract_main_fig2.py",
    "main_03": "extract_main_fig3.py",
    "main_04": "extract_main_fig4.py",
    "main_05": "extract_main_fig5.py",
    "main_06": "extract_main_fig6.py",
    "supp_01": "extract_supp_fig1.py",
    "supp_02": "extract_supp_fig2.py",
    "supp_04": "extract_supp_fig4.py",
    "supp_05": "extract_supp_fig5.py",
    "supp_07": "extract_supp_fig7.py",
    "supp_08": "extract_supp_fig8.py",
    "supp_09": "extract_supp_fig9.py",
    "supp_11": "extract_supp_fig11.py",
    "supp_13": "extract_supp_fig13.py",
    "supp_14": "extract_supp_fig14.py",
    "supp_15": "extract_supp_fig15.py",
    "supp_16": "extract_supp_fig16.py",
}


def emit(figure_id: str) -> None:
    """Write source-data CSVs for ``figure_id``.

    ----------
    figure_id:
        Final manuscript identifier such as ``"main_03"`` or ``"supp_14"``.
    """
    try:
        filename = _EXTRACTORS[figure_id]
    except KeyError as exc:
        raise ValueError(f"unsupported figure id: {figure_id}") from exc

    script = (
        paths.REPO_ROOT
        / "reports"
        / "figures"
        / "figure_data_supplement"
        / "scripts"
        / filename
    )
    if not script.is_file():
        raise FileNotFoundError(script)
    runpy.run_path(str(script), run_name="__main__")


def extracted_dir() -> Path:
    """Return the generated source-data CSV directory."""
    return paths.FIGURES / "figure_data_supplement" / "extracted"
