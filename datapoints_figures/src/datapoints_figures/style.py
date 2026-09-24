"""Palette, fonts, and matplotlib rcParams for Datapoints figures.

Two style presets:

- ``set_manuscript_style()`` — publication defaults (3.5\" / 7.2\", 10–11 pt,
  0.8 pt spines). Use this in every notebook that writes a report figure.
- ``set_datapoints_style()`` — dashboard / exploratory defaults (thicker
  spines, 14–16 pt type).

Example::

    from datapoints_figures import set_manuscript_style, DATAPOINTS_COLORS
    set_manuscript_style()
    ax.scatter(x, y, color=DATAPOINTS_COLORS["blue"],
               edgecolors=DATAPOINTS_COLORS["gray"], linewidths=0.3)
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Named hex values used throughout the notebooks. The first five are the
# documented Datapoints brand tokens; the rest are the extended keys the
# manuscript figures already index (navy, teal, amber, green, coral, slate).

DATAPOINTS_COLORS: dict[str, str] = {
    "blue": "#4A61FF",
    "pink": "#FEAEFF",
    "purple": "#956FFF",
    "gray": "#888888",
    "red": "#FF8080",
    # Extended keys sampled from original s07/figure_5 boxen fills
    # (reports/figures/s07_boxenplot_model_config.png).
    "teal": "#41B1A7",
    "amber": "#E9B363",
    "navy": "#394380",
    "green": "#68A261",
    "coral": "#DE686D",
    "slate": "#5F6A70",
}

DATAPOINTS_PALETTE: list[str] = [
    DATAPOINTS_COLORS["blue"],
    DATAPOINTS_COLORS["purple"],
    DATAPOINTS_COLORS["pink"],
    DATAPOINTS_COLORS["teal"],
    DATAPOINTS_COLORS["amber"],
    DATAPOINTS_COLORS["navy"],
    DATAPOINTS_COLORS["coral"],
    DATAPOINTS_COLORS["green"],
    DATAPOINTS_COLORS["red"],
    DATAPOINTS_COLORS["slate"],
    DATAPOINTS_COLORS["gray"],
]

WHITE_TO_PURPLE = LinearSegmentedColormap.from_list(
    "white_to_purple",
    ["#FFFFFF", "#F3EEFF", "#C9B8FF", DATAPOINTS_COLORS["purple"], DATAPOINTS_COLORS["blue"]],
)
WHITE_TO_PURPLE.set_bad("#EEEEEE")

DATAPOINTS_CMAP = LinearSegmentedColormap.from_list(
    "datapoints_gray_blue",
    ["#F4F4F4", DATAPOINTS_COLORS["gray"], DATAPOINTS_COLORS["blue"]],
)
DATAPOINTS_CMAP.set_bad("#EEEEEE")

# ---------------------------------------------------------------------------
# Size tokens (manuscript)
# ---------------------------------------------------------------------------

SINGLE_COL_WIDTH = 3.5
FULL_WIDTH = 7.2

FONT_SIZE_TICK = 10
FONT_SIZE_LABEL = 11
FONT_SIZE_TITLE = 11
FONT_SIZE_LEGEND = 9
FONT_SIZE_LEGEND_TITLE = 10

SPINE_WIDTH = 0.8
TICK_WIDTH = 0.6
TICK_LENGTH = 3.5

# Absolute inches used by ``grid_figsize`` to estimate constrained-layout
# figure size. Constrained layout still owns the real subplot spacing.
GRID_HSPACE = 0.35
GRID_WSPACE = 0.40

# Dashboard / exploratory defaults (set_datapoints_style)
LINE_WIDTH = 4
LINE_WIDTH_THIN = 1.5
FILL_ALPHA = 0.3
LINE_ALPHA = 0.8
REFERENCE_LINE_STYLE = "--"

_FONTS_DIR = Path(__file__).resolve().parent / "fonts"
_FONT_LOADED: str | None = None


def _manuscript_rc() -> dict:
    return {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.grid": False,
        "axes.linewidth": SPINE_WIDTH,
        "axes.labelsize": FONT_SIZE_LABEL,
        "axes.titlesize": FONT_SIZE_TITLE,
        "axes.titlelocation": "center",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "figure.dpi": 150,
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "xtick.major.size": TICK_LENGTH,
        "ytick.major.size": TICK_LENGTH,
        "xtick.major.width": TICK_WIDTH,
        "ytick.major.width": TICK_WIDTH,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "legend.title_fontsize": FONT_SIZE_LEGEND_TITLE,
        "legend.loc": "best",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    }


def _dashboard_rc() -> dict:
    rc = _manuscript_rc()
    rc.update({
        "axes.linewidth": 2.0,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "xtick.major.width": 2,
        "ytick.major.width": 2,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "lines.linewidth": LINE_WIDTH,
    })
    return rc


def configure_font(path: str | Path) -> str:
    """Register a ``.ttf`` / ``.otf`` file and make it the default sans font.

    Returns the font family name matplotlib will use. Falls back to the
    system sans stack if the file cannot be loaded.

    Example::

        configure_font("/path/to/Manrope-Regular.ttf")
    """
    global _FONT_LOADED
    font_path = Path(path)
    if not font_path.is_file():
        raise FileNotFoundError(f"font file not found: {font_path}")
    font_manager.fontManager.addfont(str(font_path))
    name = font_manager.FontProperties(fname=str(font_path)).get_name()
    _FONT_LOADED = name
    _apply_sans(name)
    return name


def _apply_sans(name: str | None) -> None:
    stack = [name, "Arial", "Helvetica", "DejaVu Sans"] if name else [
        "Manrope", "Arial", "Helvetica", "DejaVu Sans",
    ]
    stack = [s for s in stack if s]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = stack
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = stack


def _ensure_font() -> str | None:
    """Load bundled Manrope faces and re-apply after seaborn rc updates."""
    global _FONT_LOADED
    if _FONT_LOADED:
        _apply_sans(_FONT_LOADED)
        return _FONT_LOADED
    if _FONTS_DIR.is_dir():
        for ttf in sorted(_FONTS_DIR.glob("*.ttf")):
            try:
                font_manager.fontManager.addfont(str(ttf))
            except Exception:
                continue
    regular = _FONTS_DIR / "Manrope-Regular.ttf"
    if regular.is_file():
        try:
            return configure_font(regular)
        except Exception:
            pass
    _apply_sans(None)
    return None


def _apply_rc(rc: dict) -> None:
    name = _ensure_font()
    rc = dict(rc)
    if name:
        rc["font.family"] = "sans-serif"
        rc["font.sans-serif"] = [name, "Arial", "Helvetica", "DejaVu Sans"]
    try:
        import seaborn as sns
        sns.set_theme(style="ticks", rc=rc)
    except ImportError:
        pass
    plt.rcParams.update(rc)
    mpl.rcParams.update(rc)
    _ensure_font()


def set_manuscript_style() -> None:
    """Publication style: thin spines, 10–11 pt type, 3.5\" / 7.2\" widths.

    Example::

        from datapoints_figures import set_manuscript_style
        set_manuscript_style()
    """
    _apply_rc(_manuscript_rc())


def set_datapoints_style() -> None:
    """Dashboard / exploratory style: thicker spines and 14–16 pt type."""
    _apply_rc(_dashboard_rc())


@contextmanager
def datapoints_style():
    """Apply dashboard style for a block, then restore previous rcParams.

    Example::

        from datapoints_figures import datapoints_style
        with datapoints_style():
            fig, ax = plt.subplots()
            ax.plot(x, y)
    """
    previous = plt.rcParams.copy()
    set_datapoints_style()
    try:
        yield
    finally:
        plt.rcParams.update(previous)


def get_colors_from_cmap(
    number_of_colors: int,
    cmap: LinearSegmentedColormap | None = None,
) -> list:
    """Evenly sample a sequential colormap.

    Example::

        colors = get_colors_from_cmap(number_of_colors=7)
        ax.scatter(x, y, color=colors[i])
    """
    if number_of_colors < 1:
        return []
    cm = WHITE_TO_PURPLE if cmap is None else cmap
    if number_of_colors == 1:
        return [cm(0.65)]
    return [cm(i / (number_of_colors - 1)) for i in range(number_of_colors)]
