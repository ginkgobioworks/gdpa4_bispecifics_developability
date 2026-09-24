"""Datapoints matplotlib style, palette, and layout helpers.

This is a local stand-in for the original ``datapoints_figures`` package.
Call ``set_manuscript_style()`` at the top of every notebook that writes
figures.

Example::

    from datapoints_figures import (
        set_manuscript_style,
        DATAPOINTS_COLORS,
        wrap_label,
        equalize_axes,
    )
    set_manuscript_style()
"""

from .layout import equalize_axes, grid_figsize, wrap_label, wrap_title
from .plots import (
    generate_all_plots,
    plot_cdf,
    plot_histogram,
    plot_range_counts,
    plot_sorted_area,
    plot_sorted_bars,
)
from .style import (
    DATAPOINTS_CMAP,
    DATAPOINTS_COLORS,
    DATAPOINTS_PALETTE,
    FILL_ALPHA,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_LEGEND_TITLE,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    FULL_WIDTH,
    GRID_HSPACE,
    GRID_WSPACE,
    LINE_ALPHA,
    LINE_WIDTH,
    LINE_WIDTH_THIN,
    REFERENCE_LINE_STYLE,
    SINGLE_COL_WIDTH,
    SPINE_WIDTH,
    TICK_LENGTH,
    TICK_WIDTH,
    WHITE_TO_PURPLE,
    configure_font,
    datapoints_style,
    get_colors_from_cmap,
    set_datapoints_style,
    set_manuscript_style,
)

__all__ = [
    "DATAPOINTS_CMAP",
    "DATAPOINTS_COLORS",
    "DATAPOINTS_PALETTE",
    "FILL_ALPHA",
    "FONT_SIZE_LABEL",
    "FONT_SIZE_LEGEND",
    "FONT_SIZE_LEGEND_TITLE",
    "FONT_SIZE_TICK",
    "FONT_SIZE_TITLE",
    "FULL_WIDTH",
    "GRID_HSPACE",
    "GRID_WSPACE",
    "LINE_ALPHA",
    "LINE_WIDTH",
    "LINE_WIDTH_THIN",
    "REFERENCE_LINE_STYLE",
    "SINGLE_COL_WIDTH",
    "SPINE_WIDTH",
    "TICK_LENGTH",
    "TICK_WIDTH",
    "WHITE_TO_PURPLE",
    "configure_font",
    "datapoints_style",
    "equalize_axes",
    "generate_all_plots",
    "get_colors_from_cmap",
    "grid_figsize",
    "plot_cdf",
    "plot_histogram",
    "plot_range_counts",
    "plot_sorted_area",
    "plot_sorted_bars",
    "set_datapoints_style",
    "set_manuscript_style",
    "wrap_label",
    "wrap_title",
]
