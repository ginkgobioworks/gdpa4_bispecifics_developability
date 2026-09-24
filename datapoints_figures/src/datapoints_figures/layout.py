"""Layout helpers for manuscript figures.

``wrap_label`` / ``wrap_title`` wrap to a physical width (inches) and avoid
one-word last lines. ``equalize_axes`` makes a scatter square without
``set_aspect("equal")``, which distorts subplot boxes in a grid.
``grid_figsize`` estimates ``(fig_w, fig_h, cell_w)`` for constrained layout.

Example::

    from datapoints_figures import grid_figsize, wrap_title, equalize_axes
    w, h, cell_w = grid_figsize(2, 4)
    fig, axes = plt.subplots(2, 4, figsize=(w, h), layout="constrained")
    ax.set_title(wrap_title(long_title, cell_w))
    equalize_axes(ax)
"""
from __future__ import annotations

import textwrap

from .style import (
    FONT_SIZE_LABEL,
    FONT_SIZE_TITLE,
    FULL_WIDTH,
    GRID_HSPACE,
    GRID_WSPACE,
)


def _max_chars(width_in: float, fontsize: float) -> int:
    """Approximate characters that fit in ``width_in`` inches at ``fontsize`` pt.

    Manrope's average glyph is ~0.52 em; 72 pt = 1 inch.
    """
    if width_in <= 0 or fontsize <= 0:
        return 8
    char_in = (fontsize / 72.0) * 0.52
    return max(8, int(width_in / char_in))


def _wrap_paragraph(text: str, max_chars: int) -> str:
    collapsed = " ".join(text.split())
    if not collapsed:
        return text
    if len(collapsed) <= max_chars:
        return collapsed
    lines = textwrap.wrap(
        collapsed,
        width=max_chars,
        break_long_words=False,
        break_on_hyphens=True,
    )
    if not lines:
        return collapsed
    # Orphan prevention: a last line of one word is pulled onto with a word
    # from the previous line so the wrap does not leave a stranded token.
    if len(lines) >= 2 and len(lines[-1].split()) == 1:
        prev = lines[-2].split()
        if len(prev) >= 2:
            stolen = prev.pop()
            lines[-2] = " ".join(prev)
            lines[-1] = f"{stolen} {lines[-1]}"
    return "\n".join(lines)


def wrap_label(text: str, width_in: float, fontsize: float | None = None) -> str:
    """Wrap an axis label to ``width_in`` inches, keeping explicit newlines.

    Example::

        ax.set_xlabel(wrap_label("Expected (mean of parents)", cell_w))
        ax.set_yticklabels([wrap_label(lab, 1.5, FONT_SIZE_TICK) for lab in labs])
    """
    size = FONT_SIZE_LABEL if fontsize is None else fontsize
    max_chars = _max_chars(width_in, size)
    return "\n".join(
        _wrap_paragraph(part, max_chars) if part.strip() else part
        for part in str(text).split("\n")
    )


def wrap_title(text: str, width_in: float, fontsize: float | None = None) -> str:
    """Wrap a title to ``width_in`` inches (same orphan rule as ``wrap_label``).

    Example::

        ax.set_title(wrap_title(f"{panel}, ρ={rho:.2f}", cell_w, FONT_SIZE_TITLE))
    """
    size = FONT_SIZE_TITLE if fontsize is None else fontsize
    return wrap_label(text, width_in, size)


def equalize_axes(ax) -> None:
    """Set xlim == ylim from the current data limits. Does not change aspect.

    Use this instead of ``ax.set_aspect("equal")`` on multi-panel figures,
    which shrinks the subplot box.

    Example::

        ax.scatter(x, y)
        ax.plot(lims, lims, ls="--")
        equalize_axes(ax)
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lo = min(x0, y0)
    hi = max(x1, y1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def grid_figsize(
    nrows: int,
    ncols: int,
    cell_aspect: float = 1.0,
    full_width: float | None = None,
) -> tuple[float, float, float]:
    """Estimate figure size for a constrained-layout grid.

    ``cell_aspect`` is height / width of each panel. Returns
    ``(fig_width, fig_height, cell_width)`` in inches.

    Example::

        w, h, cell_w = grid_figsize(2, 4)
        fig, axes = plt.subplots(2, 4, figsize=(w, h), layout="constrained")
        w, h, cell_w = grid_figsize(1, 3, cell_aspect=0.85, full_width=FULL_WIDTH)
    """
    fw = FULL_WIDTH if full_width is None else float(full_width)
    nrows = max(int(nrows), 1)
    ncols = max(int(ncols), 1)
    extra_w = 0.7
    extra_h = 1.0
    cell_w = (fw - extra_w - GRID_WSPACE * max(ncols - 1, 0)) / ncols
    cell_h = cell_w * float(cell_aspect)
    fig_h = cell_h * nrows + GRID_HSPACE * max(nrows - 1, 0) + extra_h
    return fw, fig_h, cell_w
