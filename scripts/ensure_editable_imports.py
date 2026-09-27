"""Keep the editable install importable when macOS hides ``.venv``.

CPython 3.12 and later skip ``.pth`` files that have the macOS hidden flag
(``UF_HIDDEN``). ``uv pip install -e`` records this repo's ``src`` layout in
such a file. This project lives under ``~/Documents``, and the whole ``.venv``
tree can be marked hidden. Python then never adds ``src`` to ``sys.path``, so
``import prophet_ab`` fails even though the package is installed. Recreating
the venv clears the flag only until it is set again.

``sitecustomize.py`` is imported by module name. That hidden-file skip does
not apply, so the source paths stay importable after the flag comes back.

Run from the repo root with the project interpreter::

    .venv/bin/python scripts/ensure_editable_imports.py

The command is idempotent. Run it again after ``uv venv`` and
``uv pip install -e``.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Hatchling editable roots. Order matches the .pth files from `uv pip install -e`.
SOURCE_DIRS = (ROOT / "src", ROOT / "datapoints_figures" / "src")
SITECUSTOMIZE_NAME = "sitecustomize.py"


def _sitecustomize_source() -> str:
    lines = ",\n    ".join(repr(str(path)) for path in SOURCE_DIRS)
    return f'''\
"""Added by scripts/ensure_editable_imports.py. Do not edit by hand."""
import sys

_SOURCE_DIRS = (
    {lines},
)
for _path in _SOURCE_DIRS:
    if _path not in sys.path:
        sys.path.insert(0, _path)
'''


def _clear_hidden_flag(venv: Path) -> None:
    """Drop UF_HIDDEN so the editable .pth files work too.

    ``chflags`` is macOS-only. A missing binary or a non-Darwin host is fine:
    the sitecustomize hook above does not need the flag cleared.
    """
    if sys.platform != "darwin" or not venv.is_dir():
        return
    # chflags -R will not descend into a directory that is itself hidden, so
    # the first pass only clears .venv and the second pass clears the tree.
    for _ in range(2):
        subprocess.run(["chflags", "-R", "nohidden", str(venv)], check=False)


def _venv_dir() -> Path:
    # sys.prefix is the venv when this file is run with .venv/bin/python.
    prefix = Path(sys.prefix).resolve()
    expected = (ROOT / ".venv").resolve()
    if prefix != expected:
        raise SystemExit(
            f"Run this with {expected / 'bin' / 'python'}, not {sys.executable} "
            f"(prefix is {prefix})."
        )
    return prefix


def main() -> None:
    venv = _venv_dir()
    purelib = Path(sysconfig.get_path("purelib"))
    destination = purelib / SITECUSTOMIZE_NAME
    destination.write_text(_sitecustomize_source())
    _clear_hidden_flag(venv)

    hidden_pth = 0
    for path in purelib.glob("*.pth"):
        flags = getattr(path.lstat(), "st_flags", 0)
        if flags & getattr(stat, "UF_HIDDEN", 0):
            hidden_pth += 1
    print(f"wrote {destination}")
    if hidden_pth:
        print(
            f"warning: {hidden_pth} .pth file(s) still hidden; "
            "imports use sitecustomize instead"
        )


if __name__ == "__main__":
    main()
