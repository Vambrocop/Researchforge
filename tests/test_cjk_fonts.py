"""Stage 0 — CJK figure fonts.

The engine's figures are produced through one unified entry, `_init_mpl_style`
(called once per analysis run). These tests verify that entry now enables CJK
rendering: when a Chinese/JK font is installed it is prepended to the fallback
chain (with axes.unicode_minus off), so a figure with Chinese labels renders
real glyphs instead of "tofu" boxes — and that the detection degrades gracefully
when no CJK font exists.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from researchforge.executor._helpers.core import _detect_cjk_font, _init_mpl_style


def test_detect_is_graceful() -> None:
    # never raises; returns a font name or None
    r = _detect_cjk_font()
    assert r is None or isinstance(r, str)


def test_init_sets_unicode_minus() -> None:
    # unicode_minus must be off regardless of whether a CJK font exists (many CJK
    # fonts lack the U+2212 minus glyph)
    _init_mpl_style()
    import matplotlib.pyplot as plt

    assert plt.rcParams["axes.unicode_minus"] is False


@pytest.mark.skipif(_detect_cjk_font() is None, reason="no CJK font installed on this machine")
def test_cjk_label_renders_without_tofu(tmp_path: Path) -> None:
    cjk = _detect_cjk_font()
    _init_mpl_style()
    import matplotlib.pyplot as plt

    # the detected CJK font leads the fallback chain
    assert plt.rcParams["font.sans-serif"][0] == cjk

    # render a Chinese-labelled figure (incl. a minus sign) and assert matplotlib
    # emits NO "Glyph ... missing from font" warning — i.e. no tofu boxes
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, ax = plt.subplots()
        ax.set_title("中文标题：水平变化 −1.5")
        ax.set_xlabel("时间（周）")
        ax.set_ylabel("数值")
        ax.plot([0, 1, 2], [-1.5, 0.0, 1.5])
        fig.savefig(tmp_path / "cjk.png")
        plt.close(fig)
    missing = [w for w in caught if "missing from" in str(w.message).lower()]
    assert not missing, f"CJK glyphs not rendered (tofu): {[str(w.message) for w in missing]}"
    assert (tmp_path / "cjk.png").exists()
