"""Figure-label localization (English → Chinese) — the Stage-0 follow-on.

Verifies the glossary translator and the in-place figure localizer used by the
savefig chokepoint. Translation targets titles / axis labels / legend entries only,
and is gated on a CJK font being present (so headless/CI keeps English, no tofu).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchforge.executor._helpers.core import (
    _detect_cjk_font,
    _init_mpl_style,
    _localize_figure,
    _translate_label,
)


def test_translate_label_glossary() -> None:
    assert _translate_label("count") == "计数"
    assert _translate_label("survival probability") == "生存概率"
    assert _translate_label("time step") == "时间步"
    assert _translate_label("Posterior density") == "后验密度"        # case-insensitive
    assert _translate_label("number of nodes") == "节点数"            # multi-word wins
    # data / unknown tokens are left untouched
    assert _translate_label("PC1") == "PC1"
    assert _translate_label("Q=0.342") == "Q=0.342"
    assert _translate_label("") == ""


def test_translate_label_substring_in_phrase() -> None:
    # known words inside a longer label are translated; numbers/colnames survive
    out = _translate_label("Residuals vs fitted")
    assert "残差" in out and "拟合" in out
    out2 = _translate_label("frequency of x_var")
    assert "频数" in out2 and "x_var" in out2  # the column token is preserved


def test_localize_figure_translates_labels() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_title("Residuals vs fitted")
    ax.set_xlabel("fitted values")
    ax.set_ylabel("residuals")
    ax.plot([0, 1], [0, 1], label="observed")
    ax.legend()
    _localize_figure(fig)
    assert ax.get_xlabel() == "拟合值"
    assert ax.get_ylabel() == "残差"
    assert "残差" in ax.get_title() and "拟合" in ax.get_title()
    assert ax.get_legend().get_texts()[0].get_text() == "观测"
    plt.close(fig)


@pytest.mark.skipif(_detect_cjk_font() is None, reason="no CJK font -> localization stays off (English)")
def test_savefig_chokepoint_localizes(tmp_path: Path) -> None:
    _init_mpl_style()  # installs the savefig localizer when a CJK font is present
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_xlabel("time")
    ax.set_ylabel("count")
    ax.plot([0, 1, 2], [1, 2, 3])
    fig.savefig(tmp_path / "f.png")  # patched savefig localizes in place first
    assert ax.get_xlabel() == "时间"
    assert ax.get_ylabel() == "计数"
    assert (tmp_path / "f.png").exists()
    plt.close(fig)
