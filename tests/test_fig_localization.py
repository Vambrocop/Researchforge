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


def test_translate_label_does_not_break_identifiers() -> None:
    # word boundary excludes digits/underscore: never translate a fragment of an identifier
    assert _translate_label("level") == "水平"                 # standalone -> translated
    assert _translate_label("1[y > level_k]") == "1[y > level_k]"  # 'level' inside id -> kept
    assert _translate_label("age2") == "age2"                  # 'age' + digit -> kept
    assert _translate_label("x_score") == "x_score"            # 'score' inside id -> kept


def test_localize_figure_idempotent_on_resave() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_title("count by group")
    ax.set_xlabel("count")
    _localize_figure(fig)
    once_title, once_xlabel = ax.get_title(), ax.get_xlabel()
    _localize_figure(fig)  # re-save would call it again; must be a no-op (sentinel)
    assert ax.get_title() == once_title
    assert ax.get_xlabel() == once_xlabel == "计数"
    plt.close(fig)


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


@pytest.mark.skipif(_detect_cjk_font() is None, reason="no CJK font -> nothing to prove per-call re-reads")
def test_savefig_language_policy_is_read_per_call_not_frozen_at_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2-4: in a long-lived process (e.g. the FastAPI server), _init_mpl_style only
    *installs* the savefig patch once — the language decision must be re-read on every
    save, not frozen from the first call. Simulates: zh-style init (patch installed,
    translating) -> RF_FIG_LANG=en set later -> a newly-saved figure must stay English."""
    import matplotlib.pyplot as plt

    monkeypatch.delenv("RF_FIG_LANG", raising=False)
    _init_mpl_style()  # first "request": default zh policy, installs the savefig patch

    fig_zh, ax_zh = plt.subplots()
    ax_zh.set_xlabel("time")
    fig_zh.savefig(tmp_path / "zh.png")
    assert ax_zh.get_xlabel() == "时间"  # sanity: translation is indeed active
    plt.close(fig_zh)

    # a later "request" flips the policy to English without re-installing the patch
    monkeypatch.setenv("RF_FIG_LANG", "en")
    fig_en, ax_en = plt.subplots()
    ax_en.set_xlabel("time")
    ax_en.set_ylabel("count")
    fig_en.savefig(tmp_path / "en.png")
    assert ax_en.get_xlabel() == "time"  # NOT translated -> policy was re-read at call time
    assert ax_en.get_ylabel() == "count"
    plt.close(fig_en)


def test_install_savefig_localizer_is_idempotent() -> None:
    import matplotlib.figure as _mfig

    from researchforge.executor._helpers.core import _install_savefig_localizer

    _install_savefig_localizer()
    patched_once = _mfig.Figure.savefig
    _install_savefig_localizer()  # calling again must not re-wrap (no double translation)
    assert _mfig.Figure.savefig is patched_once
