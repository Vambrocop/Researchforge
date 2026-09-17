"""ONE bilingual role-vocabulary registry (Wave L · ColumnSemantics C0/C1).

Collapses the scattered treatment/block vocab definitions (executor _shared /
field_trials / statistics.group_compare, recommender.goals, profiler.roles) into a
single source. Import ``role_hint`` / ``is_treatment_named`` / ``has_design_signal`` from
here; never re-declare these. Neutral layer: no imports from executor/recommender
(profiler is the low dependency), so both may depend on it without a cycle.
"""
from __future__ import annotations

import re

ROLE_HINTS: dict[str, tuple[str, ...]] = {
    "block": ("block", "blk", "rep", "replicate", "replication", "plot", "site",
              "field", "batch", "区组", "重复", "组块", "块", "批次", "场地", "地块"),
    "treatment": ("treat", "treatment", "treated", "trt", "arm", "intervention",
                  "exposed", "exposure", "dose", "fert", "variety", "cultivar",
                  "genotype", "hybrid", "factor", "level", "处理", "品种", "剂量",
                  "水平", "施肥", "组别"),
    "time": ("year", "yr", "date", "time", "month", "quarter", "period",
             "day", "week", "wave", "日期", "年份", "月份", "季度"),
    # survival SHAPE vocabulary (a time-to-event duration + an event/censoring indicator).
    # English hints are the SAME set the recommender used, so has_survival is unchanged; the
    # Chinese synonyms only add coverage. Consumed by looks_like_survival (below).
    "duration": ("dur", "time", "tenure", "surv", "lifetime", "follow", "age_at", "tte",
                 "los", "时长", "生存", "存活", "随访"),
    "event": ("event", "status", "censor", "death", "dead", "fail", "relapse", "recur",
              "事件", "删失", "死亡", "复发"),
}

# TWO treatment vocabularies, deliberately not merged — they are used for opposite decisions
# and therefore need opposite error profiles.
#
#   _TREATMENT_RE       (WEAK / high recall)  — used to SKIP a candidate in resolve_outcome.
#                       A false positive here costs almost nothing: we decline to model one
#                       column as the outcome and move to the next.
#   _TREATMENT_BIND_RE  (STRONG / high precision) — used to BIND the treatment in
#                       resolve_treatment. A false positive here is a wrong causal estimand.
#
# The old docstring claimed boundary semantics kept `group` from matching age_group. That was
# FALSE: the alternation `(?:^|_|\b)` contains a literal `_`, so age_group / region_group /
# blood_group / skin_condition all matched. Harmless while the regex only skipped; once Wave
# H4d let it bind, a covariate named `age_group` became the "treatment" and PSM reported
# ATT=+3.11 (p=0.008) on data whose true ATT is -8 (H4d cold review, 2026-09-08). Hence the
# split: `group`/`condition` stay in the weak set, and only match for BINDING when the whole
# column name is exactly that word (a bare `group` column in an RCT really is the arm).
# ROLE_HINTS["treatment"] is a third, agronomic vocabulary (variety/dose/水平) — also separate.
_TREATMENT_RE = re.compile(
    r"(?:^|_|\b)(treat|treatment|treated|intervention|arm|group|condition|"
    r"exposed|exposure|policy|program|assigned|dose)(?:$|_|\b)",
    re.I,
)

# STRONG: `group`/`condition` removed (they matched age_group/skin_condition); `trt`/`tx`
# added — `trt` is the CDISC standard name and was already in ROLE_HINTS["treatment"].
_TREATMENT_BIND_RE = re.compile(
    r"(?:^|_|\b)(treat|treated|treatment|trt|intervention|arm|"
    r"exposed|exposure|assigned|assignment|policy|program|dose)(?:$|_|\b)",
    re.I,
)
# ...but a column whose ENTIRE name is one of these IS the arm in an RCT layout. These are
# WEAKER than the words above and must never outrank them (see resolve_treatment).
#   `tx` lives here, not above: as a substring it fires across the whole payments/crypto
#   domain (tx_date / tx_id / tx_amount / tx_online / tx_declined …) and bound `tx_online`
#   as the "treatment", reporting ATT=+7.4 where the truth was -15. A bare `tx` column is
#   rare, so the recall it buys is ~0.
#   (This note used to argue the point with "CDISC standardises on TRT01P/TRTA". True, but it
#   was an argument AGAINST the code as written: `trt` matched neither of them, because
#   TRT01P has a digit after the root rather than a word boundary. They are covered now by
#   _TREATMENT_BIND_CDISC_RE — cold review A MUST-FIX 4.)
#   `trt` stays a substring above — verified safe: no English word contains it at word
#   boundaries (trt / trt_group / pre_trt match; part / sort / trtn do not).
# NOT here on purpose: `control`/`control_group`. Binding a control indicator makes 1=control,
# so PSM/IPW would silently report -ATT.
_TREATMENT_BIND_EXACT_RE = re.compile(r"(group|grp|groups|condition|cond|tx)", re.I)

# NEVER bind, whatever else the name contains: a control/placebo indicator is the SAME
# contrast with the sign inverted, so binding it silently reports -ATT. `control_arm` used to
# score 2 (it contains `arm`) and IPW reported +6.683 on data whose true ATT is -8.
# This rule overrides both vocabularies above.
_TREATMENT_NEVER_RE = re.compile(
    r"(?:^|_|\b)(control|ctrl|placebo|comparator|sham)(?:$|_|\b)", re.I)

# NEGATED forms mark the ABSENCE of treatment, so binding one inverts the contrast exactly
# like a control flag does. Cold review A MUST-FIX 2 measured every one of these at
# strength 2 (`no_treatment` even reported high confidence): as the sole binary they gave
# IPW +7.652 against a truth of -8, and alongside a correct `treated` column they won on
# file order for +6.435. That `untreated` / `unexposed` happened to score 0 was luck, not
# design — there is no separator before the word root, so the strong regex missed them.
#
# `pre_`/`post_`/`baseline_` are here for a different reason than `no_`/`non_`: they mark a
# PERIOD, not an assignment. A calendar `post` dummy is exactly the column that hijacked the
# DiD estimand in the H4d cold review.
_TREATMENT_NEGATED_RE = re.compile(
    r"(?:^|_|\b)("
    r"no|non|not|never|without|un|anti|"
    r"pre|post|prior|baseline|before|after|"
    r"naive|untreated|unexposed|nontreated|nonexposed|"
    r"treatment_naive|treatment_failure"
    r")[_\s-]?(treat|treated|treatment|trt|exposed|exposure|arm|dose|intervention|"
    r"assigned|assignment|policy|program)(?:$|_|\b)",
    re.I,
)
# ...and the same idea written the other way round (`treatment_naive`, `dose_0`, `arm_0`).
_TREATMENT_NEGATED_SUFFIX_RE = re.compile(
    r"(?:^|_|\b)(treat|treated|treatment|trt|exposed|exposure|arm|dose|intervention)"
    r"[_\s-]?(naive|failure|free|none|0|zero|neg|negative)(?:$|_|\b)",
    re.I,
)

# MUST-FIX 4: the strong vocabulary above is ENGLISH-ONLY, while this module calls itself
# bilingual and ROLE_HINTS["treatment"] already carries 处理/组别/剂量/水平. Measured on a
# Chinese frame (true effect -7.84): 处理组 / 干预组 / 治疗组 / 实验组 / 是否用药 scored 0,
# the resolver fell through to `sex`, and the report said +1.385 with no warning at all.
# CDISC is the same hole in Latin script: TRT01P / TRT01PN / TRTA / TRTAN / ARMCD / ACTARM
# are THE standard treatment variables in clinical submissions, and `trt` never matched any
# of them (TRT01P has a digit after the root, not a word boundary).
_TREATMENT_BIND_I18N = (
    "处理", "干预", "治疗", "实验组", "试验组", "用药", "给药", "施肥", "剂量",
    "tratado", "tratamiento", "tratamento", "traitement", "behandelt", "behandlung",
    "trattamento", "лечение",
)
# Pinned to the ACTUAL CDISC variable names, not a loose prefix: TRT01P / TRT01PN / TRT02A
# (period-indexed planned/actual treatment), TRTA / TRTAN / TRTP / TRTPN, ARM / ARMCD /
# ACTARM / ACTARMCD. My first version was `trt\d*[apn]*`, which also swallowed `trtn` —
# a name test_treatment_binding.py has pinned at 0 since Wave H4d.
_TREATMENT_BIND_CDISC_RE = re.compile(
    r"^(trt\d{2}[apn]{1,2}|trt[ap]n?|armcd|actarm(cd)?|arm\d+)$", re.I)

# WEAK compound: `<study|case|experimental> + <group|grp|cohort>` is an arm in RCT/case-control
# layouts. Kept weak (never outranks a strong word) because "study group" can also just mean
# "the group being studied". `control` is excluded by the rule above even in compound form.
_TREATMENT_BIND_COMPOUND_RE = re.compile(
    r"(?:^|_|\b)(study|case|experimental)[_\s-]?(group|grp|cohort)(?:$|_|\b)", re.I)


def is_treatment_binding_named(name: str) -> bool:
    """True when a column NAME is a strong enough treatment signal to BIND the treatment.

    Stricter than :func:`is_treatment_named` on purpose — see the vocabulary note above.
    A false positive here picks the wrong causal estimand, so `group`/`condition` only
    count when they are the WHOLE name."""
    return treatment_name_strength(name) > 0


def treatment_name_strength(name: str) -> int:
    """2 = an unambiguous treatment word, 1 = a weak whole-name match (a bare `group` /
    `condition` / `tx` column), 0 = no signal.

    Ranking matters, not just membership: `named[0]` used to be plain file order, so on
    ``[score, group, treated, age]`` — `group` = study site, `treated` = the arm — the weak
    escape outranked the strong word and PSM reported ATT=+3.87 against a truth of -8."""
    n = str(name).strip()
    if treatment_never_named(n):
        return 0              # control / placebo / negated — binding it inverts the contrast
    if _TREATMENT_BIND_CDISC_RE.match(n):
        return 2              # CDISC standard treatment variables (TRT01PN / TRTA / ARMCD)
    if _TREATMENT_BIND_RE.search(n):
        return 2
    _low = n.lower()
    if any(h in _low for h in _TREATMENT_BIND_I18N):
        return 2              # 处理组 / 干预组 / tratamiento / behandlung …
    if _TREATMENT_BIND_EXACT_RE.fullmatch(n) or _TREATMENT_BIND_COMPOUND_RE.search(n):
        return 1
    return 0


def treatment_never_named(name: str) -> bool:
    """True when a column name marks the ABSENCE of treatment — a control/placebo flag, or a
    negated/period form (`no_treatment`, `pre_treatment`, `dose_0`, `treatment_failure`).

    Binding any of these reports the same contrast with the sign inverted, silently. This is
    a VETO, not a demotion: cold review A MUST-FIX 1 showed that scoring them 0 only removed
    them from the name-ranked tier, while the positional tiers below happily bound them when
    they were the only binary column in the frame."""
    n = str(name).strip()
    return bool(_TREATMENT_NEVER_RE.search(n)
                or _TREATMENT_NEGATED_RE.search(n)
                or _TREATMENT_NEGATED_SUFFIX_RE.search(n))


def role_hint(name: str, role: str) -> bool:
    """True if a column NAME carries the vocabulary signal for `role` (bilingual, 子串匹配)."""
    low = str(name).strip().lower()
    return any(h in low for h in ROLE_HINTS.get(role, ()))


def is_treatment_named(name: str) -> bool:
    """True when a column NAME carries a treatment/arm signal (treat/arm/exposed/dose…).

    Word-boundary match (not the substring ``role_hint``): a treatment INDICATOR is almost
    never the dependent variable, so the executor's outcome resolver skips such a column when
    falling back to the "first candidate". Consumed by executor _helpers.core / epidemiology;
    re-exported by profiler.roles for backward compatibility."""
    return bool(_TREATMENT_RE.search(str(name)))


def looks_like_survival(fp) -> bool:
    """True when the data has a survival SHAPE: a duration-named column (time/tenure/surv…) AND
    an event/censoring-named column (event/status/censor/death…). Single source for "is this
    time-to-event data", so a survival DURATION column named ``time`` is NOT read as a
    forecastable time index (dogfood: a clinical follow-up ``time`` got ARIMA-forecasted), and
    the recommender's has_survival stays in lockstep with this profiler-side guard."""
    names = [str(c.name) for c in fp.columns]
    return (any(role_hint(nm, "duration") for nm in names)
            and any(role_hint(nm, "event") for nm in names))


def survival_event_column(fp) -> str | None:
    """The single event/censoring indicator in a survival-shaped frame, else None.

    Same rule the survival family uses to pick it (an event-NAMED binary column, file order),
    so whatever the engine calls "the event" is consistent everywhere. Returns None when the
    frame is not survival-shaped — a binary called ``status`` in ordinary cross-sectional data
    is nobody's censoring indicator.

    Consumers: the treatment-effect family excludes it from AUTO covariates, because
    ``event = 1{T <= C}`` is a descendant of the duration and therefore of the treatment, so
    adjusting for it is post-treatment adjustment (measured: 33-73% attenuation of the ATT,
    and an outright failure once the event rate saturates into a constant column)."""
    if not looks_like_survival(fp):
        return None
    return next((str(c.name) for c in fp.columns
                 if c.kind == "binary" and role_hint(str(c.name), "event")), None)


def has_design_signal(fp) -> bool:
    """True when the data looks like a DESIGNED experiment (RCBD/factorial/split-plot…) rather
    than observational data that merely has categorical groups. Used to stop designed-experiment
    methods from crowding out the naive group comparison under ``--goal compare`` (发现16).

    Double gate (Wave L-C1) —防单歧义词误伤: a TREATMENT word (处理/dose/variety — low ambiguity,
    strongly experimental) is sufficient on its own; a BLOCK word alone (site/batch/区组… — also
    common in observational data) is NOT — it requires a LAYOUT of ≥2 design-vocab columns (e.g.
    block + replicate, or two blocking factors). So a single ambiguous block-named column can no
    longer flip observational data to "designed"."""
    names = [str(c.name) for c in fp.columns]
    if any(role_hint(nm, "treatment") for nm in names):
        return True
    design_cols = sum(
        1 for nm in names if role_hint(nm, "treatment") or role_hint(nm, "block")
    )
    return design_cols >= 2
