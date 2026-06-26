# ResearchForge — DESIGN.md

> A plain-text design-system document (the [DESIGN.md](https://github.com/VoltAgent/awesome-design-md)
> convention) that an AI agent — or a human — reads to generate or evolve UI that
> stays visually consistent with ResearchForge. It captures the *existing* design
> language already implemented in `web/static/index.html` (`:root` tokens) and
> `web/static/components/tokens.css`; treat that CSS as the source of truth and this
> file as its narrated contract. **When you change a token in the CSS, update this file.**

## 1. Theme & atmosphere
Calm, credible, **academic** — a research instrument, not a marketing page. A
forest/sage-green accent on warm-neutral paper signals rigor + nature/“growth”.
Generous whitespace, restrained color, one accent. Honesty is a visual value: rigor
verdicts and ⚠ disclosures are first-class UI, never hidden. Quiet by default; color
appears only to carry meaning (rigor lights, the accent, error states).

## 2. Color palette (semantic roles)
From `:root` in `index.html` — use the CSS variables, never raw hex, in new code.

| Token | Hex | Role |
|---|---|---|
| `--bg` | `#f7f8f7` | page background (warm off-white paper) |
| `--panel` | `#fff` | card / surface |
| `--ink` | `#1b1c1a` | primary text |
| `--muted` | `#6b6f6a` | secondary text, labels, meta |
| `--line` | `#e6e8e3` | borders, dividers |
| `--accent` | `#2f6f4f` | primary action, links, focus, active state (forest green) |
| `--accent-hover` | `#27543d` | accent hover (darker) |
| `--accent-soft` | `#e8f1ec` | accent tint (selected chips, soft fills) |
| `--green` | `#2f8f5b` | 🟢 rigor light — feasible / sound |
| `--yellow` | `#c9952b` | 🟡 rigor light — feasible with caveats |
| `--red` | `#c0492f` | 🔴 rigor light — needs informed override |

Error surface: bg `#fdecea`, border `#f3c0b8`, text `#9a2b1a`. Disabled primary:
`#a9c3b5`. **Rigor light colors are reserved** — never reuse green/yellow/red for
decoration; they mean rigor.

## 3. Typography
- Stack: `-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif`
  — system fonts, with **CJK fallbacks** (the UI is bilingual zh/EN; CJK must render natively).
- Body: `15px / 1.55`, antialiased, `optimizeLegibility`.
- Section headings (`h2`): `12px`, UPPERCASE, `letter-spacing .6px`, `font-weight 700`,
  color `--muted` — quiet eyebrow labels, not loud titles.
- Brand `h1`: `18px`, `font-weight 680`, `letter-spacing .2px`, nowrap.
- Meta / captions: `13–13.5px`, color `--muted`.

## 4. Spacing, radius, elevation
- Rhythm: multiples of ~4–6px; section gap `~30px`; card padding `18px 20px`.
- Radius: `--radius:14px` (cards, panels), `--radius-sm:9px` (buttons, inputs, chips);
  pills/toggles use `999px`.
- Elevation: `--shadow` (cards: soft 2-layer), `--shadow-sm` (buttons/toggles). Keep
  shadows subtle — paper, not material.
- Focus ring: `--ring` = `0 0 0 3px rgba(47,111,79,.18)` on `:focus-visible` for ALL
  interactive elements (accessibility — never remove the outline without this ring).

## 5. Layout
- Content column: `.wrap` `max-width:960px`, centered, `padding:24px 32px 100px`.
- **Header**: sticky, `z-index:20`, translucent glass (`rgba(247,248,247,.85)` +
  `backdrop-filter: blur(10px)`), bottom hairline. Brand (h1 + tag) on the left, a
  flex `.grow` spacer, the language toggle on the right; wraps on small screens.
- Cards (`.card`) group each step: upload → fingerprint → recommendations → run → report.
- Mobile: header and rows wrap (`flex-wrap`); long identifiers use `overflow-wrap`.

## 6. Components
- **Primary button** (`.btn-primary`): accent fill, white text, `10px 22px`, weight 600;
  hover → `--accent-hover`; `:active` nudges `translateY(1px)`; disabled → muted green.
- **Language toggle** (`.lang-toggle`): pill segmented control; active segment = accent
  fill + white. The bilingual zh/EN switch is core, not optional.
- **Rigor light**: a colored dot/badge (green/yellow/red) + the verdict text — always
  paired with words, never color alone (color-blind safe).
- **Scorecard bars**: the 6-dim methodology scorecard as compact horizontal bars.
- **Spinner**: 13px ring, accent top-border, shown inline while running.
- **Error banner** (`#error`): the error surface palette; concise, dismissible by re-action.
- Reusable partials live in `web/static/components/*.html` + `tokens.css`.

## 7. Responsive behavior
- Single fluid column ≤ 960px; comfortable down to ~360px.
- Header, chip rows, and card rows wrap rather than overflow.
- Tap targets ≥ ~36px; toggle/buttons keep padding on mobile.

## 8. Accessibility guardrails
- Every interactive element has a visible `:focus-visible` ring (`--ring`).
- Toggle buttons use `type="button"`, `aria-pressed`, and bilingual `aria-label`.
- Meaning is never color-only (rigor = dot **+** text; ⚠ = symbol **+** text).
- Body text contrast meets WCAG AA on `--bg`.
- All UI strings are bilingual via `data-i18n` + `t()`; never hardcode a visible string
  outside the i18n map.

## 9. Design guardrails (do / don't)
- **DO** use the `--*` CSS variables; add a new token rather than a one-off hex.
- **DO** keep one accent; let rigor colors carry the only other “loud” signal.
- **DO** preserve every element `id` and the `fetch` request/response contract —
  `tests/test_web*` and the JS bind to them; UI polish must not rename ids or change
  payload shapes.
- **DON'T** introduce a second brand color, gradients-as-decoration, or heavy shadows.
- **DON'T** hide disclosures/caveats to make results look cleaner — honesty is the design.
- **DON'T** remove focus outlines without replacing the ring.

## 10. Agent prompt guide
> Build/evolve ResearchForge UI in a calm academic style: warm off-white paper
> (`--bg`), a single forest-green accent (`--accent #2f6f4f`), system font stack with
> CJK fallbacks, 14px card radius, soft 2-layer shadows, 960px centered column, sticky
> translucent header with a pill zh/EN language toggle. Rigor verdicts use the reserved
> green/yellow/red dots **with** text; surface ⚠ disclosures prominently. Use the
> `:root` CSS variables (not raw hex), keep every element `id` and the fetch contract
> intact, give every control a `:focus-visible` ring and bilingual `aria-label`, and
> wrap (don't overflow) on mobile. Quiet by default; color only carries meaning.
