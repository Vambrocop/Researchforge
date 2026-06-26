# Web UI — real-user testing checklist

> The agent can run *automated* web checks (server boots, `GET /` 200, `tests/test_web*`,
> i18n-key completeness, accessibility attributes) but **cannot do real human testing** —
> no browser, no clicking, no "does this feel right". This checklist is for **you** to run
> the human pass. Tick items, note anything off, and paste findings back; fixes become the
> next web wave. Design contract: `researchforge/web/DESIGN.md`.

## 0. Launch
```bash
py -3 -m researchforge.cli web          # then open http://127.0.0.1:8000
```
Have ready: a clean CSV, a messy CSV (mixed encodings / "$1,234" / "12%" / odd missing
markers), a tiny CSV (3 rows), a wide CSV (many columns), a non-CSV file.

## 1. Core flow (happy path)
- [ ] Upload a clean CSV → data fingerprint (rows/cols, types, panel/timeseries flags) is correct.
- [ ] Recommendations list appears with rigor lights (🟢🟡🔴) + the 6-dim scorecard bars.
- [ ] Pick a goal chip → the list re-ranks sensibly.
- [ ] Run an analysis → spinner shows, then results: summary, key numbers, ⚠ disclosures, files.
- [ ] Download the outputs zip → it opens and contains the CSV/PNG/report.
- [ ] The report's "解读（自动生成）" reads sensibly and the 关键数值 are the headline numbers.

## 2. Config overrides
- [ ] A method that exposes params renders a config form (column pickers / ints / bools).
- [ ] Override a column role (e.g. outcome) and re-run → the result reflects the override.
- [ ] Leave fields blank → it falls back to the auto defaults (no error).
- [ ] Enter a bad column / bad choice → an honest ⚠ warning, not a crash.

## 3. Bilingual (zh ⇄ EN)
- [ ] Toggle EN → every visible string switches (no leftover Chinese in the chrome).
- [ ] Toggle back to 中文 → nothing is stuck in English; state (upload/results) is preserved.
- [ ] Reload the page → the last-chosen language persists (localStorage).
- [ ] Long English method names / long identifiers wrap, not overflow.

## 4. Edge cases & honesty
- [ ] Messy CSV → it still profiles (encoding fallback, numeric coercion) and discloses coercions.
- [ ] Tiny CSV (n<10) → methods that need more data skip honestly (no fake numbers).
- [ ] Upload a non-CSV / empty file → a clear error banner, recoverable (can upload again).
- [ ] An R/optional-backend method with the backend absent → honest degrade message, no crash.
- [ ] A zero / null result is reported as such (not hidden, not fabricated).

## 5. Responsive & devices
- [ ] Desktop wide (≥1200px): centered 960px column, comfortable.
- [ ] Tablet (~768px): header + rows wrap cleanly.
- [ ] Phone (~375px): everything reachable, tap targets comfy, no horizontal scroll.
- [ ] Browsers: Chrome, Firefox, Safari/Edge — render + flow consistent.

## 6. Accessibility
- [ ] Tab through the page → focus ring visible on every control (upload, chips, toggle, run, download).
- [ ] Operate the whole core flow with keyboard only.
- [ ] Language toggle announces state (aria-pressed) to a screen reader.
- [ ] Rigor verdicts are legible without relying on color (dot **+** text).
- [ ] (Optional) Run Lighthouse / axe DevTools → note any AA contrast or ARIA flags.

## 7. Performance & feel
- [ ] First paint is quick; the run spinner makes waits feel intentional.
- [ ] No layout jump when results load; scroll position is sensible.
- [ ] Re-running / switching methods feels responsive.

## What to report back
For each issue: **where** (step/element), **what** (expected vs actual), **device/browser**,
and a screenshot if visual. Group as 🔴 blocker / 🟡 polish / 💡 idea — they become the next
web wave (UI optimization guided by `web/DESIGN.md`).
