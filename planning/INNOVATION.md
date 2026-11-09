# Innovation Challenge: early plan

Started 2026-11-09. Phase 1 has priority until Nov 17. Nothing here takes time from it except the cheap
**carry-over** items in §5. Update this file as facts arrive.

---

## 1. What we know (sources only)

| fact | source |
|---|---|
| Three stages: Initial Challenge (all teams, automatic ranking) → **Innovation Challenge (top 20)** → **Final Pitch (top 5)** at Talent Arena 2027 | website text (pasted 2026-11-09) |
| Innovation: "pick analytical or tactical questions and freely explore … analysis, modeling, interpretation or insight generation. Focus on the aspects of the game you deem most relevant" | website |
| Final: a **professional jury** selects the finalists. You present the solution, your vision, and why the team "can go beyond the hack" | website |
| Organisers: Metrica Sports, FC Barcelona (Barça Innovation Hub), Mobile World Capital. Metrica works on possession "in production" | OVERVIEW.md |
| 2026's open-ended format was judged on **originality, technical solidity, use of LLMs, football relevance, presentation quality** | OVERVIEW.md (about 2026) |
| Teams with < 3 members are **not eligible** for Innovation or the Final | RULES.md, OVERVIEW.md |
| Phase 1 results announced **before 2026-11-26** | OVERVIEW.md |
| Data terms: no match data or extracts into any **cloud** LLM. Local models are fine | FAQ.md, OVERVIEW.md |

## 2. What we don't know (ask the organisers by email once Phase 1 is submitted)

1. Innovation timeline and deliverables (report? repo? demo? video?).
2. Judging criteria and weights. Is the 2026 list reused?
3. Data: same match only, more matches, or new assets (event data? corrected tracking?). Do the Phase 1 data terms carry over?
4. May we use our Phase 1 output and the public 5-min GT in Phase 2 analyses?
5. Who is on the jury (Barça analysts? Metrica product?). This decides "football relevance" vs "product" framing.
6. Are cloud LLMs allowed on **derived, aggregate** outputs (e.g., possession counts), or only local models?

**Assumption until answered:** jury-judged on roughly the 2026 criteria. Same match plus whatever we built. Local LLMs only.

## 3. Honest odds (so confidence does not turn into overconfidence)

- Reaching the top 20 is **not secured**. Our harness says v1 ≫ baseline, but the harness is 2 other-provider games
  with simulated noise, and we already found one simulator leak (frozen dead ball, 2026-11-09). We will not know our
  real rank before Nov 26. Plan Phase 2 as an option, not a certainty.
- Phase 2 work must therefore be **cheap now** (reuse of Phase 1 assets, reading, questions) and **heavy only after** the top-20 result.

## 4. Candidate directions (to be cut to one after §2 is answered)

Scored 1–5: **R**elevance to Barça/Metrica, **F**easibility with our data, **N**ovelty, **D**emo-ability, **V**alidation possible on other games.

| # | direction | what it answers | builds on | R | F | N | D | V |
|---|---|---|---|---|---|---|---|---|
| I1 | **Pressing & counter-pressing from automatic tracking**: after each loss, time-to-regain, regain location, players within 5 m of the ball in the first 5 s, PPDA-like rates | Barça 2024-25 plays a high press and high line; "how fast and where do we win it back" is the first question an analyst asks | our possession segmentation (turnovers), tracking | 5 | 4 | 3 | 4 | 5 (IDSSE events give true turnovers) |
| I2 | **Defensive-line height & offside-trap profile** per phase: line height, compactness, depth behind the line, trap events | the 2024-25 high-line identity; risks it carries | tracking only | 5 | 4 | 3 | 5 | 3 |
| I3 | **Possession value without event data**: pitch control (Spearman) + EPV/xT-style value per possession on automatic tracking | FC Barcelona's own analysts published the EPV and pitch-space work (Fernández & Bornn 2018; Fernández, Bornn & Cervone 2021). Extending it to *uncorrected automatic* data is a product question for Metrica | possession + tracking | 5 | 3 | 4 | 4 | 3 |
| I4 | **Trust layer for automatic data**: our per-frame confidence (dead-ball probability, control evidence, ball-gap context) turned into "where to trust the automatic possession / where a human should look" | Metrica sells automatic analysis; quantified uncertainty is a product feature | Phase 1 posteriors | 4 | 5 | 4 | 4 | 5 |
| I5 | **Ball recovery in tracking gaps (CV)**: re-detect the ball where the ATD lost it (41% of frames), pseudo-labelled from the ATD itself | directly improves the provider's data | Phase 1 B1 work | 4 | 3 | 4 | 4 | 4 |
| I6 | **Local-LLM match narrator / analyst assistant**: possession timeline + tactical metrics → natural-language report and Q&A, run fully locally | "use of LLMs" was a 2026 criterion; analysts want summaries | I1–I4 outputs | 3 | 4 | 3 | 5 | 2 |
| I7 | **Phase / formation states**: unsupervised team-shape states per possession phase vs Smart Tagging phases | explains *how* Barça build up vs Athletic's press | tracking + tagging | 4 | 4 | 3 | 4 | 2 |

**Current lean:** I1 (+ I2) as the football story, I4 as the technical edge, I6 as the presentation layer.
Each piece reuses Phase 1. Revisit after §2.

## 5. Carry-over: cheap things to do during Phase 1 (do not delay the Phase 1 submission)

| item | why | status |
|---|---|---|
| Keep `solution/` modular: `Match` format, features, decoder outputs | Phase 2 analyses read the same objects | in place |
| Save per-frame **posteriors** (dead probability, control evidence) next to the labels | needed for I4 and as weights in I1–I3 | todo (with the final pipeline) |
| Keep the IDSSE loader for events (`DFL_03_02_events_raw`), not only positions | true turnovers / pressing events to validate I1 on other games | todo, after Phase 1 |
| Note football observations from the harness work (e.g., dead-ball share ~42% in IDSSE) | material for the story | ongoing |
| Literature list: pitch control (Spearman 2018), EPV (Fernández et al. 2021), xT (Singh 2018), OBSO, PPDA, counter-pressing metrics | §4 directions | todo |

## 6. Guardrails that carry over

- Data stays local. No cloud LLM sees match data or extracts. Any LLM component runs a local model.
- No human-produced data about this match (event data, published stats, the known result) in any algorithm.
- Every metric is validated on other games with ground truth before it is shown to a jury.
- Every claim in a pitch comes with its uncertainty.
