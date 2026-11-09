# MTAH 2027 Phase 1 — Ball Possession: plan

Committed to the public repo, so aggregates only: never raw data extracts. Read at the start of every session, update the Status table at the end.
Written 2026-11-05. Deadline **2026-11-17**. Target submit **2026-11-16** (1 day buffer).
Accuracy levers, evidence and the experiment log live in **`planning/IMPROVEMENTS.md`**. Update it after every experiment.

## 1. What the challenge really is

Label every frame of the tactical video `H`/`A`/`D`/`X` (199,319 rows). Ranked on **spellF1**, ties on macroF1.
It is **not mainly a CV problem**: tracking, tagging and homography are given. The core is time-series
segmentation of noisy tracking (ball missing 41% of graded frames). CV/audio are optional levers to
fill those gaps and to detect dead balls.

## 2. Cheat sheet (so the docs need not be re-read)

- 25 fps. Graded: 1st half video frames 11185–90651, 2nd half 110808–190197 (158,857 frames).
  2nd half is graded from 110808 but kicks off at **113742** → 110808–113741 is all `D`.
- ATD raw line: `frame_id:slot;slot;...(46):ball`, each `x,y,speed`, `NaN` when missing.
  `video_frame = frame_id - 1`. Coords normalised 0–1; pitch 105.3 × 68.0 m (games.py).
  Slot team = XML `Player teamId`; first `Team` in XML is home (Barcelona). Slots are tracklets, not people.
  Team attack direction per half is a provider parameter in the ATD XML (read it in code).
- No ball-alive flag and no ball-owner field in the ATD. GT came from *corrected* tracking's
  dead flag + owner field, mechanically. We must infer both.
- Smart Tagging: SportsCode XML, `frame = round(seconds*25)`. 638 phases. `Team` = Barcelona / Atletic Club / N/A
  (all throw-ins are N/A). `SET PIECES` has `Type` (throw-in, corner, goal kick, free kick, kick-off). `GOALS` x4.
  Phases ≠ possessions. Loader: `starter/load_smart_tagging.py` (stdlib).
- Homography: normalised pitch → pixels, valid on 100% of tactical frames. `starter/load_homography.py`.
- Both videos have AAC audio (ffprobe). Broadcast is not frame-synced to tactical.
- Public GT (`evaluation/gt/...public.csv`): frames 11185–18684, 17 possessions, mean 14.2 s,
  21 segments (so ~4 dead-ball stretches, ~12 live turnovers in 5 min). Roughly 3.4 possessions/min.
- Floor: `baseline_smooth50` spellF1 **0.2433** (full match). Noise band on spellF1 ≈ 0.04.
- Machine: RTX 4050 6 GB, Python 3.10.11, ffmpeg at `C:/ffmpeg/bin`.

## 3. How spellF1 behaves (drives every design choice)

- A spell = maximal run of `H` or `A`. **`D` splits spells.** Matched one-to-one, same team, by temporal
  IoU, F1 averaged over IoU 0.3/0.4/0.5/0.6/0.7.
- `X` is forward-filled before spells are built → **`X` can never create a `D` gap**. Dead balls must be written as `D`.
- Missing a dead ball merges two GT spells → 2 misses + 1 spurious. A fake dead ball or a flicker
  splits one spell → 1 miss + 2 spurious. Both are expensive; the spell *count* must be right.
- Timing: 1 frame late is free, 8 frames ≈ 0.89, 1 s late ≈ 0.65. Short possessions are the fragile ones.
- So the two problems are equally important: **(a) alive/dead segmentation, (b) turnover timing**.
- Writing `X` through a ball-gap inside a possession is free; never `X` across a turnover.

## 4. Rules that bind the design

- No hand labels on this match, for any purpose. No frame-specific constants or rules for this match.
- Public 5 min = pipeline sanity check only. Never tune or choose between versions on it.
- No human-made data about this match (event data, corrected tracking). Other games: anything goes.
- Repo must regenerate the exact file (determinism: fixed seeds, no nondeterministic GPU ops in the final path).
- **No data extracts to cloud LLMs** (incl. Claude): Claude writes code; scripts print aggregates only.
- Teams must be 3–5 people.

## 5. Skills needed

People:

| skill | used for |
|---|---|
| Python / numpy / pandas, fast parsing + caching | ATD → arrays, feature tables |
| Football knowledge (restarts, possession rules) | dead-ball rules, what "controlled" means |
| Time-series / probabilistic modelling (HMM, HSMM, Viterbi, hysteresis, Kalman) | the decoder, ball-gap filling |
| Sports analytics (nearest-player control, pass detection from ball kinematics, formations) | per-frame evidence |
| Evaluation discipline (spellF1, bootstrap, held-out other games) | tuning without the public window |
| Audio DSP (STFT, band energy, peak picking) | referee whistle detection (stretch) |
| CV (YOLO inference, homography back-projection) | re-detect the ball on missing frames (stretch) |
| Reproducibility + technical writing | repo, pinned deps, the technical document |

Claude skills per phase:

| phase | skill |
|---|---|
| decoder, features, validation design | `engineering-skills:senior-data-scientist`, `engineering-skills:senior-ml-engineer` |
| ball re-detection (stretch) | `engineering-skills:senior-computer-vision` |
| diagnostic timeline plots | `dataviz` |
| score stuck / regressions | `debugging-strategies` |
| before freezing | `/simplify`, then `/code-review` |
| technical document | `doc-coauthoring` (+ `excalidraw-diagram` for one pipeline figure) |
| NOT relevant | `sports-vision-minimap` (different course) |

## 6. Architecture (target)

```
solution/
  io_atd.py     parse ATD once -> cache .npz (players [T,46,2], speed [T,46], team [46], ball [T,2])
  tagging.py    Smart Tagging -> frame intervals; set-piece restart frames
  ball.py       ball gap filling: short-gap interpolation, kinematic (Kalman) bridge; optional CV fill
  deadball.py   alive/dead evidence + segmentation
  control.py    per-frame team-control evidence
  decode.py     HSMM/Viterbi over {H, A, D}
  audio.py      whistle detector (stretch)
  run.py        one command: data/ -> evaluation/results/<team>.csv, then validate
validation/     other-game harness (Metrica open sample data etc.)
```

Dead-ball evidence: ball leaves pitch bounds / vanishes at a touchline or goal line; ball stationary
at a restart spot (corner arc, goal area, centre spot, touchline); collective stillness (mean player
speed low); kick-off formation (both teams in own halves); `SET PIECES` start ≈ restart; `GOALS` →
dead until kick-off; pre-kickoff 2nd-half stretch from the published half boundaries; whistles (stretch).

Control evidence: nearest player per team with distance, ball speed (low near a player = control),
dwell time, pass detection (ball speed spike, leaves a player of team X, arrives at team X) → a
same-team pass confirms, an opposing receive + dwell flips. Loose balls stay with the last team
(hysteresis). Smart Tagging team phases as a soft prior.

Decoder: 3-state HSMM / Viterbi with duration priors and a switching penalty, so flicker cannot
create spells. Dead-ball segments from deadball.py are hard-ish constraints.

## 7. Phases

Each phase = one focused Claude session. Start a session with: "read planning/PLAN.md, do phase N".

| # | phase | deliverable | done when | Claude cost |
|---|---|---|---|---|
| 0 | Setup (team, no Claude) | own public repo, venv, pinned deps, baseline + scorer reproduce | `run_evaluation.py` matches README board | none |
| 1 | Data layer | `io_atd.py`, `tagging.py`, cache, aggregate diagnostics script | loads in < 30 s from cache; prints gap-length histogram | low |
| 2 | Dead-ball v1 + first real submission | `deadball.py`, `run.py`, rule-based control + smoothing | validated file, sanity-checked on public window | medium |
| 3 | Validation harness on other games | open other-game tracking+events (verify availability) → possession labels from events → degrade tracking to look like ATD (drop ball ~41% in bursts, fragment tracks, strip IDs) → run pipeline + spellF1 | numbers on ≥ 2 other games | medium |
| 4 | Decoder + control model | `control.py`, `decode.py` (HSMM), tune params on the harness only | beats phase 2 on the harness beyond noise | high |
| 5 | Gap classification + ball-free possession (re-scoped 11-09) | per-gap dead/live classifier (S1), team-shape possession (P3); interpolation dropped (0.3% of missing frames are in gaps ≤ 5 s) | harness gain | medium |
| 6 | Stretch: audio whistles | `audio.py` (ffmpeg → wav locally, band energy 2.5–4.5 kHz, tonal peaks) fused into dead-ball | harness can't test audio → check agreement with set-piece tags | medium |
| 7 | CV ball recovery (promoted 11-09) | B1-spike → self-train detector on ATD-ball pseudo-labels (B1a) → run in gaps (B1b); or broadcast-ATD fusion (B2). Pick by spike numbers | GPU run ≈ 30–60 min; fill rate reported | high |
| 8 | Freeze + deliverables | determinism check (2 runs → identical hash), README, tech doc, `<team>.csv` / `<team>.pdf` | validator passes on the exact file uploaded | medium |

## 8. Calendar

| date | work |
|---|---|
| Nov 5 | planning (done) |
| Nov 6 | phase 0 + 1 |
| Nov 7 | phase 2 → first valid submission uploaded early to the form (editable later) |
| Nov 8–9 | phase 3 + 4 |
| Nov 10–11 | phase 4 tuning + 5 |
| Nov 12–13 | phase 6 or 7, only if ahead |
| Nov 14 | feature freeze |
| Nov 15 | phase 8: repro check + tech doc |
| Nov 16 | final submission. Nov 17 = buffer |

### Re-plan 2026-11-09 (3 days behind; nothing built before today)

| date | work |
|---|---|
| Mon Nov 9 | phase 0 (done) · E1 availability check · phase 1 cache · phase 2 first submission (S2+S4+P1+R1), upload early |
| Tue Nov 10 | phase 3 harness: E1 + long-gap degradation E2 + E3 |
| Wed Nov 11 | phase 4: S1 + D1 + T1 on harness. Teammate: B1-spike + B2 sizing on GPU |
| Thu Nov 12 | phase 4 tuning (D5, D6) + phase 5 (P3) |
| Fri Nov 13 | phase 7 B1 or B2, whichever the spikes favour; A1 only if ahead |
| Sat Nov 14 | feature freeze |
| Sun Nov 15 | phase 8: repro check + tech doc |
| Mon Nov 16 | final submission. Tue Nov 17 = buffer |

## 9. Saving Claude budget

- One phase per session; this file carries context instead of re-reading the docs.
- Scripts print short aggregate output (also required by the data rule).
- Teammates do non-Claude work: watching the video to understand restarts, drafting the doc,
  running long GPU/audio jobs, fetching other-game datasets.
- Skip phases 6–7 if phase 4 is late. A good decoder beats a half-done detector.

## 10. Risks

- Overfitting to the public window → only the harness picks between versions.
- Domain gap: other-game data is clean tracking; the degradation step in phase 3 must mimic ATD noise.
- Ball-in-air back-projection is wrong (homography = ground plane) → only use CV fills at low ball speed / near players.
- Non-determinism (GPU, dict order, float sums) → final path CPU + sorted + seeded.
- Data accidentally committed → `data/` is ignored; never `git add -f`.

## 11. Open questions

- Team members and roles (3–5 required)?
- Ask organisers whether sending *aggregate scores* from our scripts to an LLM is fine (we assume yes).
- Which other-game datasets are actually downloadable (Metrica sample-data, DFL/IDSSE open set)? Verify in phase 3.

## Status

| phase | state | notes |
|---|---|---|
| 0 | done 11-09 | `.venv` (py 3.10, numpy 2.2.6, pandas 2.3.3, kloppy 3.18.0, scipy, pytest); system python has no pandas. 25 tests in `solution/tests` pass (metric contract, data contract, bit-exact baseline regen). Scorer reproduces public board. data/README.md note was stale |
| 1 | mostly done 11-09 | `solution/io_atd.py` → `data/cache/` npz (13 s parse, ~1 s load), `solution/match.py` common format. `tagging.py` still to do |
| 2 | todo | |
| 3 | done 11-09 | 2 IDSSE games × 3 degrade seeds; ranks baselines like the official full match. See IMPROVEMENTS.md E1–E3, exp 001–004 |
| 4 | todo | |
| 5 | todo | |
| 6 | todo | |
| 7 | todo | |
| 8 | todo | |
