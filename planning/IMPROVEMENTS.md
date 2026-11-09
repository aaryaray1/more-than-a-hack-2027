# Accuracy improvement register

This file tracks every idea for raising **spellF1**. Started 2026-11-09. Update it after every experiment,
including the ones that fail. `PLAN.md` holds phases and dates. This file holds the levers and the evidence for each.

Status values: `idea` → `spike` (≤ 2 h feasibility check) → `building` → `testing` → `accepted` / `rejected` / `parked`.

---

## 1. How a change gets accepted (decided before any results)

| | rule |
|---|---|
| primary metric | spellF1 from the official `evaluation/possession_metrics.py`, mean over the other-game harness (E1) |
| accept | Δ mean ≥ **+0.02**, positive on ≥ 2/3 of harness games, and the game-clustered bootstrap 90% CI of Δ excludes 0 |
| otherwise | `parked` with the numbers, never silently dropped |
| secondary (reported, not decisive) | macroF1, alive/dead F1, team acc, possession-count error, transF1 @12/@25, latency |
| BAR-ATH label-free checks | possession count, mean duration, D share, spells < 2 s, D→alive agreement with set-piece tags. These catch domain gap. They are not a score. |
| public 5 min | run **once per accepted version**, written in its own column, **never** used to decide. Measured noise band there: **±0.33** |
| hygiene | one change per experiment; seeds fixed; log the git hash |

Why paired deltas: the full-match noise band on absolute spellF1 is ~0.04. A paired Δ on the same games is
tighter, so it is the number that can actually move a decision.

---

## 2. Current best

BAR-ATH label-free numbers are over the two halves only. The validator counts the whole file, which gives 669 / 7.6 s.

| version | date | harness spellF1 | BAR-ATH poss / mean dur | public (sanity only) | full match (published) |
|---|---|---|---|---|---|
| `baseline50` (= `baseline_smooth50`) | 2026-11-09 | **0.2925** (raw labels 0.2504) | 594 / 7.2 s / D 32.3% | 0.3256 | **0.2433** ← the floor |

---

## 3. What the data says (aggregate-only runs, 2026-11-09)

Script: one streaming pass over the tactical ATD, graded frames only, printing aggregates. It runs in 4 s.

| fact | number | what it means for the plan |
|---|---|---|
| ball present | 59.3% of graded frames | matches DATA.md |
| ball gaps | **150 gaps**, p50 **268 f (10.7 s)**, p90 845 f, max 5537 f (3.7 min) | gaps are long blocks, not flicker |
| missing frames in gaps ≤ 5 s | **0.3%** | short-gap interpolation / Kalman bridge (old phase 5) is worth ~nothing |
| missing frames in gaps ≤ 30 s | 56% | the other 44% sit in gaps longer than 30 s |
| ball-present stretches | 149, p50 13.8 s, p90 62 s | the tracker holds the ball for long runs, then loses it for long runs |
| ball speed (from positions) | p50 5.1, p99 24.6 m/s, 0.23% jumps > 40 m/s | the ball track is clean when present. The `speed` field is NaN 92% of the time, so compute speed from positions |
| ball shift across a gap | p50 10 m, p90 34 m | a gap is not "ball stayed put" |
| present stretches ending ≤ ~2 m from a line | **28.9%** | the ball goes out, the tracker drops it, and that gap is a dead ball |
| ball visible and outside the lines | 4.2% of present graded frames | a direct D cue |
| mean player speed, ball present vs missing | **2.09 vs 1.13 m/s** | gaps are rich in stoppages. The core task is classifying each gap |
| set-piece tag windows with ball missing | 33.6% of their frames | tags help exactly where the ball is gone |
| team-tagged phases cover | 77.8% of graded frames, 73% of ball-missing graded frames | a usable soft team prior inside gaps |
| tracked objects | 20.5/frame, < 18 on 9.2%, > 22 on 2.4% | duplicates exist (Q2) |
| slots | 24 home + 22 away; 43 used; 1218 tracklets, p50 35 s | tracklets, not players. Per-player features need linking (Q3) |
| nearest player to ball | p50 1.9 m; ≤ 1.5 m on 41.8%; both teams ≤ 3 m on 13.9% | contested frames ~14%: the hysteresis zone |
| baseline_smooth50 | 669 possessions, mean 7.6 s | over-fragmented (football runs ~3–4 possessions/min including stoppage splits; take the target from the harness, not the public window) |

**Headline:** the problem is mostly about 150 ball gaps. Each one has to be labelled dead, H-live or A-live,
and the turnovers inside it have to be found. Ball re-detection (B1/B2) and gap-context dead-ball detection (S1)
move up. Interpolation moves out.

**Open sizing question:** what share of gap frames is dead? Estimate it without labels (player-speed
mixture, restart geometry), not from the public window.

---

## 4. Register

Priority: **P0** = on the critical path to a good submission · **P1** = big expected gain, real cost · **P2** = do if ahead.

### E: evaluation (nothing else can be accepted without it)

| ID | lever | why / evidence | cost | needs | measured by | pri | status |
|---|---|---|---|---|---|---|---|
| E0 | test suite `solution/tests` (32 tests: metric contract, data contract, bit-exact baseline regen, ported baseline = official files, control labels, IDSSE shapes, degrader determinism) | locks in the spellF1 facts the design relies on | done | — | `cd solution && ../.venv/Scripts/python -m pytest` | P0 | **accepted** |
| E1 | other-game harness: IDSSE J03WMX (Köln–Bayern) + J03WPY (Düsseldorf–Nürnberg) from the HF mirror → `validation/idsse.py` (referees dropped). Per-frame BallStatus + BallPossession confirmed. Labels: **control** (primary: a switch needs ≥ 2 s, CHALLENGE.md's "keeps the ball for a few seconds"; shorter flips stay with the last team; `validation/labels.py`) and **raw** DFL (secondary) | the only legal tuning signal. Raw DFL flips on any touch: 37% of spells < 2 s | done | — | baseline ranking matches the official full match (§5) | P0 | **accepted** |
| E2 | `validation/degrade.py`: ball lost/found by a hazard process (more loss when dead / airborne), fitted by `calibrate_degrade.py` to §3 (loss 2.3 → 0.10); players cut into tracklets reusing 23 slots/team, absence bursts, 2.4% duplicates, per-tracklet offsets; no ball z, no speed field | without it the harness rewards ball-dependent methods | done | E1 | fit gaps: missing 46–49% (target 41), speed ratio 0.61–0.64 (0.54), line-end 34–38% (29): harness gaps are a little less dead-heavy than ours | P0 | **accepted v1**; refit if S1 turns out sensitive |
| E3 | `validation/harness.py <pipeline> [--vs other] [--log NOTE]`: 2 games × 3 degrade seeds, official scorer, paired possession bootstrap of Δ (90% CI), BAR-ATH label-free line, appends to `planning/experiments.csv` | makes §1 mechanical | done | E1 E2 | — | P0 | **accepted** |
| E4 | BAR-ATH label-free checks: possessions, per min, mean length, D share, spells < 2 s (in `harness.py`). Still to add: D→alive agreement with set-piece tags | catches harness/match domain gap | S | — | — | P0 | **partly done** |

### S: alive/dead segmentation

| ID | lever | why / evidence | cost | needs | measured by | pri | status |
|---|---|---|---|---|---|---|---|
| S1 | **gap-context dead-ball classifier**: per gap, use the last-seen ball near a line, team-mean speed, gap length, re-acquisition spot (corner arc, 6-yd box, touchline, centre), restart formation, set-piece tag overlap. Rules first, then GBM on the harness | 28.9% of stretches end at a line; players 1.13 vs 2.09 m/s in gaps | M | E1–E3 | alive/dead F1 + spellF1 on harness | P0 | idea |
| S2 | ball visible outside the lines → D (with margin + hysteresis) | 4.2% of present frames | S | — | harness | P0 | idea |
| S3 | set-piece tags as restart anchors: the tag marks the restart, then search backwards for the D start | 59 tags, all restarts | S | S1 | harness can't test tags → BAR-ATH agreement | P1 | idea |
| S4 | kickoff detector (players in own halves + ball on centre spot + first move) for both halves, goals (4 tags) → D until kickoff | 2nd half: 2,934 graded D frames before the kickoff; detect it, don't hard-code it | S | — | kickoff found = DATA.md frames ± 25 | P0 | idea |
| S5 | collective slowdown / standing-still detector (fouls, injuries, VAR) | stoppages that aren't ball-out | S | E1 | harness | P1 | idea |
| S6 | = A1 whistle | | | | | | |

### P: who has it

| ID | lever | why / evidence | cost | needs | measured by | pri | status |
|---|---|---|---|---|---|---|---|
| P1 | nearest-player control **with hysteresis**: switch only after the other team shows control for k frames or a completed pass. Replaces majority vote | "loose ball stays with last team" (CHALLENGE.md); contested zone ~14% of frames | S | E1 | harness | P0 | idea |
| P2 | control features: distance, ball/player co-movement (dribble), dwell time, pass detection (speed jump + direction change), same-team pass chains | "controlled" proxies in CHALLENGE.md | M | P1 | harness | P1 | idea |
| P3 | **ball-free team possession from shape**: width/depth/compactness, centroid velocity, line heights, who is advancing. Trained on harness games | the only team signal on gap frames that are live | M | E1 E2 | team acc on degraded gaps | P1 | idea |
| P4 | Smart Tagging team prior as a soft emission (never a label; phases ≠ possessions) | covers 73% of ball-missing frames | S | D1 | BAR-ATH only (harness has no tags) | P1 | idea |
| P5 | restart-team inference: which team's player is at the restart spot when play resumes. Also resolves the N/A throw-ins | first possession after every D | S | S1 | harness | P1 | idea |

### B: getting the ball back (the CV core)

| ID | lever | why / evidence | cost | needs | measured by | pri | status |
|---|---|---|---|---|---|---|---|
| B1-spike | run a pretrained small-object ball detector on sampled tactical frames **where the ATD has the ball**, project the ATD ball through the homography, report recall@10 px and pixel error (aggregates only). Candidates: WASB-SBDT / TrackNetV3-style heatmap nets (3-frame input, built for tiny balls), YOLO11 + SAHI 640-px tiles | decides whether B1 is worth a day; label-free evaluation on this match | S (2 h, GPU) | torch, ultralytics | recall ≥ 0.6 → go | P1 | idea |
| B1a | **self-training on this match**: ATD ball (94k graded frames) → homography → pixel boxes as pseudo-labels; fine-tune the detector; hold out time blocks. RULES allow "pseudo-labels produced by an algorithm" | removes the broadcast→tactical domain gap. Check first that ATD xy re-projects onto the ball (if ATD came from the same homography, airborne balls re-project correctly too) | M | B1-spike | held-out recall / px error | P1 | idea |
| B1b | run in gaps only: search window = last-seen ball ± R and the player cluster, temporal-consistency gating, reject implausible ground-plane speeds (> 35 m/s ⇒ airborne), back-project with the inverse homography | 41% of graded frames are gaps; even 30% fill rate is ~12% of the match | M (GPU ≈ 30–60 min for ~65k gap frames) | B1a | fill rate + harness-free: agreement with S1/P3 | P1 | idea |
| B2 | **broadcast fusion**: sync the broadcast ATD to the tactical ATD algorithmically (both in pitch coords: match player configurations in sliding windows; replays show up as non-monotone matches; audio cross-correlation as a second check), then use the broadcast ball where the tactical one is missing | a different camera with different occlusions; 50.5% ball-present | M–H | — | sizing first: % of tactical gap frames covered after sync | P1 | idea: **size it first** (1 h) |
| B3 | ball-location imputation from player configuration (set transformer / GBM trained on the harness) | coarse ball position in gaps for P1/P2 | M | E1 E2 | median error on degraded harness gaps | P2 | idea |

### V: other CV

| ID | lever | why / evidence | cost | needs | measured by | pri | status |
|---|---|---|---|---|---|---|---|
| V1 | jersey-colour check per tracklet (crop at projected position, colour clustering) → catch mis-teamed tracklets, GK/referee confusion | a wrong team on one tracklet poisons nearest-player for its whole life | M | Q1 first (free) | % tracklets whose colour disagrees with XML team | P2 | idea |
| V2 | throw-in / GK-hands detection (pose: arms above head at touchline) → restart type + restart team | 26 throw-ins, all team N/A | M | — | agreement with throw-in tags | P2 | idea |
| V3 | SoccerNet action spotting on the broadcast video (out-of-play, throw-in, corner, kick-off) mapped back via B2 | strong pretrained models exist for the broadcast domain | H | B2 | agreement with set-piece tags | P2 | parked until B2 |
| V4 | zero-shot CLIP/SigLIP "stoppage vs play" frame classifier | cheap to try, probably weak on a wide view | S | — | agreement with S1 | P2 | idea |

### T: transition timing (1 frame late is free, 8 ≈ 0.89, 1 s ≈ 0.65)

| ID | lever | why / evidence | cost | needs | measured by | pri | status |
|---|---|---|---|---|---|---|---|
| T1 | snap every decoded turnover to the first frame the new team's player is within r of the ball | decoders lag by design | S | D1 | latency + spellF1 on harness | P0 | idea |
| T2 | D boundaries: D starts at the ball-crosses-line frame, ends at the restart touch | | S | S1 | harness | P1 | idea |
| T3 | systematic lag correction measured on the harness | | S | E3 | latency mean → 0 | P2 | idea |

### D: decoder

| ID | lever | why / evidence | cost | needs | measured by | pri | status |
|---|---|---|---|---|---|---|---|
| D1 | HSMM/Viterbi over {H, A, D}: duration priors from harness games, switch penalties, min durations | the spell count must be right; smoothing alone gives 669 spells | M | E1 | harness | P0 | idea |
| D2 | richer states: D-out / D-foul / restart, plus a "contested" sub-state that keeps the label | encodes restart structure and loose balls | M | D1 | harness | P1 | idea |
| D3 | per-frame LightGBM on harness features → posteriors → D1 | learned emissions | M | E1 E2 | harness | P1 | idea |
| D4 | BiGRU/TCN sequence model | only if D3 plateaus | H | D3 | harness | P2 | idea |
| D5 | seeded black-box tuning (Optuna) of decoder params on harness spellF1 | | S | E3 | harness | P1 | idea |
| D6 | post-rules: drop spells < k s without strong evidence; merge same-team spells split by a short D without dead evidence | each fake split costs 1 miss + 2 spurious | S | D1 | harness | P1 | idea |

### Q: tracking clean-up

| ID | lever | why / evidence | cost | needs | measured by | pri | status |
|---|---|---|---|---|---|---|---|
| Q1 | behavioural team-assignment check: each tracklet's mean x per half vs that team's attack direction; GK tracklets | free, aggregate-only | S | — | count of suspicious tracklets | P1 | idea |
| Q2 | duplicate suppression: same-team detections < 0.5 m apart | > 22 objects on 2.4% of frames | S | — | objects/frame | P2 | idea |
| Q3 | tracklet linking (1218 tracklets) | only needed for per-player pass chains | M | P2 | — | P2 | idea |

### A: audio

| ID | lever | why / evidence | cost | needs | measured by | pri | status |
|---|---|---|---|---|---|---|---|
| A1 | whistle detector on the tactical audio (2.5–4.5 kHz tonal peaks) → D starts and restarts | the referee defines "in play" | M | ffmpeg → wav | agreement with set-piece tag starts | P2 | idea |

### R: reproducibility and submission

| ID | lever | why / evidence | cost | needs | measured by | pri | status |
|---|---|---|---|---|---|---|---|
| R1 | write the CSV with `newline="\n"` explicitly | `core.autocrlf=true` here and `open(..., "w")` on Windows writes CRLF; "the same file" must hold on any OS | S | — | 2 runs, same SHA-256 | P0 | idea |
| R2 | pinned `requirements-solution.txt`, fixed seeds, CPU-only final path, sorted iteration | RULES §3 | S | — | 2-run hash check | P0 | idea |
| R3 | never write X on graded frames | X = forward-fill on spellF1 and a miss on macroF1, so writing the filled label ourselves is never worse | S | — | — | P0 | **accepted** (by argument) |
| R4 | `<team>.csv` / `<team>.pdf`, validator run on the exact uploaded file | SUBMISSION.md | S | — | — | P0 | idea |

---

## 5. Experiment log (append-only, newest last)

| date | exp | IDs | change | harness spellF1 [90% CI] | Δ vs best | macroF1 | poss err | BAR-ATH poss / dur / D% | public (sanity) | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-09 | 000 | E0 | test suite + env (`.venv`) | — | — | — | — | 669 / 7.6 s / 35.9% (baseline_smooth50) | 0.3256 [±0.33] | accepted (infra) |
| 2026-11-09 | 001 | E1–E3 | baseline_raw on harness | 0.0504 | — | 0.6271 | +1281 | 1949 / 1.9 s / 40.7% | 0.0378 | starting point |
| 2026-11-09 | 002 | E1–E3 | baseline25 on harness | 0.2306 | — | 0.6659 | +296 | 746 / 5.4 s / 36.5% | 0.2868 | starting point |
| 2026-11-09 | 003 | E1–E3 | **baseline50** on harness | **0.2925** | — | 0.6809 | +183 | 594 / 7.2 s / 32.3% | 0.3256 | **current best** |
| 2026-11-09 | 004 | E1–E3 | baseline125 on harness | 0.2814 | — | 0.6688 | +62 | 398 / 12.4 s / 22.3% | 0.4242 | starting point |

---

**Harness check (001–004):** the harness ranks the baselines raw < 25 < 125 < 50, exactly like the official full match (0.049 / 0.200 / 0.224 / 0.243). The public window ranks 125 first. Harness spellF1 reads ~0.04 high in absolute terms, so compare deltas, not levels. Run-to-run spread for one pipeline is ±0.03, so trust the paired Δ CI, not one run.

## 6. Parked / rejected (keep the reason)

| ID | idea | why |
|---|---|---|
| old phase 5 | short-gap interpolation + Kalman bridge as a main lever | 0.3% of missing frames are in gaps ≤ 5 s (§3). Keep a trivial version only for gaps ≤ 1 s |
| — | write X through uncertain stretches | see R3 |
| — | tune anything on the public window, or use its counts or durations as priors | rule; and its noise band is ±0.33 |

---

## 7. Next up (in order)

1. **Phase 2 v1 pipeline** on the harness, one lever at a time, each `--vs baseline50 --log`: P1 hysteresis control → S2 ball-out-of-lines D → S1 gap-context dead-ball rules → S4 kickoff. Then write the first BAR-ATH submission (R1 LF, R4 validator) and upload early.
2. **D1 + T1** decoder and turnover snapping on the harness.
3. In parallel, a teammate with the GPU runs the **B1-spike** (2 h) and the **B2 sizing** (1 h). Their numbers decide whether day 4–5 goes to B1, B2 or P3.
4. Optional: 2 more IDSSE matches (J03WN1, J03WOH) if paired Δs keep landing near the +0.02 bar.
