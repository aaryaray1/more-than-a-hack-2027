# Ball possession from automatic tracking: technical document

More than a Hack 2027, Phase 1. Team **中華民國海軍** (Republic of China Navy, ROCN). Pipeline version `v4`.

## 1. 方法 | Approach

> A learned model first separates dead ball from live play, a second one names the team in possession, and a third marks the moments a turnover is likely. Each is decoded into segments, so the number of possessions comes out right instead of flickering frame by frame.

<!-- figure:pipeline -->

The ball is missing on 41% of graded frames, and it is missing in long blocks (150 gaps, median
10.7 s), not in flicker. Only 0.3% of the missing frames sit in gaps of 5 s or less, so
interpolating the ball is worth almost nothing. The task is therefore time-series segmentation:
decide for every stretch whether the ball is dead or alive, and decide which team controls it
while it is alive.

No label from the provided match is used anywhere. All models are trained on seven open matches
from another league, whose clean tracking is first degraded to look like the provided tracking
(section 4). Every design choice was made on those matches with the official scorer (section 6).

The pipeline has five steps. All of them read the tactical tracking (ATD). One rule also reads the
Smart Tagging.

1. **Dead or alive, learned.** A gradient-boosted classifier (scikit-learn
   `HistGradientBoostingClassifier`, 200 trees) gives each frame a probability that the ball is
   dead. It uses 55 features per frame. About the ball: distance to the nearest line, corner and
   centre spot; speed; distance to the nearest home and away player; time since it was last seen
   and until it is next seen; gap length; where it was last seen and where it reappears; how far
   it moved across the gap; players within 5 m and 10 m. About the players only, so they work
   while the ball is hidden: mean speed over 1 s, 4 s and 10 s windows and at eight offsets out to
   8 s either side; share of players moving, walking and running fast; share in their own half;
   how close the nearest player stands to the edge of the pitch and how many are outside it; how
   spread out each team is and whether that is changing.
2. **Dead or alive, decoded.** A two-state Viterbi decoder turns the probabilities into segments,
   with a fixed penalty for every switch. This stops flicker from creating fake stoppages, each of
   which would split one possession into two.
3. **Kickoff rule.** Episodes of at least 2 s where 90% or more of the tracked players stand in
   their own half are dead ball. The start of each half is dead until the first
   `Kick Off (Start)` tag in the Smart Tagging.
4. **Team in possession, learned.** A second gradient-boosted classifier (300 trees) gives each
   alive frame a probability that the home team has the ball. It uses 106 features per frame
   (section 2). Every training frame is used twice, once as it is and once with home and away
   swapped, and at prediction time the two views are averaged, so the model cannot prefer a side.
5. **Team in possession, decoded.** A second Viterbi decoder runs inside each alive segment. Its
   switch penalty is not fixed: a third classifier (200 trees, same features as the team model)
   gives each frame the probability that possession changes within a quarter of a second, and
   switching is cheap where that probability is high and expensive elsewhere. A possession shorter
   than 2 s that interrupts the other team's possession is absorbed back, following the
   challenge's "keeps the ball for a few seconds".

**Why this and not something else.** The ranking metric counts whole possessions, so the number of
segments has to be right before the timing matters. A per-frame classifier with smoothing (the
baseline) reports about twice as many possessions as a football match has. A decoder with a switch
penalty fixes the count directly.

**How we found where to work.** We replaced each half of an earlier rule-based version with the
truth, one at a time, on the other-game matches. With perfect dead-ball detection and our own team
rule the score was 0.65. With our own dead-ball detection and perfect team labels it was 0.89. The
team step was the bottleneck, so that is where the learned model went.

## 2. 球隊模型特徵 | The team model's features

> The team model reads 106 features: how far each side's nearest player is from the ball, how hard the ball was struck, and which way each team is moving, each looked at a few seconds back and ahead.

All features are computed for both teams, so swapping the teams is a column permutation.

| group | features |
|---|---|
| distance stacks | distance from the ball to the team's nearest player at 11 time offsets from 2 s before to 2 s after; the home minus away difference at the same offsets |
| windowed minima | closest approach of the team to the ball in three past and three future windows, out to 5 s |
| control | share of frames in which the team's player is within 1 m of the ball and at least 0.3 m closer than any opponent, in two past and two future windows; time since and until the last such frame; the same for a 2 m radius |
| kick strength | change of the ball's velocity over half a second, counted for a team when its player is the nearest and within 2.5 m, in five windows |
| co-movement | speed of the ball relative to the team's nearest player |
| team advance | mean velocity of the team's players along its own attack direction, now, 3 s earlier and 3 s later; depth of the team's centroid |
| ball along the pitch | ball position and ball velocity along the team's attack direction |
| gap context | whether the ball is visible, time since and until it is seen, how far it moved along the pitch across the gap |

The attack direction is not read from metadata. In each half the team whose players stand at
lower x on average defends that end.

The future-looking windows are what let the model apply the "few seconds" rule: a touch only
becomes a change of possession when the new team is still on the ball afterwards.

## 3. 使用的資料 | Assets used

> Every feature comes from the tactical-camera tracking. Smart Tagging is used for the pre-kickoff dead stretch and for label-free checks only.

| asset | used | how |
|---|---|---|
| tactical tracking (ATD) | yes | every feature of every model |
| Smart Tagging | one tag type in the output | `Kick Off (Start)` marks the end of the pre-kickoff dead stretch in each half |
| Smart Tagging, other tags | checks only | set-piece tags and team phases are compared with our output as a label-free check (section 6); they do not change it |
| homography | no | not needed without video features |
| tactical and broadcast video | no | ball re-detection in the gaps was planned and not built |
| broadcast tracking | no | fusing it with the tactical tracking was planned and not built |
| audio | no | whistle detection was planned and not built |
| public 5-minute ground truth | sanity check only | scored once per accepted version, never used to choose |

## 4. 訓練資料 | Training data

> The models are trained on seven open German-league matches, degraded and smoothed until their statistics match the provided match. No label from the provided match is used.

No label of any kind from the provided match is used for training or tuning.

All models are trained on the seven matches of the open DFL / IDSSE dataset (Bassek et al., *An
integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data 2025,
CC-BY 4.0; paper <https://doi.org/10.1038/s41597-025-04505-y>, data
<https://doi.org/10.6084/m9.figshare.28196177>): J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ
and J03WR9, from the first and second German divisions. Each has 25 fps tracking of all players
and the ball, with a ball-status flag (in play or not) and a ball-possession flag on every frame.
These flags were produced by the data provider. We labelled nothing ourselves.

The possession flag changes on every touch. The challenge asks for controlled possession, so the
training target keeps a change of team only when the new team holds the ball for at least 2 s.

That tracking is clean and ours is not, so each training match is degraded to look like the
provided tracking (`validation/degrade.py`):

- the ball is dropped by a hazard process, with more loss when the ball is dead, in the air, near
  a line, or within 1 m of a player;
- the ball track is smoothed with a Gaussian of 6 frames (0.24 s). The provided ball track is
  smooth: its frame-to-frame acceleration has median 0.23 m/s and 99th percentile 1.5 m/s,
  against 0.75 and 23 on the raw training matches. After smoothing the training matches read 0.21
  and 1.4, and the distance from the ball to the nearest player matches as well;
- the provider freezes the ball during stoppages, so dead-ball positions are un-frozen to stop the
  model from learning "ball not moving means dead";
- players are cut into tracklets with dropout bursts and duplicates, and their tracks are smoothed
  with a Gaussian of 2 frames. The provided player tracks are smooth (median frame-to-frame
  acceleration 0.03 m/s); the training ones carry rounding jitter (0.25 m/s) until smoothed (0.03).

The degrader's parameters were fitted to aggregate statistics of the provided tracking (share of
missing ball frames, median gap length, ball acceleration quantiles, and so on). No labels were
involved. We also compare the distribution of every model feature between the degraded matches
and the provided match, and drop or repair features that differ. A first set of "ball touch"
features failed this check (they fired every 2 s on the training matches and almost never on the
provided match), which is how the smoothing was found.

The smoothing mattered more than any model change. The earlier rule-based version scored 0.59 on
the unsmoothed training matches and 0.51 once they were smoothed, because a smoothed ball cuts
corners and no longer reaches the player who plays it.

Training uses two random seeds of the degrader under three settings, which gives 42 degraded
matches. Every fifth frame is used. The three settings differ in how strongly losing the ball is
tied to a stoppage: the fitted one, one where it is barely tied, and one in between that matches
the provided match's share of missing ball frames (41%) exactly. We could not fit a single setting
that matches every statistic, so the models see all three.

## 5. 控球規則 | Possession rules implemented

> A contested or loose ball stays with the last team. A new team must keep the ball for two seconds before possession changes. A dead ball is always D.

| case | rule |
|---|---|
| loose or contested ball | the team decoder's switch penalty keeps the last team until the evidence for the other team is sustained |
| what counts as control | learned from the training target: a change of team that lasts at least 2 s |
| brief touch, deflection, duel | a spell under 2 s that interrupts the other team is absorbed |
| ball not visible | the model falls back on the gap context and on which team is advancing; the decoder carries the last team through |
| restart | the first alive frame after a dead segment takes the team the model gives it |
| dead ball | always `D`, never a team |
| frames outside the two halves | `X` |
| graded frames | never `X`. The scorer forward-fills `X`, so writing our own best label is never worse |

## 6. 驗證 | Validation

> Versions are chosen only by spellF1 from the official scorer on matches the models never saw. Label-free checks confirm that the models transfer to the provided match.

Versions are compared only on the other-game matches, degraded as above, with the official
`possession_metrics.py`. Models are trained on the other matches and scored on the one left out.
A change is accepted when the mean paired gain is at least +0.02 spellF1 and the bootstrap 90%
interval excludes zero.

| version | other-game spellF1 | macroF1 |
|---|---|---|
| baseline (nearest player, majority vote over ±50 frames) | 0.31 | 0.75 |
| `v1k` (rule-based team step) | 0.50 | 0.85 |
| `v3` (learned team model) | 0.69 | 0.90 |
| `v4` (submitted) | 0.70 | 0.91 |

<!-- figure:scores -->

The gain of `v3` over `v1k` is +0.18 (90% interval +0.17 to +0.20), better on every left-out
match. The gain of `v4` over `v3`, on a simulator seed neither had been tuned on, is +0.014 (90%
interval +0.007 to +0.020), better on 5 of 7 matches. That is below our +0.02 bar. We adopted `v4`
on the second rule we set ourselves: a change may go in when it does no harm on the other matches
and fixes something measurable on the provided match without ground truth. The next paragraph
is that measurement.

**Why `v4` despite the small gain.** On the other matches a true stoppage almost never lasts under
3 s (21 of 713). On the provided match `v3` predicted 44 such stoppages out of 162, each one
splitting a possession in two. `v4` predicts 13 out of 124, and still has a dead-ball segment
at 55 of the 58 set-piece tags.

The baseline scores higher on these matches than its published full-match 0.2433, so we expect
the real score of `v4` to be lower than its other-game score as well.

**Label-free checks on the provided match.** These use no ground truth.

- The submitted file reports 268 possessions (2.5 per minute) of mean 13.2 s,
  with 44.4% of graded frames dead.
- Of the 58 set-piece tags inside the two halves, 55 have a predicted dead-ball segment ending
  close to the tag.
- Smart Tagging names a team for phases such as `BUILD UP` and `PROGRESSION`. On frames inside
  those phases with the ball visible, `v4` gives the tagged team 84 to 97% of the time. The
  rule-based version gave 80 to 98%.

On the public 5 minutes, used only to check frame alignment and label meaning, `v4` scores
spellF1 0.62 and macroF1 0.85. That window has 17 possessions, so the number
carries almost no information about the full match.

## 7. 嘗試與捨棄 | Tried and dropped

> A larger model, a second stage, pass-chain features and a tagging prior were all tested and left out. The numbers are listed as measured.

| idea | result on the other-game matches | decision |
|---|---|---|
| short-gap ball interpolation | covers 0.3% of missing frames | dropped |
| rule-based team step (nearest player within 1 m, last team to show control keeps the ball) | 0.18 below the learned model | replaced |
| first learned team model with 13 features | +0.017 over the rules | replaced by the 106-feature model |
| larger team model (more trees, more leaves, every second frame) | +0.002 to +0.004 | not adopted |
| second-stage model that reads the first model's probabilities 8 s either side | +0.003 | not adopted |
| minimum possession of 1 s, 1.5 s or 2.5 s instead of 2 s | −0.006 to −0.016 | kept 2 s |
| Smart Tagging team phases as a prior on the team | cannot be tested on other matches; on the provided match it merged 34 possessions, and the other-game results say we already report too few | not adopted |
| training on 3 other matches instead of 6, same number of training runs | −0.012 | all seven matches are used |
| pass-chain features for the team model (which team struck the ball before and after, second-nearest players) | −0.004 | not adopted |
| refit of the simulator's ball-loss model on all seven matches | missing-ball share stays at 53% against 41% | kept the earlier fit and added the in-between setting to training (section 4) |

Three later changes were adopted and make up `v4`: the turnover model that sets the team decoder's
switch penalty (+0.008), the player-only dead-ball features with a stronger dead-ball penalty
(+0.009 to +0.012), and the third simulator setting with smoothed player tracks (+0.003).

## 8. 侷限 | Limitations

> Possessions under six seconds, and stretches where the ball stays hidden, are where most of the remaining loss is.

- **Short possessions.** On the other-game matches, possessions under 3 s are matched 30% of the
  time and those of 3 to 6 s 56%, against 90% for possessions over 12 s. We report about 13% fewer
  possessions than there are.
- **Turnover timing.** 62% of the team model's errors with the ball visible fall within 3 s of a
  true change of possession. More trees, a second stage and pass-chain features did not move this, so we read it as
  the limit of what tracking alone tells, or as noise in the training labels' timing.
- **Long ball gaps.** When the ball is visible for under a quarter of a possession, it is matched
  35% of the time. Ball re-detection from the video is the obvious fix and was not built.
- **Label definition.** The training target is the provider's possession flag with a 2 s rule
  applied by us. The challenge's ground truth comes from a different provider. A systematic
  difference between the two definitions would not show in our validation.
- **Simulator dependence.** All models learned from simulated tracking noise. No single simulator
  setting matches every statistic of the provided match, so three are mixed.
- **Events the training matches never show.** Goals, VAR checks and injuries have no dedicated
  rule beyond the kickoff formation check.

<!-- figure:length -->

With more time: ball re-detection in the gaps, a sequence model on the raw positions in place of
hand-built windows, and a simulator whose ball-loss model can match ball visibility and player
speed at the same time.

## 9. 重現 | Reproduction

> One command rebuilds the identical submission file from the raw data.

Python 3.10, CPU only, about 8 GB of memory.

```bash
pip install -r requirements-solution.txt
```

1. Unpack the match bundle into `data/`.
2. Run `python validation/fetch_idsse.py`. It downloads the match-information and position XML
   files of the seven matches listed in section 4 into `validation/cache/idsse/` (2.6 GB).
3. Run:

```bash
python solution/run.py --name <team_name>
```

This parses the tracking, trains the three models, writes `evaluation/results/<team_name>.csv`, prints
its SHA-256 and runs the official validator on it. Seeds are fixed and the run is single-threaded.
