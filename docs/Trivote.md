# Trivote — Ball possession on the tactical feed

Technical document for More than a Hack 2027, Phase 1. The submission labels every frame of the
Barcelona–Athletic tactical video (199,319 frames, 25 fps) as `H`, `A`, `D` or `X`.

- **Possession file:** `Trivote.csv`, SHA256
  `3cf89a2c375dcfc6eedd1a4ed5489c9726482d9f65c0f2cfdc17ee98fa91f7f9`.
- **Method:** a rule-based possession and dead-ball system. A small temporal convolutional network
  (TCN v6), trained only on other matches, adds bounded evidence to it.
- **Reproduction:** one command, CPU only (about 30 s on a laptop), deterministic. It reproduced
  the same bytes in three independent runs (see [Reproduction](#6-reproduction)).

## 1. Approach

The ranking metric scores whole possessions (`spellF1`), not frames. A system that flickers between
teams on every touch can score well per frame and still lose almost every possession. So the core of
the system is two **sequence decoders with switching costs**: one for dead and live play, and one
for home and away. Every per-frame signal is only *evidence* fed to these decoders. A label changes
only when enough evidence builds up to pay for the switch.

```
ATD tracking + Smart Tagging
   ├─► dead-ball evidence (log-odds) ─┐
   │                                   + TCN D cue ─► 2-state Viterbi (dead/live) + football rules
   └─► control evidence (signed H/A) ─┐
                                       + TCN H/A cue ─► per live spell: restart prior,
                                                         2-state Viterbi (H/A), snapping ─► H/A/D/X
```

**Why rules first, a learned model second.** No labelled data exists for this match, and the rules
forbid creating any. The tactical camera's tracking is also very different from any public training
data: the ball is visible on only about 57% of graded frames, while public tracking datasets observe
it almost all the time. A model trained elsewhere cannot be trusted to decide alone in that domain.
Rules derived from the Laws of the Game and from physics transfer across matches. The TCN is
therefore used as a capped extra vote inside the rule decoders, not as the decoder.

### 1.1 Rule-based core (`src/possession/rule_based/`)

The rule-based system turns every graded frame into a label in five steps. The steps run in
the order below, and each later step can only use what the earlier ones produced. All thresholds
are the constants in `dead_ball.py` and `model.py`; they are physical or football priors, written
down before any scoring, and none was fitted to labels. The two TCN cues of §1.2 enter at exactly
two points, marked **[TCN]**.

**Step 0 — Load and align** (`loader.py`, `dead_ball.ball_track`)

- ATD rows are read in order and every `frame_id` is checked: `video_frame = frame_id − 1`.
- The graded halves come from `games.half_video_blocks()`. Frames outside them are never decoded.
- Normalised coordinates are scaled to metres with the pitch size in `games.py`.
- Within each half, ball gaps of up to 10 frames (0.4 s) are interpolated linearly. Longer gaps
  stay missing. Ball speed is computed in m/s from this track.

**Step 1 — Per-frame features** (`dead_ball.features`)

| Feature | Definition |
|---|---|
| `d_home`, `d_away` | Distance from the ball to the nearest tracked player of each team. A "goalkeeper" track standing in the other team's penalty area is given to the team defending that goal, because the tracker swaps identities. |
| collective speed | Mean speed of tracked outfield players, smoothed over 1 s. |
| ball out | Ball beyond the lines for at least 5 frames. The margin is 0.5 m on the near touchline and 1.5 m elsewhere, because a ball in the air is projected away from the camera. |
| dribble | Ball inside the pitch at 2–8 m/s, a player within 1.5 m, and the nearest opponent at least 1 m further away. |
| pass | The ball leaves a player's touch (≤ 1.5 m) and is next touched by a team-mate at least 5 m away within 3 s. A ball-out or a set-piece start in between breaks it. |
| pass chain | Two or more consecutive passes by one team. |
| keeper | A goalkeeper with the ball in his own penalty area that he got in play: no ball-out in the 3 s before. For a ball gap that ends with the ball reappearing at him, the whole gap counts as his. |
| set-piece tag | Each Smart Tagging SET PIECES instance is checked against the tracking. Weight 1.0 if the ball is put down on a matching spot, 0.5 if no rest is seen, ignored if the ball rests on a different kind of spot. |
| kickoff formation | At least 8 tracked outfield players of each team, all in their own half (1 m tolerance), for at least 1 s. The last 1 s of each run carries no evidence, because the kick happens there. |

**Step 2 — Dead-ball evidence** (`dead_ball.evidence_terms`)

Each frame gets a dead-vs-live log-odds score: the sum of the terms below. A positive score means
the frame looks dead.

| Term | Log-odds | When |
|---|---|---|
| speed | `clip(1.5 · (1.6 − v̄), −2, +2)` | Always. Players walk (v̄ < 1.6 m/s) after a whistle. |
| out | +3.0 | Ball out (Step 1). |
| missing | +0.3 | Ball not seen. Weak, because the ball is also lost in live play. |
| fast | −1.0 | Ball inside the pitch faster than 8 m/s: a pass or a shot. |
| dribble | −0.8 | Dribble (Step 1). |
| chain | −2.0 | Pass chain (Step 1). |
| keeper | cancels the positive part of *speed* and *missing* | Keeper frames: a ball in his hands is not missing, and a team waiting for his release is not walking because play stopped. |
| set piece | +1.0 (or +0.5) | From 4 s before to 2 s after a validated tag start. |
| kickoff | +3.0 | Kickoff formation. |
| **[TCN]** dead cue | `0.25 · clip(log(pD/(1 − pD)), ±2)`, so at most ±0.5 | Only on frames where the raw ball is observed. |

**Step 3 — Decide dead or live** (`dead_ball.detect_dead`, once per half)

1. **Viterbi.** A two-state (live/dead) Viterbi decodes the scores, paying 30 log-odds per switch.
   A stoppage therefore needs sustained evidence, not one dead-looking frame.
2. **Protected stoppages.** A second Viterbi runs without the play terms (dribble, chain, keeper).
   If it finds a stoppage within 2 s of a strong trigger (ball out, or a validated set-piece start)
   that the first pass erased entirely, that stoppage is restored. Play evidence may shorten a
   throw-in or a goal kick, but cannot erase it.
3. **Kickoff.** Every frame of a half before its first kickoff formation is dead.
4. **Ball-out timing.** If the ball goes out within 5 s of a decoded stoppage start, the start moves
   to the frame the ball crossed the line.
5. **Goals.** A GOALS tag counts only if a kickoff formation ends within 10 s of the tag's end and
   the kicker is within 2 m of the centre spot. A drinks break also puts every player in his own
   half, but nobody stands on the centre spot. The frames from the last shot at that goal (within
   25 m, at 8 m/s or faster, towards the goal) to the kickoff are dead.
6. **Restart spots.** The ball rests (slower than 1 m/s for at least 1 s) on a spot where the Laws
   restart play, with 1.5 m tolerance: the goal area, a corner arc, a touchline band of ±2 m, a
   penalty spot or the centre spot. The opponents must be where the Laws put them: 9.15 m away, or
   2 m at a throw-in, minus 1.5 m tolerance. Then the frames are dead from the ball's last contact
   with the relevant line, provided no play happened since (looking back at most 300 s), until the
   ball leaves the spot.
7. **Wait for the restart.** After a protected stoppage, if the ball comes to rest for 0.5 s within
   30 s of the decoded end, before any play evidence, the stoppage lasts until the ball departs. A
   ball carried back by hand is not a restart.
8. **Wait for the ball to move.** If the ball sits still at the end of a stoppage (seen on at least
   60% of the last 1 s, never more than 0.5 m from its spot), the stoppage lasts until the ball is
   more than 0.5 m away, for at most 30 s.
9. **Minimum length.** Dead runs shorter than 3 s are returned to live play, except the forced
   stretch before a kickoff.

**Step 4 — Decide who has possession** (`model.build_labels`, once per live spell)

Each maximal live run between two stoppages is a *spell*. The spell is decoded on its own; no
possession crosses a stoppage.

1. **Signed evidence per frame** (+ home, − away). These are summed:
   - **control:** `near · sep · gate`, signed towards the team of the nearest player, where
     - `near = clip((4 − d_min) / 2.5, 0, 1)`: full weight within 1.5 m, none beyond 4 m;
     - `sep = clip((d_other − d_min) / 2, 0, 1)`: zero in a duel, full when the opponent is 2 m
       further away;
     - `gate = clip((10 − v_ball) / 5, 0, 1)`: full weight up to 5 m/s, none from 10 m/s, so a
       pass, shot, clearance or deflection is not control.
   - **keeper:** on keeper frames, ±1.5 for the keeper's team replaces *control*, with no speed
     gate. Catching a fast ball is control.
   - **pass:** ±12.5 (half the switching cost) for the passing team at each completed pass's
     reception.
   - **Smart Tagging:** ATTACKING/DEFENDING tags with a known ball side give a per-frame side. A
     mirrored pair counts once; conflicting sides give zero. Each contiguous run adds 12 in total,
     spread evenly over its frames: below the switching cost, so a tag can tip a decision but never
     force one.
   - **[TCN]** **team cue:** `0.25 · clip(log(pH/pA), ±2) · (1 − pD)`, so at most ±0.5 per frame,
     and only where the raw ball is observed.
2. **Restarting team.** The first source that applies decides:
   1. The *release frame* is the last frame the ball is at rest before it moves, searched from
      1 s before the spell to 3 s into it. If a player is within 3 m of the ball there, his team
      restarts. At release the nearest player is the taker; a wall or a marker does not decide.
   2. Otherwise, a validated (not contradicted) set-piece tag with a known team that starts inside
      the stoppage ending here, or within 5 s of the restart, decides. The latest such tag wins.
      After a confirmed goal, the conceding team restarts.
   3. Otherwise, the first 10 sightings of a player within 3 m vote by majority. They are taken
      from 1 s before the spell until the ball first reaches 10 m/s, because the players a long
      kick flies past are receivers, not the taker. If there are none, the first 3 s of the spell
      are used instead.
   4. Otherwise there is no prior.

   The restarting team gets +50 (twice the switching cost) at the spell's first frame.
3. **Viterbi.** A two-state (H/A) Viterbi decodes the summed evidence, paying 25 per change. A
   change needs about 1 s of clear, uncontested control by the other team, more than two
   consecutive passes, or a keeper holding the ball.
4. **Snapping.** In a stretch without evidence, the Viterbi would place a change right after the old
   team's last touch. So each change is moved forward to the new team's first frame whose
   *tracking* evidence (control plus passes, ≥ 0.1) points to it. Tags and the TCN cue do not move
   a change. Until that frame, a loose or unseen ball stays with the old team.

**Step 5 — Write the labels** (`model.build_labels`, `predict.apply_published_dead_intervals`)

| Frame | Label |
|---|---|
| graded and dead | `D` |
| graded and live | `H` or `A`, from Step 4 |
| not graded | `X` |
| the published second-half pre-kickoff interval `[110808, 113742)` | `D`, overriding the above |

### 1.2 TCN evidence (`src/possession/model_v2.py`, `src/possession/rule_based_tcn_evidence.py`)

- **Architecture and inputs.** The TCN has 8 centred residual 1-D convolution layers, 128 channels
  and a receptive field of 511 frames. It reads 25 per-frame features (schema version 2), of which 4 are tag channels held at zero:
  - per-team distance, control support, speed, spread and nearby-player counts;
  - ball speed and its change, and distance to the boundary;
  - ball observed, interpolated, rejected and available flags;
  - distance and control advantage;
  - time since the ball was last observed, and length of the current ball gap.
- **Outputs.** H/A/D probabilities, plus an auxiliary transition head used only in training.
- **No tag input.** The training matches have no Smart Tagging, so the tag channels are zero both in
  training and at inference.
- **Fusion.** The TCN probabilities become two bounded cues, added to the rule system's own
  evidence:
  - team cue: `0.25 · clip(log(pH/pA), ±2) · (1 − pD)`, added to the H/A control evidence;
  - dead cue: `0.25 · clip(log(pD/(1 − pD)), ±2)`, added to the dead-ball log-odds.

  Both cues are zero on frames where the raw ball is not observed. The rule decoders, restart rules
  and snapping are unchanged. A TCN that outputs nothing reproduces the rule-based system exactly.
- **Where the fusion design came from.** The 0.25 scale, the ±2 cap and the observed-ball-only
  gating were fixed in a pre-registered protocol (`docs/rule_based_tcn_experiment.md`,
  `configs/rule_based_tcn_frozen.json`). That protocol compared rules only, rules plus team cue,
  rules plus D cue and both, using an earlier checkpoint (v4) on an external DFL development match.
  "Both" was selected there. v6 reuses the same fusion unchanged, so this exact checkpoint-and-fusion
  combination was not separately validated. The manifest records this as
  `user_selected_untested_combination`.

### 1.3 Labels outside play

Frames outside the official graded blocks (`games.half_video_blocks()`) are `X`. The published
pre-kickoff part of the second half, frames `[110808, 113742)`, is labelled `D`. Each half is
processed independently. No frame numbers from this match appear in any decision rule.

## 2. Assets used

| Asset | Used | How |
|---|---|---|
| Tactical ATD tracking (`BAR-ATH_tactical_atd.txt/.xml`) | **yes** | All player and ball positions. `video_frame = ATD frame_id − 1`, checked row by row. Normalised coordinates are scaled to metres with the pitch size in `games.py`. Track slots are not treated as player identities. |
| Smart Tagging (automatic) | **yes** | Weak evidence only: ball-side runs, set-piece candidates (validated against where the ball is put down), and goals (only when a kickoff follows). |
| `games.half_video_blocks()` | **yes** | Graded half boundaries and the published second-half pre-kickoff interval. |
| Tactical video | no | A local ball detector (WASB) was planned but never run. No pixels are used. |
| Homography | no | Not needed: ATD is already in pitch coordinates. Aerial-ball height cannot be recovered from a ground homography anyway. |
| Broadcast video and its ATD | no | Different timeline, and it contains replays. |
| Public 5-minute ground truth | pipeline check only | See §3 and §5. It is not used for training, thresholds, early stopping or model selection, and no inference or training code reads `evaluation/gt`. |

Tried and dropped:
- A pure TCN decoder (`scripts/predict_temporal.py`). It is kept as a research path; it has no
  notion of restarts or the Laws, which the rule decoders supply.
- The pace of the players around the ball as a dead-ball signal. Players sprint to take quick
  restarts, so it misfires.
- Treating a whole set-piece tag window as `D`.
- Earlier TCN checkpoints. v4 and v5 are still selectable with `--tcn-model`.

## 3. Training data

The rule system has no training data. The TCN v6 was trained on the datasets below; **none of it
comes from the provided match**.

| Dataset | Source / licence | Labels and who produced them | Amount used |
|---|---|---|---|
| IDSSE / DFL Bundesliga, 7 matches (`J03WN1`, `J03WMX`, `J03WPY`, `J03WQQ`, `J03WOY`, `J03WOH`, `J03WR9`) | Bassek et al., *Scientific Data* (2025); figshare article 28196177 (`https://api.figshare.com/v2/articles/28196177`), CC BY 4.0. Fetched with checksums by `scripts/fetch_dfl_matches.py`. | Per-frame `BallStatus` (alive/dead) and `BallPossession` (home/away), taken as published in the official DFL position-data feed. We did not create or edit any label. | Every labelled frame of all 7 matches, as both an original and an augmented copy (see below). |
| Metrica Sports sample data, Games 1 and 2 | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data), pinned revision; attribution requested by the README. | Human-annotated events published by Metrica. We derive H/A **proxy** labels only inside PASS events; all other frames are masked, and there is no `D` supervision. | Pass-interior frames of both games. |

In total there were 2,045,533 supervised sequence frames: H 608,610, A 537,845, D 899,078. Each
DFL match appears twice, so these are not independent frames.

**Augmentation.** The second copy of each DFL match has its ball observations removed following
the target match's *ball-visibility pattern*. Only the boolean "ball seen / not seen" timing is
taken from the provided tactical ATD; no positions or labels from the target are used. This brings
the long ball gaps of the tactical camera into training. Other augmentations are team swap (0.5),
random missing-ball blocks (25 frames, p = 0.25) and tag dropout.

**Training setup.** Fixed before training: seed 2032, 12 epochs, learning rate 3·10⁻⁴, boundary
loss 0.1, smoothness loss 0.05. There was no validation split and no checkpoint selection; the last
epoch was used. That took 52,224 steps (681 s of training on a single CUDA GPU). The 12-epoch budget was set from an
earlier 7-fold leave-one-match-out cross-validation on DFL, in which the best epoch was usually the
last of 6. In that study, the same method family (`context_masked`) reached a mean spellF1 of 0.690
on clean held-out DFL matches and 0.537 with synthetic ball masking. Because v6 is trained on all 7
matches, **there is no held-out DFL test for v6 itself.**

## 4. Possession rules implemented

- **Change of possession** only when the other team gains *controlled* possession. That means about
  1 s of clear control (close, uncontested, a ball at carrying pace), or more than two consecutive
  completed passes, or a goalkeeper holding the ball.
- **Loose balls, deflections, shots, clearances and aerial balls** are not control: the speed gate
  removes their evidence. Possession stays with the last controlling team until the other team's
  first frame of real control.
- **Challenges and duels:** when both teams are equally close, there is no evidence; the state
  carries on.
- **Ball not visible** is not `D` and not a change. The state carries on unless other evidence
  indicates a stoppage.
- **Restarts:** the stoppage runs from the moment the ball crosses the line until the ball is played from the restart spot. The new spell starts with the restarting
  team: the taker at release, then the set-piece tag team, then the first nearby player.
- **Set pieces:** the tag window itself is not `D`. Only validated tag starts count as stoppage
  evidence, and the stoppage ends when the ball leaves the spot.
- **Goals:** dead from the decisive shot until the following kickoff, which the conceding team
  takes.
- **`Team = N/A`** tags give no team evidence. A defending-phase tag is read as evidence for the
  other team, never as possession for the defending team.

## 5. Results and checks

- **Format.** `evaluation/validate_submission.py` passes: 199,319 rows, frames 0…199318.
- **Label distribution.** H 25.5%, A 18.6%, D 35.6%, X 20.3%. The validator counts 241
  possessions, with a mean of 14.6 s.
- **Determinism.** Three independent full runs produced byte-identical CSVs (SHA256 above).
- **Public 5-minute window,** scored once on the frozen file as an alignment and pipeline check:
  spellF1 0.579, macroF1 0.898, and 12 predicted possessions against 17 true ones. In the same run,
  `baseline_smooth50` scored spellF1 0.326. With 17 possessions the scorer's noise band is about
  ±0.39, so this says the frames are aligned and the labels mean what we think. **It is not an
  estimate of the full-match score**, and no parameter was changed after seeing it. Earlier research
  candidates were also scored once each on this window as pipeline checks; those numbers did not set
  any threshold or training choice.

## 6. Reproduction

Requirements: [uv](https://docs.astral.sh/uv/); tested on Windows 11. Python 3.11 and every
dependency are pinned in `uv.lock`. Inference runs on the CPU, and no GPU is needed. Unpack the
organisers' data bundle into `data/` as described in `DATA.md`. The trained checkpoint
`data/runs/training/dfl_v6.pt` is committed to the repository, and inference refuses any file that
does not match its SHA256.

```bash
uv sync --locked --extra starter --extra training

# Possession file (+ a VLC .ass overlay next to the tactical video)
uv run --locked --extra starter --extra training python scripts/rule_based/pipeline.py
#   -> data/runs/final_pipeline/Trivote.csv

# Official format check
uv run --locked --extra starter python evaluation/validate_submission.py data/runs/final_pipeline/Trivote.csv

# Two clean processes, validator on each, SHA256 comparison
uv run --locked --extra starter --extra training python scripts/check_reproduction.py \
    --run-dir data/runs/final_pipeline/reproduction \
    --final-output data/runs/final_pipeline/reproduction/Trivote.csv
```

The expected SHA256 of `Trivote.csv` is `3cf89a2c375dcfc6eedd1a4ed5489c9726482d9f65c0f2cfdc17ee98fa91f7f9`.
Each run also writes a manifest with the hashes of the inputs, config, checkpoint and source files
to `data/runs/Trivote.manifest.json`.

Retraining the TCN (optional; needs CUDA):

```bash
uv run --locked --extra starter --extra training python scripts/fetch_dfl_matches.py \
    --matches J03WN1 J03WMX J03WPY J03WQQ J03WOY J03WOH J03WR9
uv run --locked --extra starter --extra training python scripts/fetch_external_data.py
uv run --locked --extra starter --extra training python scripts/train_dfl_final.py \
    --output data/runs/training/dfl_v6_retrained.pt --epochs 12 --learning-rate 0.0003 --device cuda
```

A retrained checkpoint is close to v6 but not guaranteed to be bit-identical, because of GPU
non-determinism and later refactors of the feature code. The submission is reproduced from the
committed checkpoint.

## 7. Limitations and what we would do next

- **Domain shift.** The TCN learned from DFL tracking, where the ball is almost always observed, and
  from Metrica pass proxies. Masking with the target's visibility pattern narrows this gap but does
  not model the tactical camera's other errors, such as wrong ball candidates and identity swaps.
- **Fusion not re-validated for v6.** The cue weights were chosen with the v4 checkpoint, and v6 has
  no independent held-out test.
- **Dead ball.** Stoppages decided only by a referee's whistle (fouls, offside) with no ball-out and
  no clear restart spot are hard to detect from tracking, and short stoppages are often missed.
- **Aerial balls and long ball gaps.** The ground-plane projection mis-places aerial balls. During
  long gaps, possession simply carries on, which costs us at real hand-overs that happen while the
  ball is unseen.
- **Smart Tagging is automatic** and sometimes wrong. We validate set pieces against the ball, but
  a wrong tag can still shift a restart's team.
- **With more time:**
  - run a local ball detector on the tactical video to fill ball gaps;
  - validate the TCN fusion weights for v6 on a held-out match;
  - add explicit foul/offside stoppage cues;
  - make the TCN team-symmetric by averaging over team-swapped inputs.
