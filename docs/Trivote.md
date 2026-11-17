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

**Dead ball** (`dead_ball.py`). Several weak signals are summed into a per-frame dead/live
log-likelihood ratio, then decoded with a two-state Viterbi and a switching cost:

- **Signals pointing to dead ball:** collective outfield speed (players walk after a whistle), ball
  beyond the lines (with a tighter margin on the near touchline, because a ball in the air projects
  away from the camera), missing ball (weak), validated Smart Tagging set-piece starts, and the
  kickoff formation.
- **Signals pointing to live play:** fast ball inside the pitch, a dribble (carrying pace with no
  close opponent), and chains of two or more passes by one team.
- **Rules applied on top of the decoder:**
  - Strong triggers (ball out, a set-piece tag) protect a stoppage.
  - Each half starts with a kickoff.
  - A validated goal is dead until the kickoff.
  - A ball resting on a Laws-defined restart spot (goal area, corner arc, touchline, penalty spot,
    centre spot), with opponents where the Laws put them, is a stoppage until it leaves the spot.
  - An out-of-play stoppage starts at the frame the ball crosses the line.
  - Play resumes only when the ball is played.
  - The minimum stoppage length is `MIN_STOP_S`.

**Possession** (`model.py`), decoded separately within each live spell:

- **Control evidence (signed H/A).** Distance of the nearest player: full weight within 1.5 m, none
  beyond 4 m. Separation from the nearest opponent: none in a duel. A ball-speed gate: full weight up
  to 5 m/s, none from 10 m/s, because a pass, a shot or a deflection is not control.
- **Passes.** Each completed pass gives the passing team half the switching cost. On their own,
  more than two consecutive passes by the other team change possession.
- **Goalkeeper.** A keeper with the ball in his own area counts as control, with no speed gate.
- **Smart Tagging.** Its ball side is weak evidence: each contiguous tag run adds 12 units in total,
  below the switching cost of 25. A tag can tip an open decision but cannot force one.
- **Restart rule.** Every live spell starts with the team that restarts it. This is the team of the
  player nearest the ball at release. If that is not available, the team of a set-piece tag
  belonging to this restart is used. If neither is available, the team of the first nearby
  sightings is used.
- **Decoding.** A two-state Viterbi with switching cost 25 needs about 1 s of clear control by the
  other team to change possession. Each change is then moved forward to the new team's first frame
  of control evidence, so a loose or unseen ball stays with the last team.

All constants are physical or football priors written down before any scoring. None was fitted to
labels.

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
