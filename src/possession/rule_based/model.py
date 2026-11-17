"""Possession model v1: control evidence, restart rule and a switching cost.

    python -m possession.rule_based.model              # writes evaluation/results/possession_v1.csv
                                            # and data/debug/possession_v1.explain.npz
    python -m possession.rule_based.model --selftest   # synthetic checks of the decoder and the snapping

What v0 got wrong, and what v1 does instead
-------------------------------------------
v0 gave every frame with a visible ball one vote for the team of the nearest player, however far
away, however fast the ball, and smoothed with a +/- 2 s majority. Every touch counted, so it
produced far too many, far too short possessions. v1 follows CHALLENGE.md's definition instead:
possession changes only when the other team gains CONTROLLED possession; loose and contested
phases stay with the last team that had it; every restart has a clear possessor.

1. Dead ball: possession.rule_based.dead_ball, unchanged. Its D stretches split possessions (the scorer does
   the same), so everything below runs per live spell.

2. Control evidence, per frame, signed (+ home, - away), at most CONTROL_WEIGHT in size:

     near   how close the nearest player is: full weight within NEAR_M, none beyond FAR_M.
            A player in control has the ball within a stride (~1-1.5 m) plus tracking noise.
     sep    how much closer than the nearest opponent: none when both are equally close
            (a duel), full when the opponent is SEP_M further. A contested ball is not control.
     gate   ball speed: full weight up to DRIBBLE_MS, none from FLIGHT_MS. A dribbled ball moves
            at running pace; a faster ball is a pass, a shot, a clearance or a deflection, and an
            aerial ball is projected away from the camera, so 'nearest player' means least then.
            This is the physical form of "a touch or a deflection is not control".

   No ball (41% of graded frames) -> no evidence; the state simply carries on.

3. Smart Tagging, weakly: one per-frame ball-side signal from the union of tag intervals (a
   mirrored ATTACKING/DEFENDING pair counts once). Each contiguous run of that signal carries
   TAG_TOTAL evidence in all, spread evenly, which is below SWITCH_COST: a tag can tip a decision
   the tracking leaves open, never force one, and never imports its phase boundaries by itself.

4. Restart rule: each live spell starts with the team that restarts play. First choice: the team of
   the player nearest the ball at the moment the ball is released -- the last frame it is at
   rest (placed, or held for a throw-in) before it moves, searched from RELEASE_BACK_S before the
   spell to RESTART_S into it -- if he is within RESTART_RADIUS_M. At that moment the nearest
   player is the taker; a marker or a wall standing close does not decide it. Second: the team of
   a Smart Tagging SET PIECES tag (unless N/A) that belongs to this restart -- it starts inside
   the stoppage that ends here, or within RESTART_TAG_S of the restart (the tags start a few
   seconds before the restart); the latest such tag wins. Third: the team of the nearest player on
   the first RESTART_FRAMES sightings within RESTART_RADIUS_M, from RELEASE_BACK_S before the
   spell until the ball first reaches FLIGHT_MS: after a long kick or throw the players the ball
   flies past are receivers, not the taker. If nobody is seen there, the first sightings in the
   first RESTART_S decide (a restart usually goes to a team-mate). That is the
   weakest source at a set piece -- right after a corner or a free kick the ball is in the box
   among the defenders -- so it only decides when nothing better is there. Otherwise no prior.
   The restarter gets RESTART_BONUS at the spell's first frame.

   Passes: every completed pass -- the ball leaves a player and is next touched by a team-mate
   at least PASS_MIN_M away within PASS_MAX_S, with no stoppage in between (dead_ball.pass_chains)
   -- is PASS_EVIDENCE for the passing team at the reception. PASS_EVIDENCE is half the switching
   cost, so a change of possession needs more than two consecutive passes by the other team on
   their own: CHALLENGE.md's proxy for controlled possession ("a team completes more than two
   consecutive passes"), now used literally. Touches alone still count as before.

   Goalkeepers: a keeper with the ball in his own area after getting it in play (dead_ball's
   `keeper` feature, which also covers a ball lost in his hands, and a whole gap that ends with
   the ball reappearing with him) is control by his team,
   KEEPER_CONTROL per frame, with no speed gate: catching a fast ball is control.

5. Decoding: two-state (H/A) Viterbi per live spell, SWITCH_COST per change of possession. A
   change therefore needs about SWITCH_COST frames' worth of clear control by the other team
   (~1 s of a player alone on the ball, or several touches in a row), which is the challenge's
   "keeps the ball for a few seconds / more than two passes" proxy in evidence units.

6. Snapping: in a gap without evidence, the decoder's tie-breaking would place a change at the
   START of the gap (right after the old team's last touch). The rules say a loose or unseen ball
   stays with the last team, so each change is moved forward to the new team's first frame of
   control evidence (>= SNAP_MIN). Both the decoder's frame and the snapped one are recorded.

Every constant is a prior from football or physics, frozen before any scoring. None was fitted to
labels; the public 5 minutes are a pipeline check only (RULES.md, EVALUATION.md).
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from .dead_ball import REST_MS, _runs, dead_evidence, detect_dead, features  # noqa: E402
from .explain import (SRC_DEAD, SRC_MODEL, SRC_UNGRADED, dead_fields,  # noqa: E402
                                write_explain, write_submission)
from .loader import HOME, load_match  # noqa: E402

FPS = 25
NAME = "possession_v1"
OUT_PATH = os.path.join(ROOT, "evaluation", "results", NAME + ".csv")

# Params -- priors, frozen. See the module docstring for the reason behind each.
NEAR_M = 1.5             # m; within this the nearest player gets full weight
FAR_M = 4.0              # m; beyond this, no evidence
SEP_M = 2.0              # m; opponent this much further away -> uncontested
DRIBBLE_MS = 5.0         # m/s; ball at or below running-with-the-ball pace -> full weight
FLIGHT_MS = 10.0         # m/s; ball at or above this is in flight -> no evidence
CONTROL_WEIGHT = 1.0     # evidence per frame of clear control

TAG_TOTAL = 12.0         # evidence per contiguous Smart Tagging ball-side run, < SWITCH_COST

RESTART_S = 3.0          # look for the restarter in the spell's first 3 s
RESTART_RADIUS_M = 3.0   # a player this close to the ball is taking the restart
RESTART_FRAMES = 10      # first sightings that vote on the restarter
RELEASE_BACK_S = 1.0     # the release may come up to 1 s before the decoded spell start
KEEPER_CONTROL = 1.5     # evidence per frame of a keeper holding the ball (hands: certain)
PASS_EVIDENCE = None     # set below: SWITCH_COST / 2 (more than two passes = controlled possession)
RESTART_TAG_S = 5.0      # a SET PIECES tag starting within this of the restart (or in its stoppage)
SWITCH_COST = 25.0       # ~1 s of clear control by the other team
RESTART_BONUS = 2 * SWITCH_COST
PASS_EVIDENCE = SWITCH_COST / 2

SNAP_MIN = 0.1           # evidence a frame needs to count as the new team's first control

NO_TEAM = -1
# How a possession started (explain["poss_kind"]).
START_RESTART_TRACKING, START_RESTART_TAG, START_RESTART_NONE, START_CHANGE = 0, 1, 2, 3
START_RESTART_RELEASE = 4


# ---------------------------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------------------------

def control_evidence(d_home, d_away, ball_speed):
    """Signed per-frame control evidence (+ home, - away) and its three factors."""
    d_h = np.where(np.isnan(d_home), np.inf, d_home)
    d_a = np.where(np.isnan(d_away), np.inf, d_away)
    d_min, d_other = np.minimum(d_h, d_a), np.maximum(d_h, d_a)
    with np.errstate(invalid="ignore"):
        near = np.clip((FAR_M - d_min) / (FAR_M - NEAR_M), 0, 1)
        sep = np.clip((d_other - d_min) / SEP_M, 0, 1)
        sep = np.where(np.isinf(d_other) & np.isfinite(d_min), 1.0, sep)
        gate = np.clip((FLIGHT_MS - np.nan_to_num(ball_speed, nan=0.0))
                       / (FLIGHT_MS - DRIBBLE_MS), 0, 1)
    sign = np.where(d_h <= d_a, 1.0, -1.0)
    e = CONTROL_WEIGHT * near * sep * gate * sign
    e = np.where(np.isfinite(d_min), e, 0.0)
    return e, near, sep, gate


def tag_evidence(m):
    """Per-frame Smart Tagging ball side (+1 home, -1 away, 0 none or conflicting) and the evidence
    it contributes: TAG_TOTAL per contiguous run, spread evenly."""
    vote_h = np.zeros(m.n_frames, dtype=bool)
    vote_a = np.zeros(m.n_frames, dtype=bool)
    for t in m.tags:
        if t["role"] in ("attacking", "defending") and t["ball_side"] is not None:
            target = vote_h if t["ball_side"] == HOME else vote_a
            target[t["start_frame"]:t["end_frame"]] = True
    side = np.where(vote_h & ~vote_a, 1, np.where(vote_a & ~vote_h, -1, 0)).astype(np.int8)
    ev = np.zeros(m.n_frames)
    for s, e in _runs(side != 0):
        # Runs of one side; a flip inside counts as two runs.
        for s2, e2 in _runs(side[s:e] == side[s]) + _runs(side[s:e] == -side[s]):
            ev[s + s2:s + e2] = side[s + s2] * TAG_TOTAL / (e2 - s2)
    return side, ev


# ---------------------------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------------------------

def viterbi_possession(ev, switch_cost, prior=(0.0, 0.0)):
    """Most likely H(0)/A(1) path for signed evidence ev (+ favours H), paying switch_cost per
    change. prior is added to the two states' scores before the first frame. Ties stay."""
    n = len(ev)
    gain_h, gain_a = np.maximum(ev, 0.0), np.maximum(-ev, 0.0)
    back = np.zeros((n, 2), dtype=np.int8)
    sh, sa = prior[0] + gain_h[0], prior[1] + gain_a[0]
    for t in range(1, n):
        stay_h, move_h = sh, sa - switch_cost
        stay_a, move_a = sa, sh - switch_cost
        back[t, 0] = 0 if stay_h >= move_h else 1
        back[t, 1] = 1 if stay_a >= move_a else 0
        sh = max(stay_h, move_h) + gain_h[t]
        sa = max(stay_a, move_a) + gain_a[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = 0 if sh >= sa else 1
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def snap_switches(path, control):
    """Move each change forward to the new team's first frame of control evidence.

    Returns (snapped path, [(decoder_frame, snapped_frame)]).
    """
    path = path.copy()
    moves = []
    changes = list(np.flatnonzero(np.diff(path) != 0) + 1) + [len(path)]
    for k in range(len(changes) - 1):
        t, nxt = changes[k], changes[k + 1]
        new = path[t]
        own = control[t:nxt] if new == 0 else -control[t:nxt]
        hits = np.flatnonzero(own >= SNAP_MIN)
        t2 = t + int(hits[0]) if len(hits) else t
        path[t:t2] = path[t - 1]
        moves.append((t, t2))
    return path, moves


def release_frame(s, e, seen, ball_speed):
    """Last frame the ball is at rest before it moves, in [s - RELEASE_BACK_S, s + RESTART_S)."""
    a = max(1, s - int(RELEASE_BACK_S * FPS))
    b = min(e, s + int(RESTART_S * FPS))
    with np.errstate(invalid="ignore"):
        still = seen & (np.nan_to_num(ball_speed, nan=99.0) < REST_MS)
        moving = seen & (np.nan_to_num(ball_speed, nan=0.0) >= REST_MS)
    for k in range(a, b):
        if moving[k] and still[k - 1]:
            return k - 1
    return -1


def restarter(m, s, e, control_sign, d_min, tag_restarts, seen=None, ball_speed=None,
              stop_start=None):
    """(team, kind, distance, frame) of the team restarting play at a live spell [s, e).

    frame: the release frame (kind RELEASE) or the tag's start (kind TAG), else -1. stop_start is
    the first frame of the stoppage that ends at s, if any.
    """
    if seen is not None:
        q = release_frame(s, e, seen, ball_speed)
        if q >= 0 and d_min[q] <= RESTART_RADIUS_M:
            team = 0 if control_sign[q] > 0 else 1
            return team, START_RESTART_RELEASE, float(d_min[q]), q
    w = int(RESTART_TAG_S * FPS)
    lo = (s - w) if stop_start is None else min(stop_start, s - w)
    mine = [(start, side) for start, side in tag_restarts if lo <= start <= s + w]
    if mine:
        start, side = max(mine)
        return side, START_RESTART_TAG, np.nan, start
    end = min(e, s + int(RESTART_S * FPS))
    close = np.zeros(0, dtype=int)
    if ball_speed is not None:
        # The taker is at the ball before it flies; once it reaches FLIGHT_MS the players it
        # passes are receivers. So look from RELEASE_BACK_S before the spell (as the release
        # does) up to the first in-flight frame. Only if nobody is seen there, the first
        # sightings after the spell starts decide (a restart usually goes to a team-mate).
        lo = max(0, s - int(RELEASE_BACK_S * FPS))
        while lo < s and m.half[lo] != m.half[s]:
            lo += 1
        stop = end
        flying = np.flatnonzero(np.nan_to_num(ball_speed[lo:end], nan=0.0) >= FLIGHT_MS)
        if len(flying):
            stop = lo + int(flying[0])
        close = np.flatnonzero(d_min[lo:stop] <= RESTART_RADIUS_M)[:RESTART_FRAMES] + lo
    if len(close) == 0:
        close = np.flatnonzero(d_min[s:end] <= RESTART_RADIUS_M)[:RESTART_FRAMES] + s
    if len(close):
        home = int(np.sum(control_sign[close] > 0))
        team = 0 if home * 2 >= len(close) else 1
        return team, START_RESTART_TRACKING, float(d_min[close[0]]), -1
    return NO_TEAM, START_RESTART_NONE, np.nan, -1


# ---------------------------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------------------------

def build_labels(m, f, dead, team_evidence=None):
    n = m.n_frames
    if team_evidence is None:
        team_evidence = np.zeros(n, dtype=float)
    else:
        team_evidence = np.asarray(team_evidence, dtype=float)
        if team_evidence.shape != (n,) or not np.isfinite(team_evidence).all():
            raise ValueError('team evidence must be finite with one value per frame')
    graded = m.half > 0
    live = graded & ~dead
    labels = np.full(n, "X", dtype="<U1")
    labels[graded & dead] = "D"

    d_home, d_away = f["d_home"], f["d_away"]       # each team's nearest player (dead_ball)
    control, near, sep, gate = control_evidence(d_home, d_away, f["ball_speed"])
    keeper = f.get("keeper_poss", f["keeper"]) & live   # who has it: the whole keeper gap
    control = np.where(keeper, np.where(f["keeper_team"] == 0, KEEPER_CONTROL, -KEEPER_CONTROL),
                       control)
    control[~live] = 0.0
    pass_ev = np.zeros(n)
    for _, reception, team in f.get("passes", []):
        if live[reception] and team in (0, 1):
            pass_ev[reception] += PASS_EVIDENCE if team == 0 else -PASS_EVIDENCE
    tag_side, tag_ev = tag_evidence(m)
    tag_ev[~live] = 0.0
    d_min = np.fmin(d_home, d_away)
    control_sign = np.where(np.nan_to_num(d_home, nan=np.inf) <= np.nan_to_num(d_away, nan=np.inf),
                            1, -1)
    seen = ~np.isnan(f["ball_x"])
    # Set piece tags the tracking does not contradict (dead_ball.validate_setpieces).
    tag_restarts = [(t["start_frame"], t["side"]) for t in f["setpiece_tags"]
                    if t["status"] != "contradicted" and t["side"] is not None]
    # After a confirmed goal the team that conceded kicks off (Law 8).
    tag_restarts += [(kickoff, 1 - scorer) for _, kickoff, scorer in f.get("goals", [])
                     if scorer is not None]

    ex = {
        "source": np.where(graded & dead, SRC_DEAD, SRC_UNGRADED).astype(np.int8),
        "d_home": d_home.astype(np.float32), "d_away": d_away.astype(np.float32),
        "ball_speed": f["ball_speed"].astype(np.float32), "ball_seen_filled": ~np.isnan(f["ball_x"]),
        "control": control.astype(np.float32), "near": near.astype(np.float32),
        "sep": sep.astype(np.float32), "gate": gate.astype(np.float32),
        "tag_side": tag_side, "tag_ev": tag_ev.astype(np.float32),
        "spell_lo": np.full(n, -1, np.int32), "spell_hi": np.full(n, -1, np.int32),
        "poss_lo": np.full(n, -1, np.int32), "poss_hi": np.full(n, -1, np.int32),
        "poss_kind": np.full(n, -1, np.int8), "poss_decoder_frame": np.full(n, -1, np.int32),
        "poss_restart_dist": np.full(n, np.nan, np.float32),
        "poss_restart_frame": np.full(n, -1, np.int32), "keeper": keeper,
        "poss_ev_home": np.zeros(n, np.float32), "poss_ev_away": np.zeros(n, np.float32),
        "poss_start_ev": np.zeros(n, np.float32), "pass_ev": pass_ev.astype(np.float32),
        "poss_pass_home": np.zeros(n, np.int16), "poss_pass_away": np.zeros(n, np.int16),
        "pass_evidence": PASS_EVIDENCE,
        "switch_cost": SWITCH_COST, "restart_bonus": RESTART_BONUS,
    }

    stop_start_at = {e: s for s, e in _runs(graded & dead)}   # stoppage end -> its start

    for s, e in _runs(live):
        ev = control[s:e] + pass_ev[s:e] + tag_ev[s:e] + team_evidence[s:e]
        team, kind, dist, release = restarter(m, s, e, control_sign, d_min, tag_restarts,
                                              seen, f["ball_speed"], stop_start_at.get(s))
        prior = (0.0, 0.0) if team == NO_TEAM else \
            ((RESTART_BONUS, 0.0) if team == 0 else (0.0, RESTART_BONUS))
        path = viterbi_possession(ev, SWITCH_COST, prior)
        path, moves = snap_switches(path, control[s:e] + pass_ev[s:e])
        decoder_at = {t2: t for t, t2 in moves}
        labels[s:e] = np.where(path == 0, "H", "A")
        ex["source"][s:e] = SRC_MODEL
        ex["spell_lo"][s:e], ex["spell_hi"][s:e] = s, e

        for ps, pe in _runs(path == 0) + _runs(path == 1):
            a, b = s + ps, s + pe
            seg = slice(a, b)
            ex["poss_lo"][seg], ex["poss_hi"][seg] = a, b
            sign = 1.0 if path[ps] == 0 else -1.0
            if ps == 0:
                ex["poss_kind"][seg] = kind
                ex["poss_restart_dist"][seg] = dist
                ex["poss_restart_frame"][seg] = release
            else:
                ex["poss_kind"][seg] = START_CHANGE
                ex["poss_decoder_frame"][seg] = s + decoder_at.get(ps, ps)
            c = control[seg] + pass_ev[seg]
            ex["poss_ev_home"][seg] = float(np.sum(np.maximum(c, 0)))
            ex["poss_ev_away"][seg] = float(np.sum(np.maximum(-c, 0)))
            ex["poss_pass_home"][seg] = int(np.sum(pass_ev[seg] > 0))
            ex["poss_pass_away"][seg] = int(np.sum(pass_ev[seg] < 0))
            ex["poss_start_ev"][seg] = float(np.sum(np.maximum(sign * c[:2 * FPS], 0)))
    return labels, ex


# ---------------------------------------------------------------------------------------------
# GT-free diagnostics
# ---------------------------------------------------------------------------------------------

def diagnostics(labels, half):
    """Counts a football person can sanity-check without any ground truth."""
    rows = []
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        lab = labels[idx[0]:idx[-1] + 1]
        live = (lab == "H") | (lab == "A")
        spells = _runs(live)
        poss = [(s, e) for k in "HA" for s, e in _runs(lab == k)]
        dur = np.array([e - s for s, e in poss]) / FPS
        live_min = live.sum() / FPS / 60
        changes = sum(int(np.sum(lab[s + 1:e] != lab[s:e - 1])) for s, e in spells)
        rows.append({
            "half": h, "live spells": len(spells), "possessions": len(poss),
            "per spell": len(poss) / max(1, len(spells)),
            "median s": float(np.median(dur)) if len(dur) else 0.0,
            "mean s": float(dur.mean()) if len(dur) else 0.0,
            "< 2 s": float((dur < 2).mean()) if len(dur) else 0.0,
            "H share": float((lab == "H").sum() / max(1, live.sum())),
            "changes / live min": changes / max(1e-9, live_min),
        })
    return rows


def print_diagnostics(name, labels, half):
    print(name)
    for r in diagnostics(labels, half):
        print("  half {half}: {possessions} possessions in {live spells} live spells "
              "({per spell:.1f}/spell), duration median {median s:.1f} s / mean {mean s:.1f} s, "
              "{pct:.0%} under 2 s, H share {H share:.0%}, {changes / live min:.1f} H<->A changes "
              "per live minute".format(pct=r["< 2 s"], **r))


# ---------------------------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------------------------

def selftest():
    """Synthetic checks of decoding and snapping. No match data needed."""
    # H control, 50 frames without evidence, then A control: the change lands on A's first frame.
    ev = np.r_[np.full(40, 1.0), np.zeros(50), np.full(40, -1.0)]
    path, moves = snap_switches(viterbi_possession(ev, SWITCH_COST), ev)
    assert path[89] == 0 and path[90] == 1, (np.flatnonzero(np.diff(path)) + 1, moves)

    # A short touch by the other team (10 frames) inside a possession does not change it.
    ev = np.r_[np.full(40, 1.0), np.full(10, -1.0), np.full(40, 1.0)]
    assert np.all(viterbi_possession(ev, SWITCH_COST) == 0)

    # A sustained spell of control (40 frames) by the other team does.
    ev = np.r_[np.full(40, 1.0), np.full(40, -1.0)]
    path = viterbi_possession(ev, SWITCH_COST)
    assert path[39] == 0 and path[40] == 1

    # The restarter keeps the ball through an immediate brief touch by the opponent.
    ev = np.r_[np.full(5, -1.0), np.full(15, 1.0), np.zeros(30), np.full(20, -1.0)]
    path = viterbi_possession(ev, SWITCH_COST, prior=(0.0, RESTART_BONUS))
    assert path[0] == 1

    # Control evidence: alone on the ball -> full, duel -> none, ball in flight -> none.
    e, *_ = control_evidence(np.array([0.5, 1.0, 0.5]), np.array([6.0, 1.0, 6.0]),
                             np.array([2.0, 2.0, 15.0]))
    assert e[0] == CONTROL_WEIGHT and e[1] == 0 and e[2] == 0, e
    # Restarter priority: a named set-piece tag inside the stoppage beats the first sightings,
    # which right after a corner are the defenders in the box.
    d_min = np.full(400, 1.0)
    sign = np.full(400, -1)                             # away players nearest after the kick
    team, kind, _, frame = restarter(None, 300, 400, sign, d_min, [(150, 0)], stop_start=100)
    assert (team, kind, frame) == (0, START_RESTART_TAG, 150), (team, kind, frame)
    team, kind, _, _ = restarter(None, 300, 400, sign, d_min, [(20, 0)], stop_start=100)
    assert (team, kind) == (1, START_RESTART_TRACKING)  # tag belongs to an earlier stoppage

    # Long kick: the home taker is at the ball just before the spell (frames 95-101), the ball
    # flies from 105 and passes away players (110-130). The taker decides, not the receivers.
    from types import SimpleNamespace
    mh = SimpleNamespace(half=np.ones(200, dtype=np.int8))
    d_min = np.full(200, 10.0)
    d_min[95:102] = 1.0
    d_min[110:131] = 1.0
    sign = np.where(np.arange(200) < 105, 1, -1)
    v = np.r_[np.full(105, 2.0), np.full(95, 20.0)]
    team, kind, _, _ = restarter(mh, 100, 200, sign, d_min, [], ball_speed=v)
    assert (team, kind) == (0, START_RESTART_TRACKING), (team, kind)
    team, _, _, _ = restarter(mh, 100, 200, sign, d_min, [])   # without speed: the old rule
    assert team == 1

    # Release: ball at rest 0-29 (held for a throw), moving from 30: the release is frame 29.
    seen = np.ones(100, dtype=bool)
    v = np.r_[np.full(30, 0.3), np.full(70, 6.0)]
    assert release_frame(30, 100, seen, v) == 29
    print("selftest ok")
    return 0


# ---------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="run the synthetic checks and exit")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()

    m = load_match(with_homography=False)
    f = features(m)
    llr = dead_evidence(m, f)
    dead, reason = detect_dead(m, llr, f, return_reason=True)
    labels, ex = build_labels(m, f, dead)
    ex.update(dead_fields(f, llr, reason))

    graded = m.half > 0
    print("graded frames: {}".format({k: int((labels[graded] == k).sum()) for k in "HADX"}))
    print("wrote", write_submission(OUT_PATH, labels))
    print("wrote", write_explain(NAME, "viterbi", labels, ex))
    spell_starts = [s for s, _ in _runs(graded & ~dead)]
    print("restarters: release {}, tracking {}, tag {}, none {}".format(
        *[sum(1 for s in spell_starts if ex["poss_kind"][s] == k)
          for k in (START_RESTART_RELEASE, START_RESTART_TRACKING, START_RESTART_TAG,
                    START_RESTART_NONE)]))
    print()
    print_diagnostics(NAME, labels, m.half)
    v0 = os.path.join(ROOT, "evaluation", "results", "possession_v0.csv")
    if os.path.isfile(v0):
        import pandas as pd
        print_diagnostics("possession_v0", pd.read_csv(v0)["label"].astype(str).values, m.half)
    return 0


if __name__ == "__main__":
    sys.exit(main())
