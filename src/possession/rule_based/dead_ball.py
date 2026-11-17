"""Dead-ball detector: for every graded frame, is the ball in play or not?

    python -m possession.rule_based.dead_ball                 # run on the full match, print diagnostics

    from possession.rule_based.dead_ball import dead_evidence, detect_dead
    llr  = dead_evidence(m)     # per-frame log-likelihood ratio, > 0 means "looks dead"
    dead = detect_dead(m)       # per-frame bool, from a two-state Viterbi over llr

Approach
--------
No single signal separates live from dead play, so several weak, physically motivated pieces of
evidence are summed into one per-frame log-likelihood ratio (dead vs live), and a two-state Viterbi
with a switching cost turns that into contiguous stoppages. The switching cost is what makes the
output look like football -- a few dozen stoppages per half, not thousands of flickers.

Evidence, each a term in log-odds units (positive = dead):

  speed     Mean speed of the outfield players. When the referee stops play, players walk.
            Centred on 1.6 m/s, just above ordinary human walking speed (~1.4 m/s).
            (The pace of the players around the ball was tried and dropped: before a throw-in or
            a quick restart the players near the ball run to take it, so a fast local pace does
            not mean the ball is in play. It is still computed, for the debug overlay only.)
  out       Ball outside the pitch lines. The tactical camera sits on the near touchline (y = 1),
            so a ball in the air is projected AWAY from it -- beyond the far touchline or the goal
            lines. The near touchline therefore gets a tight margin, the other three a wider one.
  missing   Ball not seen. Weak: the ball is lost in live play too, but more often in stoppages.
  fast      Ball moving fast inside the pitch -- passes and shots. Evidence for live play.
  dribble   Ball inside the pitch, moving at a carrying pace (PLAY_MIN_MS .. FAST_BALL), with one
            player within CARRY_M and no opponent as close. A ball is not dribbled at a stoppage:
            it is still, out, held in the hands (walking pace) or not seen. Evidence for live play.
  keeper    A goalkeeper with the ball in his own penalty area, after the ball came to him in play
            (not after it went out): at his feet (ball seen) or in his hands (ball lost right
            after he had it, for at most KEEPER_HOLD_MAX_S, the Law 12 limit). The ball is also
            often lost BEFORE he has it -- a cross or a shot he catches is lost in flight -- and
            stays lost while he holds it. So a whole gap is his too when the ball reappears with
            him (within KEEPER_M, in his own area, not in the goal area: a goal kick may be about
            to be taken there, Law 16; the penalty spot is no exception -- at a penalty the keeper
            stands on his line, Law 14) and it was last seen in that same area (Law
            12: he can handle it only there), not at rest on a restart spot (placed for a
            restart), with no ball-out in the KEEPER_FROM_PLAY_S before the gap. The reappearance
            says WHO had the ball for the whole gap (the possession model uses all of it); it
            does not say play was live: a whistle while he holds it (offside, a foul) is not in
            the tracking. So the dead-ball terms are cancelled only for the first
            KEEPER_HOLD_MAX_S of any gap (Law 12: in play he must release it by then); after that
            the usual evidence decides. It is NOT evidence
            for live play -- a keeper also holds the ball at a stoppage, after a foul on him --
            but it cancels the two terms it would otherwise trigger falsely: a ball lost in his
            hands is not "missing", and a team waiting for his release is not "walking because play
            stopped". So on these frames the dead side of speed and missing is removed, nothing
            else. Goalkeeper slots are checked first: a
            "goalkeeper" track standing in the other team's penalty area is the other keeper
            (the tracker swaps identities), so it takes the team that defends that goal.
  chain     Two or more passes in a row by one team: the ball leaves one player and is next touched
            by a team-mate at least PASS_MIN_M away, within PASS_MAX_S. A single pass is not used
            -- at a stoppage the ball is often handed back to the taker -- but a chain of passes
            is play. Strong evidence for live play. A pass only counts with the ball inside the
            pitch at both touches, and a chain breaks where the ball goes out or a Smart Tagging
            set piece starts: play cannot carry on through a stoppage.
  setpiece  Smart Tagging SET PIECES instances begin a few seconds before the restart, so the
            frames just around a tag's start are evidence of a stoppage. It is automatic tagging,
            so it is only evidence, and it is checked against the tracking first
            (validate_setpieces): a tag whose ball is put down on a matching spot counts fully, one
            with no rest seen counts half, one whose ball rests on another kind of spot (a
            "throw-in" with the ball 16 m inside the pitch) is ignored.
  kickoff   Kickoff formation: every tracked outfield player of each team in its own half. That
            happens only while waiting for a kickoff, so it is strong evidence of a dead ball --
            except for its last second, because the ball is struck just before the formation breaks.

Six football rules are applied on top of the Viterbi:

  * Strong triggers protect a stoppage. The decoder runs twice: with every term, and without the
    two play terms (dribble, chain). A stoppage the second pass finds that has a strong trigger --
    the ball out of the pitch, or a Smart Tagging set piece starting -- within ANCHOR_S of it, and
    that the first pass removed entirely, is restored. Play evidence may shorten such a stoppage,
    never erase it: a throw-in or a goal kick is a stoppage even when the restart is quick.

  * Play in a half starts with a kickoff. Frames of a half before its first detected kickoff
    formation are dead. (This is what keeps the stretch before the second-half kickoff dead without
    any frame number.)
  * A goal is dead until the kickoff. A Smart Tagging GOALS instance counts only when a kickoff
    follows it (Law 8): a formation with every player in his own half ending within
    GOAL_KICKOFF_S of the tag's end, AND the kicker at the centre spot (within KICKER_M) -- a
    drinks break also puts everyone in his own half, but nobody at the centre spot. The goal is
    the last shot at the conceded goal (within SHOT_RANGE_M, towards it, SHOT_MS or faster)
    before the scoring team is back in its own half; from there to the kickoff everything is
    dead -- players running to celebrate are not play. The team that conceded kicks off.
  * Restart spots (Laws 8, 14, 15, 16, 17). A ball at rest for REST_SPOT_MIN_S on a spot where
    the Laws restart play -- the goal area (goal kick), a corner arc, a touchline (throw-in),
    a penalty spot, the centre spot (kick-off) -- with the opponents where the Laws put them
    (outside the penalty area; 9.15 m away; 2 m at a throw-in, less ENCROACH_TOL_M for players
    edging in and tracking error) is a stoppage that ends when the ball leaves the spot. It starts
    at the ball's last contact with the relevant line (goal line for goal kicks, corners and
    goals; touchline for throw-ins) if no play (dribble or pass chain) happened since, within
    BACKTRACK_MAX_S; otherwise where the rest starts. Goal kicks and corners need that line
    contact -- a keeper waiting with the ball in his area during play looks like a goal kick
    otherwise. Spot sizes are the Laws' plus SPOT_TOL_M for placement and tracking.
  * A stoppage caused by the ball going out starts when the ball crosses the line. If the ball
    goes out within ALIGN_S of where the decoder starts a stoppage (before or after), the start
    moves to the frame the ball crossed the line: the decoder's start comes from pace and
    lags, or leads, the actual event.
  * After the ball goes out or a set piece is tagged, play restarts only with a restart: the ball
    is put at rest (placed, or held for a throw-in) and leaves that spot. If, within MAX_HOLD_S
    of the stoppage's decoded end and before any play evidence, the ball rests for REST_MIN_S
    and departs, the stoppage lasts until that departure. A ball carried back onto the pitch by
    hand is not a restart.
  * Play restarts when the ball is played, not when the players start moving. If the ball sits
    still on a spot at the end of a stoppage (a free kick, a corner, a goal kick, a kickoff), the
    stoppage lasts until it leaves that spot. "Still": seen on at least STILL_MIN_SEEN of the last
    STILL_WINDOW_S and never more than STILL_M from where it rests. "Leaves": first seen more than
    MOVE_M away; the restart is the frame after it was last seen on the spot. If it is never seen
    leaving within MAX_HOLD_S, nothing changes (more likely a stuck track than a 30 s set piece).
    The rule only ever extends a stoppage; it is applied before the next one.
  * A stoppage is never shorter than MIN_STOP_S. Shorter dead runs are returned to live play.

Every constant below is a prior chosen from physics or football. None was fitted to labels. The
public 5-minute ground truth was looked at once, to understand what the organisers mean by a dead
ball (where a stoppage starts and ends), as CHALLENGE.md invites. It is never used to choose these
numbers (RULES.md), and this package does not read it.

The detector uses no frame numbers of this game. Graded halves come from games.half_video_blocks,
published boundaries the organisers explicitly allow; every decision inside them comes from data.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from .loader import load_match  # noqa: E402

FPS = 25

# Params -- priors, not fitted. See module docstring.
SPEED_SMOOTH_S = 1.0        # centred window for the collective-speed signal
WALK_SPEED = 1.6            # m/s; collective pace below this looks like a stoppage
SPEED_WEIGHT = 1.5          # log-odds per m/s away from WALK_SPEED
SPEED_CLIP = 2.0            # cap so no single term dominates

BALL_GAP_FILL = 10          # frames; ball gaps up to 0.4 s are linearly interpolated
OUT_MARGIN_NEAR = 0.5       # m beyond the near touchline (y = width)
OUT_MARGIN_OTHER = 1.5      # m beyond the far touchline and both goal lines (aerial projection)
OUT_MIN_FRAMES = 5          # ball must stay out this long (0.2 s) to count, filters glitches
OUT_WEIGHT = 3.0

MISSING_WEIGHT = 0.3

FAST_BALL = 8.0             # m/s; a pass or a shot
FAST_WEIGHT = -1.0

LOCAL_R_M = 15.0            # m; "players around the ball"
LOCAL_MIN_PLAYERS = 3       # tracked outfield players needed there to use their pace

PLAY_MIN_MS = 2.0           # m/s; faster than a ball carried in the hands at walking pace
CARRY_M = 1.5               # m; a player this close is playing the ball
CARRY_SEP_M = 1.0           # m; the nearest opponent at least this much further away
DRIBBLE_WEIGHT = -0.8

TOUCH_M = 1.5               # m; ball this close to a player counts as his touch
PASS_MIN_M = 5.0            # m; ball travel between two team-mates' touches to call it a pass
PASS_MAX_S = 3.0            # s; at most this long between the two touches
CHAIN_MIN_PASSES = 2        # consecutive passes by one team that make a chain
CHAIN_WEIGHT = -2.0
PLAY_TERMS = ("dribble", "chain", "keeper")

ANCHOR_S = 2.0              # s; a strong trigger this close to a stoppage protects it

PEN_DEPTH_M = 16.5          # Law 1: penalty area depth from the goal line
PEN_HALF_WIDTH_M = 20.16    # Law 1: half its width (40.32 m)
KEEPER_M = 1.5              # m; a goalkeeper this close has the ball
KEEPER_FROM_PLAY_S = 3.0    # no ball-out in the 3 s before: the keeper got the ball in play
KEEPER_HOLD_MAX_S = 8.0     # Law 12: a keeper may hold the ball in his hands at most 8 s

ALIGN_S = 5.0               # s; ball out this close to a decoded stoppage start re-times it

REST_SPOT_MIN_S = 1.0       # s; ball at rest on a restart spot at least this long
SPOT_TOL_M = 1.5            # m; placement and tracking tolerance around the Laws' spots
GOAL_AREA_DEPTH_M = 5.5     # Law 1
GOAL_AREA_HALF_WIDTH_M = 9.16
PENALTY_SPOT_M = 11.0       # Law 1
CORNER_ARC_M = 1.0          # Law 1
THROW_BAND_M = 2.0          # m either side of a touchline: the ball is held overhead, and the
                            # camera projects a raised ball off its true position
OPP_DIST_M = 9.15           # Laws 8, 13, 14, 16, 17
THROW_OPP_DIST_M = 2.0      # Law 15
ENCROACH_TOL_M = 1.5        # players edging in, and tracking error
BACKTRACK_MAX_S = 300.0     # longest stoppage worth looking back over (VAR reviews included)
GOAL_KICKOFF_S = 10.0       # a kickoff formation ending this close to a GOALS tag's end confirms it
KICKER_M = 2.0              # m; Law 8: the kicker stands at the ball on the centre mark
SHOT_RANGE_M = 25.0         # m from the goal line: a shot, not a long ball
SHOT_MS = FAST_BALL         # a shot is at least as fast as a pass
REST_MS = 1.0               # m/s; a ball slower than this is at rest (placed or held)
REST_MIN_S = 0.5            # s; at rest at least this long to count as put down for a restart

SETPIECE_BEFORE_S = 4.0     # evidence window around a SET PIECES tag start
SETPIECE_AFTER_S = 2.0
SETPIECE_WEIGHT = 1.0
TAG_REST_BEFORE_S = 4.0     # look for the ball put down from this long before a tag's start ...
TAG_REST_AFTER_S = 5.0      # ... to this long after it: the same window model.RESTART_TAG_S uses to tie a tag to its restart; longer reaches the next set piece
TAG_UNCONFIRMED_WEIGHT = 0.5  # a tag the tracking can neither confirm nor contradict

KICKOFF_MIN_PLAYERS = 8     # tracked outfield players per team needed to judge a formation
KICKOFF_TOL_M = 1.0         # m a player may stand over the halfway line and still count
KICKOFF_MIN_S = 1.0         # a formation must hold this long to count
KICKOFF_RELEASE_S = 1.0     # last part of a formation run carries no evidence (kick happens there)
KICKOFF_WEIGHT = 3.0

SWITCH_COST = 30.0          # log-odds paid per live<->dead switch in the Viterbi
MIN_STOP_S = 3.0            # shortest plausible stoppage

STILL_WINDOW_S = 1.0        # look this far back from the end of a stoppage for a ball at rest
STILL_MIN_SEEN = 0.6        # share of that window the ball must be seen in
STILL_M = 0.5               # m; a ball at rest stays within tracking jitter of its spot
MOVE_M = 0.5                # m; further than this from the spot and it has been played
MAX_HOLD_S = 30.0           # longest a stoppage is held open waiting for the ball to move


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def _runs(mask):
    """[(start, end_exclusive)] of True runs."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _nan_smooth(x, window):
    """Centred moving average that ignores NaN. NaN only where the whole window is NaN."""
    if window <= 1:
        return x.astype(np.float64)
    ok = ~np.isnan(x)
    k = np.ones(window)
    num = np.convolve(np.where(ok, x, 0.0), k, mode="same")
    den = np.convolve(ok.astype(np.float64), k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _fill_short_gaps(x, max_gap):
    """Linear interpolation across NaN runs of at most max_gap frames, bounded on both sides."""
    x = x.astype(np.float64).copy()
    for s, e in _runs(np.isnan(x)):
        if s == 0 or e == len(x) or e - s > max_gap:
            continue
        x[s:e] = np.interp(np.arange(s, e), [s - 1, e], [x[s - 1], x[e]])
    return x


def half_slices(m):
    """[(half, slice)] over the graded halves."""
    out = []
    for h in (1, 2):
        idx = np.flatnonzero(m.half == h)
        if len(idx):
            out.append((h, slice(idx[0], idx[-1] + 1)))
    return out


# ---------------------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------------------

def ball_track(m):
    """(x_m, y_m, speed_m_s) per frame, short gaps filled, computed per half. NaN elsewhere."""
    n = m.n_frames
    L, W = m.pitch
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    for _, sl in half_slices(m):
        bx[sl] = _fill_short_gaps(m.ball[sl, 0] * L, BALL_GAP_FILL)
        by[sl] = _fill_short_gaps(m.ball[sl, 1] * W, BALL_GAP_FILL)
    v = np.full(n, np.nan)
    v[1:-1] = np.hypot(bx[2:] - bx[:-2], by[2:] - by[:-2]) * FPS / 2.0
    return bx, by, _nan_smooth(v, 5)


def collective_speed(m):
    """Mean speed (m/s) of tracked outfield players, smoothed over SPEED_SMOOTH_S."""
    spd = m.players[:, ~m.is_goalkeeper, 2].astype(np.float64)
    n_seen = (~np.isnan(spd)).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(n_seen > 0, np.nansum(spd, axis=1) / n_seen, np.nan)
    return _nan_smooth(raw, int(round(SPEED_SMOOTH_S * FPS)))


def ball_out(bx, by, pitch):
    """Ball beyond the lines by the side-dependent margin, for at least OUT_MIN_FRAMES."""
    L, W = pitch
    with np.errstate(invalid="ignore"):
        out = ((by > W + OUT_MARGIN_NEAR) | (by < -OUT_MARGIN_OTHER)
               | (bx < -OUT_MARGIN_OTHER) | (bx > L + OUT_MARGIN_OTHER))
    keep = np.zeros_like(out)
    for s, e in _runs(out):
        if e - s >= OUT_MIN_FRAMES:
            keep[s:e] = True
    return keep


TAG_SPOTS = {"Throw-in": {3}, "Corner Kick": {2}, "Goal Kick": {1},
             "Kick Off (Start)": {5}, "Kick Off (After Goal)": {5}}   # Free Kick: anywhere


def validate_setpieces(m, bx, by, bv):
    """Check each Smart Tagging SET PIECES instance against where the ball is put down.

    Looks for the ball at rest (REST_MS, REST_MIN_S) from TAG_REST_BEFORE_S before the tag's
    start to TAG_REST_AFTER_S after it, and asks whether the last such rest is on a spot where the
    tag's type is taken (spot_of: touchline for a throw-in, corner arc, goal area, centre spot;
    a free kick may be anywhere). Returns (weight per frame, [tag dicts with "status"]):
      confirmed     rest on a matching spot  -> weight 1, a strong trigger
      unconfirmed   no rest seen             -> weight TAG_UNCONFIRMED_WEIGHT, evidence only
      contradicted  rest on another spot     -> weight 0, ignored
    The rest, not any sighting: a ball seen in flight after the kick is far from the spot.
    """
    n = m.n_frames
    weight = np.zeros(n)
    with np.errstate(invalid="ignore"):
        still = ~np.isnan(bx) & (np.nan_to_num(bv, nan=99.0) < REST_MS)
    rests = [(a, b) for a, b in _runs(still) if b - a >= int(REST_MIN_S * FPS)]
    before, after = int(SETPIECE_BEFORE_S * FPS), int(SETPIECE_AFTER_S * FPS)
    lo_w, hi_w = int(TAG_REST_BEFORE_S * FPS), int(TAG_REST_AFTER_S * FPS)
    out = []
    for t in m.tags:
        if t["role"] != "set_piece":
            continue
        s = t["start_frame"]
        kind_ok = TAG_SPOTS.get((t["tags"].get("Type") or "").strip())
        mine = [(a, b) for a, b in rests if s - lo_w <= a <= s + hi_w]
        if not mine:
            status, w = "unconfirmed", TAG_UNCONFIRMED_WEIGHT
        else:
            a, b = mine[-1]
            kind, _ = spot_of(float(np.median(bx[a:b])), float(np.median(by[a:b])), m.pitch)
            ok = kind_ok is None or kind in kind_ok
            status, w = ("confirmed", 1.0) if ok else ("contradicted", 0.0)
        sl = slice(max(0, s - before), min(n, s + after))
        weight[sl] = np.maximum(weight[sl], w)
        out.append(dict(t, status=status))
    return weight, out


def kickoff_formation(m):
    """Runs of frames where each team's tracked outfield players all stand in their own half.

    Either orientation is accepted, so nothing depends on which way a team attacks in a half.
    Returns [(start, end_exclusive)] of runs at least KICKOFF_MIN_S long.
    """
    L = m.pitch[0]
    x = m.players[:, :, 0].astype(np.float64) * L
    outfield = ~m.is_goalkeeper
    home, away = x[:, (m.team == 0) & outfield], x[:, (m.team == 1) & outfield]
    n_home, n_away = (~np.isnan(home)).sum(1), (~np.isnan(away)).sum(1)
    with np.errstate(invalid="ignore"):
        home_left = (home < L / 2 + KICKOFF_TOL_M).sum(1) == n_home
        home_right = (home > L / 2 - KICKOFF_TOL_M).sum(1) == n_home
        away_left = (away < L / 2 + KICKOFF_TOL_M).sum(1) == n_away
        away_right = (away > L / 2 - KICKOFF_TOL_M).sum(1) == n_away
    enough = (n_home >= KICKOFF_MIN_PLAYERS) & (n_away >= KICKOFF_MIN_PLAYERS)
    form = enough & ((home_left & away_right) | (home_right & away_left))
    return [(s, e) for s, e in _runs(form) if e - s >= KICKOFF_MIN_S * FPS]


def kickoff_evidence(m, runs):
    """Formation frames minus each run's last KICKOFF_RELEASE_S."""
    mask = np.zeros(m.n_frames, dtype=bool)
    release = int(KICKOFF_RELEASE_S * FPS)
    for s, e in runs:
        mask[s:max(s, e - release)] = True
    return mask


def in_penalty_area(x, y, pitch):
    """(n,) int8: 0 in the area at x = 0, 1 in the area at x = length, -1 elsewhere or NaN."""
    L, W = pitch
    with np.errstate(invalid="ignore"):
        wide = np.abs(y - W / 2) <= PEN_HALF_WIDTH_M
        left = wide & (x >= -2) & (x <= PEN_DEPTH_M)
        right = wide & (x <= L + 2) & (x >= L - PEN_DEPTH_M)
    return np.where(left, 0, np.where(right, 1, -1)).astype(np.int8)


def goalkeeper_teams(m):
    """Per goalkeeper slot, (n,) int8 effective team and (n,) bool "in his own penalty area".

    Each team's keeper defends one goal per half, read off the median position of its slot. A
    goalkeeper slot standing in a penalty area is the keeper of the team defending that goal,
    whatever team the tracker gave it (the tracker sometimes follows the other keeper).
    """
    L, W = m.pitch
    gk = np.flatnonzero(m.is_goalkeeper)
    eff = {g: np.full(m.n_frames, m.team[g], dtype=np.int8) for g in gk}
    own = {g: np.zeros(m.n_frames, dtype=bool) for g in gk}
    m.defends_left = np.full(m.n_frames, -1, dtype=np.int8)   # team defending the goal at x = 0
    for _, sl in half_slices(m):
        defends = {}                                   # goal (0 left, 1 right) -> team
        for g in gk:
            med = np.nanmedian(m.players[sl, g, 0] * L)
            if not np.isnan(med):
                defends[0 if med < L / 2 else 1] = int(m.team[g])
        if len(defends) != 2:
            continue                                   # cannot tell the ends apart: leave as is
        m.defends_left[sl] = defends[0]
        for g in gk:
            area = in_penalty_area(m.players[sl, g, 0] * L, m.players[sl, g, 1] * W, m.pitch)
            team = np.where(area == 0, defends[0], np.where(area == 1, defends[1], m.team[g]))
            eff[g][sl] = team
            own[g][sl] = area >= 0
    return eff, own


def around_ball(m, bx, by, gk_team=None):
    """Per frame, from the players' positions relative to the ball (NaN / -1 where it is not seen):

    slot, dist     nearest tracked player (any team, goalkeepers included) and his distance, m
    team           that player's team (goalkeepers: their effective team, see goalkeeper_teams)
    d_home, d_away each team's nearest player's distance, m
    local_speed    mean speed of tracked outfield players within LOCAL_R_M (NaN if fewer than
                   LOCAL_MIN_PLAYERS)
    """
    L, W = m.pitch
    n = m.n_frames
    slot = np.full(n, -1, dtype=np.int16)
    near_team = np.full(n, -1, dtype=np.int8)
    dist = np.full(n, np.nan)
    d_team = np.full((2, n), np.nan)
    local = np.full(n, np.nan)
    outfield = ~m.is_goalkeeper
    seen = np.flatnonzero(~np.isnan(bx))
    for chunk in np.array_split(seen, 20):
        px = m.players[chunk, :, 0].astype(np.float64) * L
        py = m.players[chunk, :, 1].astype(np.float64) * W
        d = np.hypot(px - bx[chunk, None], py - by[chunk, None])
        filled = np.where(np.isnan(d), np.inf, d)
        best = filled.argmin(axis=1)
        best_d = filled[np.arange(len(chunk)), best]
        ok = np.isfinite(best_d)
        teams = np.tile(m.team, (len(chunk), 1))
        for g, team in (gk_team or {}).items():
            teams[:, g] = team[chunk]
        slot[chunk[ok]], dist[chunk[ok]] = best[ok], best_d[ok]
        near_team[chunk[ok]] = teams[np.arange(len(chunk)), best][ok]
        for t in (0, 1):
            dt = np.where(teams == t, filled, np.inf).min(axis=1)
            d_team[t, chunk] = np.where(np.isfinite(dt), dt, np.nan)
        near = (filled <= LOCAL_R_M) & outfield[None, :]
        spd = np.where(near, m.players[chunk, :, 2], np.nan)
        cnt = (~np.isnan(spd)).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.nansum(spd, axis=1) / cnt
        local[chunk] = np.where(cnt >= LOCAL_MIN_PLAYERS, mean, np.nan)
    return {"slot": slot, "team": near_team, "dist": dist, "d_home": d_team[0],
            "d_away": d_team[1], "local_speed": local}


def keeper_possession(m, around, bx, out, gk_team, gk_own, by=None, bv=None):
    """(held, team, has_ball): has_ball -- frames where a goalkeeper has the ball in his own area
    after getting it in play, including frames where the ball is lost in his hands: right after
    he had it, and -- with by -- a whole gap that ends with the ball reappearing with him after
    it was last seen in his area (see the module docstring). held -- the same, but within a gap
    only its first KEEPER_HOLD_MAX_S (Law 12): the frames on which play is taken to be live.
    team -- his team on has_ball frames."""
    n = m.n_frames
    seen = ~np.isnan(bx)
    with_keeper = np.zeros(n, dtype=bool)
    team = np.full(n, -1, dtype=np.int8)
    for g, own in gk_own.items():
        hit = seen & (around["slot"] == g) & (around["dist"] <= KEEPER_M) & own
        with_keeper |= hit
        team[hit] = gk_team[g][hit]
    back = int(KEEPER_FROM_PLAY_S * FPS)
    cum_out = np.r_[0, np.cumsum(out)]
    lost = _runs(~seen)
    if by is not None:                                 # ball reappears with the keeper
        area = in_penalty_area(bx, by, m.pitch)
        for s, e in lost:
            if s == 0 or e >= n or not with_keeper[e] or area[s - 1] < 0 \
                    or area[s - 1] != area[e]:
                continue
            if spot_of(bx[e], by[e], m.pitch)[0] == 1:   # goal area: may be a goal kick (Law 16)
                continue
            if bv is not None and bv[s - 1] <= REST_MS and spot_of(bx[s - 1], by[s - 1],
                                                                   m.pitch)[0] != 0:
                continue                                         # lost while placed for a restart
            if cum_out[s] - cum_out[max(0, s - back)] > 0:      # came to him from outside play
                continue
            with_keeper[s:e] = True
            team[s:e] = team[e]
    hold = int(KEEPER_HOLD_MAX_S * FPS)
    for s, e in lost:                                  # ball lost right after a keeper had it
        if s > 0 and with_keeper[s - 1] and not with_keeper[s]:
            k = min(e, s + hold)
            with_keeper[s:k] = True
            team[s:k] = team[s - 1]
    for s, e in _runs(with_keeper):
        if cum_out[s] - cum_out[max(0, s - back)] > 0:  # came to him from outside play
            with_keeper[s:e] = False
            team[s:e] = -1
    # Law 12: in play he may hold it at most KEEPER_HOLD_MAX_S. Who has the ball holds for the
    # whole gap; that play is live is only granted for that long (a whistle while he holds it --
    # offside, a foul -- is not in the tracking).
    held = with_keeper.copy()
    for s, e in lost:
        held[s + hold:e] = False
    return held, team, with_keeper


def pass_chains(slot, team_of_frame, bx, by, breaks=None):
    """Frames covered by chains of CHAIN_MIN_PASSES+ consecutive passes by one team.

    slot is the player within TOUCH_M of the ball per frame (-1 if none; the caller leaves frames
    with the ball out of the pitch at -1); team_of_frame is that player's team per frame. A touch is a run of one slot (short interruptions of
    the same slot are one touch). A pass is a touch followed by a team-mate's touch (another slot
    of the same team) within PASS_MAX_S with the ball at least PASS_MIN_M from where it left, and
    no `breaks` frame (ball out, set piece) from the one touch to the next.
    Returns (mask, [(start, end_exclusive, n_passes)], [(release, reception, team)]) -- the last
    list has every single completed pass: the frame the ball left the passer, the frame the
    team-mate first touched it, and their team.
    """
    if breaks is None:
        breaks = np.zeros(len(slot), dtype=bool)
    cum_break = np.r_[0, np.cumsum(breaks)]
    n = len(slot)
    change = np.flatnonzero(np.diff(slot) != 0) + 1
    bounds = np.r_[0, change, n]
    touches = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        k = int(slot[a])
        if k < 0:
            continue
        if touches and touches[-1][2] == k and a - touches[-1][1] <= PASS_MAX_S * FPS:
            touches[-1][1] = b                       # same player again: still one touch
        else:
            touches.append([a, b, k, int(team_of_frame[a])])
    gap = PASS_MAX_S * FPS
    is_pass = []
    for (a0, b0, k0, t0), (a1, b1, k1, t1) in zip(touches[:-1], touches[1:]):
        travel = np.hypot(bx[a1] - bx[b0 - 1], by[a1] - by[b0 - 1])
        broken = cum_break[a1 + 1] - cum_break[b0 - 1] > 0
        is_pass.append(k0 != k1 and t0 == t1
                       and a1 - b0 <= gap and travel >= PASS_MIN_M and not broken)
    mask = np.zeros(n, dtype=bool)
    chains = []
    i = 0
    while i < len(is_pass):
        if not is_pass[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(is_pass) and is_pass[j + 1]:
            j += 1
        if j - i + 1 >= CHAIN_MIN_PASSES:
            start, end = touches[i][0], touches[j + 1][1]
            mask[start:end] = True
            chains.append((start, end, j - i + 1))
        i = j + 1
    passes = [(touches[i][1] - 1, touches[i + 1][0], touches[i][3])
              for i, ok in enumerate(is_pass) if ok]
    return mask, chains, passes


def features(m):
    bx, by, bv = ball_track(m)
    runs = kickoff_formation(m)
    gk_team, gk_own = goalkeeper_teams(m)
    around = around_ball(m, bx, by, gk_team)
    speed_all = collective_speed(m)
    speed_local = _nan_smooth(around["local_speed"], int(round(SPEED_SMOOTH_S * FPS)))
    pace = speed_all                       # local pace: display only, see the docstring
    out = ball_out(bx, by, m.pitch)

    d_h = np.nan_to_num(around["d_home"], nan=np.inf)
    d_a = np.nan_to_num(around["d_away"], nan=np.inf)
    with np.errstate(invalid="ignore"):
        v = np.nan_to_num(bv)
        dribble = (~np.isnan(bx) & ~out & (v >= PLAY_MIN_MS) & (v < FAST_BALL)
                   & (np.minimum(d_h, d_a) <= CARRY_M)
                   & (np.abs(d_h - d_a) >= CARRY_SEP_M))
    touch_slot = np.where((np.nan_to_num(around["dist"], nan=np.inf) <= TOUCH_M) & ~out,
                          around["slot"], -1)
    setpiece, setpiece_tags = validate_setpieces(m, bx, by, bv)
    breaks = out.copy()
    for t in setpiece_tags:
        if t["status"] != "contradicted" and 0 <= t["start_frame"] < m.n_frames:
            breaks[t["start_frame"]] = True
    chain, chains, passes = pass_chains(touch_slot, around["team"], bx, by, breaks)
    keeper, keeper_team, keeper_poss = keeper_possession(m, around, bx, out, gk_team, gk_own,
                                                         by, bv)
    return {
        "speed": pace, "speed_all": speed_all, "speed_local": speed_local,
        "ball_x": bx, "ball_y": by, "ball_speed": bv,
        "d_home": around["d_home"], "d_away": around["d_away"],
        "dribble": dribble, "chain": chain, "chains": chains, "passes": passes,
        "keeper": keeper, "keeper_team": keeper_team, "keeper_poss": keeper_poss,
        "near_team": around["team"],
        "gk_team": gk_team,
        "out": out,
        "missing": np.isnan(bx),
        "setpiece": setpiece, "setpiece_tags": setpiece_tags,
        "kickoff_runs": runs,
        "kickoff": kickoff_evidence(m, runs),
    }


# ---------------------------------------------------------------------------------------------
# evidence and decoding
# ---------------------------------------------------------------------------------------------

# The on/off evidence terms (speed is continuous and always on), for explanations.
FLAG_TERMS = ("out", "kickoff", "setpiece", "missing", "fast", "dribble", "chain", "keeper")


def evidence_terms(f):
    """Each evidence term separately, in log-odds (dead vs live). Useful for debugging."""
    speed = np.nan_to_num(f["speed"], nan=WALK_SPEED)   # nobody tracked -> no opinion
    inside = ~f["out"] & ~f["missing"]
    with np.errstate(invalid="ignore"):
        fast = inside & (np.nan_to_num(f["ball_speed"]) > FAST_BALL)
    return {
        "speed": np.clip(SPEED_WEIGHT * (WALK_SPEED - speed), -SPEED_CLIP, SPEED_CLIP),
        "out": OUT_WEIGHT * f["out"],
        "missing": MISSING_WEIGHT * f["missing"],
        "fast": FAST_WEIGHT * fast,
        "dribble": DRIBBLE_WEIGHT * f["dribble"],
        "chain": CHAIN_WEIGHT * f["chain"],
        # cancels the dead side of speed and missing on keeper frames (see the docstring)
        "keeper": -1.0 * f["keeper"] * (np.maximum(np.clip(SPEED_WEIGHT * (WALK_SPEED - speed),
                                                     -SPEED_CLIP, SPEED_CLIP), 0)
                                  + MISSING_WEIGHT * f["missing"]),
        "setpiece": SETPIECE_WEIGHT * f["setpiece"],
        "kickoff": KICKOFF_WEIGHT * f["kickoff"],
    }


def dead_evidence(m, f=None):
    """Per-frame log-likelihood ratio dead vs live. 0 outside the graded halves."""
    f = f or features(m)
    llr = sum(evidence_terms(f).values())
    llr[m.half == 0] = 0.0
    return llr


def viterbi_two_state(llr, switch_cost):
    """Most likely live(0)/dead(1) path for per-frame log-odds llr, paying switch_cost per change."""
    n = len(llr)
    back = np.zeros((n, 2), dtype=np.int8)
    score = np.array([0.0, llr[0]])
    for t in range(1, n):
        stay0, move0 = score[0], score[1] - switch_cost
        stay1, move1 = score[1], score[0] - switch_cost
        back[t, 0] = 0 if stay0 >= move0 else 1
        back[t, 1] = 1 if stay1 >= move1 else 0
        score = np.array([max(stay0, move0), max(stay1, move1) + llr[t]])
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(score[1] > score[0])
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path.astype(bool)


# Why detect_dead decided what it did, per frame (return_reason=True).
LIVE_VITERBI, DEAD_VITERBI, DEAD_PREKICKOFF, LIVE_SHORT_STOP, DEAD_BALL_STILL = 0, 1, 2, 3, 4
DEAD_ANCHORED = 5
DEAD_UNTIL_RESTART, DEAD_OUT_ALIGN, LIVE_OUT_ALIGN = 6, 7, 8
DEAD_RESTART_SPOT = 9
DEAD_GOAL = 10
SPOT_NAMES = {1: "kale vuruşu", 2: "korner", 3: "taç", 4: "penaltı", 5: "santra"}


def crossing_frame(o, bx, by, pitch):
    """First frame of the run, ending at out-frame o, in which the ball is beyond a line."""
    L, W = pitch
    with np.errstate(invalid="ignore"):
        beyond = (bx < 0) | (bx > L) | (by < 0) | (by > W)
    c = o
    while c > 0 and beyond[c - 1]:
        c -= 1
    return c


def align_to_out(dead, out, bx, by, pitch):
    """Re-time stoppage starts to the ball crossing the line when it goes out within ALIGN_S.

    Returns (dead, made_dead, made_live) masks for one half.
    """
    dead = dead.copy()
    made_dead = np.zeros_like(dead)
    made_live = np.zeros_like(dead)
    w = int(ALIGN_S * FPS)
    out_starts = [s for s, _ in _runs(out)]
    for s, e in _runs(dead.copy()):
        near = [o for o in out_starts if s - w <= o <= min(e - 1, s + w)]
        if not near or s == 0:
            continue
        c = crossing_frame(near[0], bx, by, pitch)
        if c < s:
            made_dead[c:s] = ~dead[c:s]
            dead[c:s] = True
        elif c > s:
            made_live[s:c] = True
            dead[s:c] = False
    return dead, made_dead, made_live


def hold_until_restart(dead, anchored, bx, ball_speed, play):
    """Extend anchored stoppages until the ball is put at rest and leaves its spot.

    For each stoppage [s, e) with anchored[k] True: look from e for up to MAX_HOLD_S, but only
    before the first play-evidence frame; take the last rest (ball seen, slower than REST_MS for
    REST_MIN_S) that starts in that window; the restart is the frame after it. All one half.
    Returns (dead, [(old_end, new_end)]).
    """
    dead = dead.copy()
    n = len(dead)
    hold, rest_min = int(MAX_HOLD_S * FPS), int(REST_MIN_S * FPS)
    with np.errstate(invalid="ignore"):
        still = ~np.isnan(bx) & (np.nan_to_num(ball_speed, nan=99.0) < REST_MS)
    rests = [(a, b) for a, b in _runs(still) if b - a >= rest_min]
    moves = []
    for (s, e), anc in zip(_runs(dead.copy()), anchored):
        if not anc or e >= n:
            continue
        stop = min(n, e + hold)
        p = np.flatnonzero(play[e:stop])
        stop = e + int(p[0]) if len(p) else stop
        cand = [(a, b) for a, b in rests if s <= a < stop and b > e and b < n]
        if not cand:
            continue
        restart = cand[-1][1]
        if restart > e:
            dead[e:restart] = True
            moves.append((e, restart))
    return dead, moves


def anchors(m, f):
    """(n,) bool: frames with a strong stoppage trigger -- ball out, or the start of a set piece
    tag confirmed by the ball put down on a matching spot (validate_setpieces)."""
    a = f["out"].copy()
    for t in f["setpiece_tags"]:
        if t["status"] == "confirmed" and 0 <= t["start_frame"] < m.n_frames:
            a[t["start_frame"]] = True
    return a


def restore_anchored(d_full, d_base, anchor, pad):
    """Stoppages of d_base that d_full erased entirely and that have an anchor within pad frames.

    All arrays cover one half. Returns the mask of frames to restore.
    """
    restore = np.zeros_like(d_full)
    for s, e in _runs(d_base):
        if d_full[s:e].any():
            continue                       # shortened at most: allowed
        if anchor[max(0, s - pad):min(len(anchor), e + pad)].any():
            restore[s:e] = True
    return restore


def hold_until_ball_moves(dead, bx, by):
    """Extend each stoppage whose ball is at rest until the ball leaves its spot.

    dead, bx, by cover one half (bx, by in metres, NaN where the ball is not seen). Returns the new
    dead mask and [(old_restart, new_restart)] for every stoppage that was extended.
    """
    dead = dead.copy()
    n = len(dead)
    window, hold = int(STILL_WINDOW_S * FPS), int(MAX_HOLD_S * FPS)
    moves = []
    for s, e in _runs(dead):
        if e >= n or not dead[e - 1] or dead[e]:
            continue                       # half ends dead, or an earlier hold already covered it
        w0 = max(s, e - window)
        seen = ~np.isnan(bx[w0:e])
        if seen.mean() < STILL_MIN_SEEN:
            continue                       # cannot tell whether the ball is at rest
        rest_x, rest_y = np.median(bx[w0:e][seen]), np.median(by[w0:e][seen])
        dist = np.hypot(bx[w0:min(n, e + hold)] - rest_x, by[w0:min(n, e + hold)] - rest_y)
        if np.nanmax(dist[:e - w0]) > STILL_M:
            continue                       # the ball is moving already: nothing to wait for
        after = dist[e - w0:]
        with np.errstate(invalid="ignore"):
            away = np.flatnonzero(after > MOVE_M)
            if not len(away):
                continue                   # never seen leaving: leave the stoppage alone
            on_spot = np.flatnonzero(after[:away[0]] <= MOVE_M)
        restart = e + (int(on_spot[-1]) + 1 if len(on_spot) else 0)
        if restart > e:
            dead[e:restart] = True
            moves.append((e, restart))
    return dead, moves


def spot_of(x, y, pitch):
    """(kind, goal side) of the restart spot at (x, y): kind 1 goal area, 2 corner, 3 touchline,
    4 penalty spot, 5 centre spot, 0 none. Goal side 0 (x = 0) or 1 (x = length), -1 if none."""
    L, W = pitch
    side = 0 if x < L / 2 else 1
    gx = x if side == 0 else L - x                     # distance from that goal line, inwards
    if np.hypot(x - L / 2, y - W / 2) <= SPOT_TOL_M:
        return 5, -1
    if np.hypot(gx - PENALTY_SPOT_M, y - W / 2) <= SPOT_TOL_M:
        return 4, side
    if min(np.hypot(gx, y), np.hypot(gx, y - W)) <= CORNER_ARC_M + SPOT_TOL_M:
        return 2, side
    if -SPOT_TOL_M <= gx <= GOAL_AREA_DEPTH_M + SPOT_TOL_M and \
            abs(y - W / 2) <= GOAL_AREA_HALF_WIDTH_M + SPOT_TOL_M:
        return 1, side
    if min(abs(y), abs(y - W)) <= THROW_BAND_M and -SPOT_TOL_M <= x <= L + SPOT_TOL_M:
        return 3, -1
    return 0, -1


def opponents_placed(m, f, frames, kind, side, bx, by):
    """True if, on most of `frames`, the taker's opponents stand where the Laws put them."""
    L, W = m.pitch
    ok = []
    for t in frames:
        taker = int(f["near_team"][t])
        teams = m.team.astype(np.int8).copy()
        for g, team in f["gk_team"].items():
            teams[g] = team[t]
        if kind != 1 and taker < 0:
            continue
        px, py = m.players[t, :, 0] * L, m.players[t, :, 1] * W
        opp = (teams != taker) & ~np.isnan(px)
        if kind == 1:
            # Goal kick: the attackers of this goal outside the penalty area (Law 16). Attackers
            # are the team NOT defending it -- not "not the nearest player's team", because the
            # tracker can give the keeper at the ball the wrong team. One player is allowed in
            # for that same reason (an identity swap), or a player still walking out.
            defender = int(m.defends_left[t]) if side == 0 else 1 - int(m.defends_left[t])
            if int(m.defends_left[t]) < 0:
                continue
            att = (teams != defender) & ~np.isnan(px)
            gx = px[att] if side == 0 else L - px[att]
            inside = (gx <= PEN_DEPTH_M) & (np.abs(py[att] - W / 2) <= PEN_HALF_WIDTH_M)
            ok.append(int(inside.sum()) <= 1)
            continue
        if not opp.any():
            continue
        need = THROW_OPP_DIST_M if kind == 3 else OPP_DIST_M
        d = np.hypot(px[opp] - bx[t], py[opp] - by[t])
        ok.append(d.min() >= need - ENCROACH_TOL_M)
    return len(ok) > 0 and np.mean(ok) >= 0.5


def restart_spots(m, f, sl):
    """Stoppages implied by a ball at rest on a restart spot, for one half.

    Returns (mask, kind) over the half: kind per frame (SPOT_NAMES keys) where the rule made it dead.
    """
    L, W = m.pitch
    bx, by = f["ball_x"][sl], f["ball_y"][sl]
    v = f["ball_speed"][sl]
    n = len(bx)
    off = sl.start
    with np.errstate(invalid="ignore"):
        still = ~np.isnan(bx) & (np.nan_to_num(v, nan=99.0) < REST_MS)
        goal_line = (bx < 0) | (bx > L)
        touchline = (by < 0) | (by > W)
    play = f["chain"][sl]          # passes; a ball carried to the spot looks like a dribble
    mask = np.zeros(n, dtype=bool)
    kinds = np.zeros(n, dtype=np.int8)
    back_max = int(BACKTRACK_MAX_S * FPS)
    for a, b in _runs(still):
        if b - a < int(REST_SPOT_MIN_S * FPS) or b >= n:
            continue
        x, y = float(np.median(bx[a:b])), float(np.median(by[a:b]))
        kind, side = spot_of(x, y, m.pitch)
        if kind == 0:
            continue
        frames = range(off + max(a, b - FPS), off + b, 5)
        if not opponents_placed(m, f, frames, kind, side, f["ball_x"], f["ball_y"]):
            continue
        line = touchline if kind == 3 else goal_line
        start, contact = a, False
        k = a - 1
        while k >= max(0, a - back_max) and not play[k]:
            if line[k]:
                start, contact = k, True
                break
            k -= 1
        if kind in (1, 2) and not contact:
            continue                                    # a keeper waiting with the ball in play
        mask[start:b] = True
        kinds[start:b] = kind
    return mask, kinds


def kicker_at_centre(m, kickoff):
    """Law 8: at a kickoff the ball is on the centre mark and the kicker stands at it. True if,
    over the last 2 s of the formation, the median distance of the nearest player to the centre
    spot is within KICKER_M. Tells a kickoff from other moments when everyone stands in his own
    half (a drinks break, players at their benches)."""
    L, W = m.pitch
    d = []
    for t in range(max(0, kickoff - 2 * FPS), kickoff):
        dt = np.hypot(m.players[t, :, 0] * L - L / 2, m.players[t, :, 1] * W - W / 2)
        if not np.all(np.isnan(dt)):
            d.append(np.nanmin(dt))
    return len(d) > 0 and float(np.median(d)) <= KICKER_M


def goal_moment(m, f, start, kickoff, scorer):
    """The goal: after the last shot at the conceded goal (ball within SHOT_RANGE_M of that goal
    line, moving towards it at SHOT_MS or more) seen before the scoring team is back in its own
    half (half its outfield players, after the celebration), the first frame the ball, last seen
    near that goal, disappears; else that last shot. If no shot is seen, the return; if neither,
    None (the caller keeps the tag's start)."""
    L = m.pitch[0]
    conceder = 1 - scorer
    left = int(m.defends_left[kickoff])
    if left < 0:
        return None
    goal_x = 0.0 if left == conceder else L
    outfield = (m.team == scorer) & ~m.is_goalkeeper
    back = None
    for t in range(start, kickoff, FPS // 5):
        x = m.players[t, outfield, 0] * L
        x = x[~np.isnan(x)]
        if len(x) and np.mean(x < L / 2 if left == scorer else x > L / 2) >= 0.5:
            back = t
            break
    stop = back if back is not None else kickoff
    bx = f["ball_x"][start:stop]
    v = np.nan_to_num(f["ball_speed"][start:stop], nan=0.0)
    vx = np.r_[0.0, np.diff(bx)]
    with np.errstate(invalid="ignore"):
        shot = (np.abs(bx - goal_x) <= SHOT_RANGE_M) & (v >= SHOT_MS) & \
            (np.sign(goal_x - bx) == np.sign(vx))
    k = np.flatnonzero(shot)
    if not len(k):
        return back      # no shot seen: at least the celebration is over by then
    # The scoring shot itself is often unseen (crowded box): play can go on after the last shot
    # we see (a rebound, a cut-back). The goal is where the ball, last seen near that goal,
    # disappears -- into the net -- after the last seen shot; failing that, the last shot.
    near = np.abs(bx - goal_x) <= SHOT_RANGE_M
    for i in range(int(k[-1]) + 1, len(bx)):
        if np.isnan(bx[i]) and not np.isnan(bx[i - 1]) and near[i - 1]:
            return start + i
    return start + int(k[-1])


def goal_stoppages(m, f):
    """[(start, kickoff, scorer)] for each goal. A Smart Tagging GOALS instance counts only when a
    kickoff follows it: a kickoff formation (Law 8: every player in his own half) ending within
    GOAL_KICKOFF_S of the tag's end, with the kicker at the centre spot. Everything from the goal
    (goal_moment; failing that, the scoring team's return to its half, or the tag's start) to the
    kickoff is dead, however much the players move while celebrating."""
    out = []
    w = int(GOAL_KICKOFF_S * FPS)
    for t in m.tags:
        if t["code"] != "GOALS":
            continue
        ends = [e for s, e in f["kickoff_runs"] if abs(e - t["end_frame"]) <= w]
        if not ends:
            continue
        kickoff = max(ends)
        if not kicker_at_centre(m, kickoff):
            continue
        start = t["start_frame"]
        if t["side"] is not None:
            g = goal_moment(m, f, t["start_frame"], kickoff, t["side"])
            if g is not None:
                start = g
        out.append((start, kickoff, t["side"]))
    return out


def detect_dead(m, llr=None, f=None, return_reason=False):
    """(n,) bool, True where the ball is judged dead. False outside the graded halves.

    With return_reason=True also returns an (n,) int8 array: LIVE_VITERBI / DEAD_VITERBI (the
    decoder decided), DEAD_ANCHORED (strong trigger, restored), DEAD_PREKICKOFF (kickoff rule),
    DEAD_OUT_ALIGN / LIVE_OUT_ALIGN (start moved to the ball crossing the line), DEAD_UNTIL_RESTART
    (held until the ball was put down and played), DEAD_BALL_STILL (held until the ball was
    played), LIVE_SHORT_STOP (too short, undone).
    """
    f = f or features(m)
    llr = dead_evidence(m, f) if llr is None else llr
    f["goals"] = goal_stoppages(m, f)
    dead = np.zeros(m.n_frames, dtype=bool)
    reason = np.full(m.n_frames, LIVE_VITERBI, dtype=np.int8)
    min_stop = int(MIN_STOP_S * FPS)
    L, W = m.pitch
    terms = evidence_terms(f)
    play = sum(terms[k] for k in PLAY_TERMS)
    llr_base = llr - np.where(m.half > 0, play, 0.0)
    anchor = anchors(m, f)
    pad = int(ANCHOR_S * FPS)
    for _, sl in half_slices(m):
        d = viterbi_two_state(llr[sl], SWITCH_COST)
        r = np.where(d, DEAD_VITERBI, LIVE_VITERBI).astype(np.int8)

        # Rule: strong triggers protect a stoppage from being erased by play evidence.
        d_base = viterbi_two_state(llr_base[sl], SWITCH_COST)
        back = restore_anchored(d, d_base, anchor[sl], pad)
        d = d | back
        r[back] = DEAD_ANCHORED

        # Rule: a half's play starts with a kickoff. If the first formation seen in this half
        # starts inside it, everything before that formation is still the wait for the kickoff.
        # A formation that began before the half (the opening kickoff) imposes nothing.
        starts = [s for s, e in f["kickoff_runs"] if e > sl.start and s < sl.stop]
        if starts and starts[0] > sl.start:
            d[:starts[0] - sl.start] = True
            r[:starts[0] - sl.start] = DEAD_PREKICKOFF

        # Rule: a stoppage caused by the ball going out starts when it crosses the line.
        bx, by = f["ball_x"][sl], f["ball_y"][sl]
        d, made_dead, made_live = align_to_out(d, f["out"][sl], bx, by, m.pitch)
        r[made_dead] = DEAD_OUT_ALIGN
        r[made_live] = LIVE_OUT_ALIGN

        # Rule: a goal (tag + kickoff formation) is dead until the kickoff.
        for gs, ko, _ in f["goals"]:
            a, b = max(gs, sl.start) - sl.start, min(ko, sl.stop) - sl.start
            if a < b:
                r[a:b][~d[a:b]] = DEAD_GOAL
                d[a:b] = True

        # Rule: a ball waiting on a restart spot, with the opponents placed by the Laws.
        spot, spot_kind = restart_spots(m, f, sl)
        r[spot & ~d] = DEAD_RESTART_SPOT
        d = d | spot
        kinds_all = f.setdefault("spot_kind", np.zeros(m.n_frames, dtype=np.int8))
        kinds_all[sl] = np.where(spot, spot_kind, kinds_all[sl])

        # Rule: after a ball-out or a set piece, play restarts with a restart (ball put at rest,
        # then played), not with the ball carried back in.
        a_sl = anchor[sl]
        anchored = [a_sl[max(0, s - pad):min(len(a_sl), e + pad)].any() for s, e in _runs(d)]
        play_sl = (f["dribble"] | f["chain"] | f["keeper"])[sl]
        held, _ = hold_until_restart(d, anchored, bx, f["ball_speed"][sl], play_sl)
        r[held & ~d] = DEAD_UNTIL_RESTART
        d = held

        # Rule: play restarts when the ball is played. Applied before the minimum length, so a
        # short stoppage that turns out to be a set piece survives.
        held, _ = hold_until_ball_moves(d, m.ball[sl, 0] * L, m.ball[sl, 1] * W)
        r[held & ~d] = DEAD_BALL_STILL
        d = held

        # Rule: no stoppage shorter than MIN_STOP_S. The forced pre-kickoff stretch is kept.
        for s, e in _runs(d):
            if e - s < min_stop and s > 0:
                d[s:e] = False
                r[s:e] = LIVE_SHORT_STOP
        dead[sl] = d
        reason[sl] = r
    return (dead, reason) if return_reason else dead


# ---------------------------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------------------------

def describe(m, dead):
    """Counts a football person can sanity-check without any ground truth."""
    lines = []
    for h, sl in half_slices(m):
        d = dead[sl]
        stops = _runs(d)
        dur = np.array([e - s for s, e in stops]) / FPS
        lines.append("half {}: dead {:.1%} of {:.1f} min, {} stoppages, duration median {:.1f} s"
                     " / mean {:.1f} s / max {:.1f} s, {} live spells".format(
                         h, d.mean(), d.size / FPS / 60, len(stops),
                         np.median(dur) if len(dur) else 0, dur.mean() if len(dur) else 0,
                         dur.max() if len(dur) else 0, len(_runs(~d))))
    return lines



def selftest():
    """Synthetic checks of hold_until_ball_moves. No match data needed."""
    n = 400
    dead = np.zeros(n, dtype=bool)
    dead[50:150] = True                               # stoppage ends at 150
    bx, by = np.full(n, 30.0), np.full(n, 20.0)       # ball at rest on a spot...
    bx[230:] = 30.0 + 0.6 * np.arange(1, n - 229)     # ...played at frame 230
    held, moves = hold_until_ball_moves(dead, bx, by)
    assert moves == [(150, 230)] and held[229] and not held[230], moves

    bx2 = bx.copy()
    bx2[200:240] = np.nan                             # occluded at the kick: restart = first
    held, moves = hold_until_ball_moves(dead, bx2, by)  # frame after last seen on the spot
    assert moves == [(150, 200)], moves

    moving = 30.0 + 0.3 * np.arange(n)                # ball already moving: no change
    held, moves = hold_until_ball_moves(dead, moving, by)
    assert moves == [] and np.array_equal(held, dead)

    hidden = np.full(n, np.nan)                       # ball not seen: no change
    held, moves = hold_until_ball_moves(dead, hidden, hidden)
    assert moves == []

    # Pass chains: slots 0,1,2 home, 3 away. Ball moves 10 m between touches.
    team = np.array([0, 0, 0, 1])
    sl = np.full(100, -1)
    bx3 = np.linspace(0, 40, 100)
    by3 = np.zeros(100)
    sl[0:10], sl[30:40], sl[60:70] = 0, 1, 2            # two passes 0->1->2: a chain
    tf = np.where(sl >= 0, team[np.maximum(sl, 0)], -1)
    mask, chains, passes = pass_chains(sl, tf, bx3, by3)
    assert chains == [(0, 70, 2)] and mask[50] and not mask[80], chains
    assert passes == [(9, 30, 0), (39, 60, 0)], passes
    sl2 = sl.copy()
    sl2[60:70] = 3                                      # 0->1 then lost to the opponent: no chain
    tf2 = np.where(sl2 >= 0, team[np.maximum(sl2, 0)], -1)
    mask, chains, _ = pass_chains(sl2, tf2, bx3, by3)
    assert chains == [] and not mask.any(), chains
    brk = np.zeros(100, dtype=bool)
    brk[50] = True                                      # ball out between the 2nd and 3rd touch
    mask, chains, _ = pass_chains(sl, tf, bx3, by3, brk)
    assert chains == [], chains

    # Keeper gap: ball lost in flight in the right-hand area (keeper 0 not near it), unseen for
    # 12 s, reappears with keeper 0 -> the whole gap is his; not if it reappears in the goal area,
    # was last seen outside the area, or went out just before.
    from types import SimpleNamespace
    nk = 500
    mk = SimpleNamespace(n_frames=nk, pitch=(105.0, 68.0))

    def keeper_case(last=(90.0, 45.0), back=(92.0, 40.0), out_at=None):
        kx, ky = np.full(nk, np.nan), np.full(nk, np.nan)
        kx[:100], ky[:100] = last
        kx[400:], ky[400:] = back
        slot = np.where(np.arange(nk) >= 400, 0, 5)
        dist = np.where(np.arange(nk) >= 400, 0.5, 10.0)
        ko = np.zeros(nk, dtype=bool)
        if out_at is not None:
            ko[out_at:out_at + 10] = True
        own = {0: np.ones(nk, dtype=bool)}
        kt = {0: np.ones(nk, dtype=np.int8)}
        return keeper_possession(mk, {"slot": slot, "dist": dist}, kx, ko, kt, own, ky)

    kheld, kteam, kmask = keeper_case()
    assert kmask[100:400].all() and (kteam[100:400] == 1).all(), "keeper gap not back-filled"
    hold_f = int(KEEPER_HOLD_MAX_S * FPS)                # live granted for the first 8 s only
    assert kheld[100:100 + hold_f].all() and not kheld[100 + hold_f:400].any(), "Law 12 limit"
    _, _, kmask = keeper_case(back=(102.0, 34.0))
    assert not kmask[100:400].any(), "goal area: may be a goal kick"
    _, _, kmask = keeper_case(last=(60.0, 34.0))
    assert not kmask[100:400].any(), "last seen outside his area"
    _, _, kmask = keeper_case(out_at=40)
    assert not kmask[100:400].any(), "ball out just before"

    # Anchored stoppages: erased one with a trigger comes back, a shortened one is left alone.
    d_full = np.zeros(300, dtype=bool)
    d_full[210:230] = True                              # second stoppage only shortened
    d_base = np.zeros(300, dtype=bool)
    d_base[50:120] = True                               # first stoppage erased by play evidence
    d_base[200:260] = True
    anchor = np.zeros(300, dtype=bool)
    anchor[60] = anchor[205] = True
    back = restore_anchored(d_full, d_base, anchor, 50)
    assert back[50:120].all() and not back[120:].any()

    # Restart: ball out, carried back in (moving), put down at 200-240, kicked at 240.
    n4 = 400
    dead4 = np.zeros(n4, dtype=bool)
    dead4[50:150] = True
    bx4 = np.full(n4, 30.0)
    v4 = np.full(n4, 3.0)                               # moving (carried) ...
    v4[200:240] = 0.2                                   # ... at rest on the spot ...
    v4[240:] = 12.0                                     # ... then kicked
    held, moves = hold_until_restart(dead4, [True], bx4, v4, np.zeros(n4, dtype=bool))
    assert moves == [(150, 240)], moves
    play4 = np.zeros(n4, dtype=bool)
    play4[170] = True                                   # play seen before the rest: no hold
    held, moves = hold_until_restart(dead4, [True], bx4, v4, play4)
    assert moves == [], moves
    held, moves = hold_until_restart(dead4, [False], bx4, v4, np.zeros(n4, dtype=bool))
    assert moves == [], moves                           # not anchored: no hold

    # Restart spots on a 105 x 68 pitch.
    P = (105.0, 68.0)
    assert spot_of(52.5, 34.0, P) == (5, -1)            # centre spot
    assert spot_of(94.0, 34.5, P) == (4, 1)             # penalty spot, right goal
    assert spot_of(0.5, 0.5, P)[0] == 2                 # corner
    assert spot_of(5.5, 37.3, P) == (1, 0)              # goal-area line, left goal
    assert spot_of(40.0, 67.5, P) == (3, -1)            # touchline
    assert spot_of(40.0, 30.0, P) == (0, -1)            # open play

    # Out alignment: decoded start at 100, ball crosses the line at 180 (out flag from 185).
    dead5 = np.zeros(400, dtype=bool)
    dead5[100:300] = True
    bx5 = np.full(400, 10.0)
    bx5[180:] = -3.0
    out5 = np.zeros(400, dtype=bool)
    out5[185:300] = True
    d5, md, ml = align_to_out(dead5, out5, bx5, np.full(400, 30.0), (105.0, 68.0))
    assert not d5[179] and d5[180] and ml[100:180].all(), np.flatnonzero(d5)[:3]
    print("selftest ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="run the synthetic checks and exit")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()

    m = load_match(with_homography=False)
    f = features(m)
    terms = evidence_terms(f)
    llr = dead_evidence(m, f)
    dead = detect_dead(m, llr, f)

    graded = m.half > 0
    print("evidence terms, mean over graded frames (log-odds, + = dead):")
    for name, val in terms.items():
        print("  {:<9} {:+.3f}   active on {:.1%}".format(
            name, val[graded].mean(), (val[graded] != 0).mean()))
    print()
    for line in describe(m, dead):
        print(line)
    print("overall: dead on {:.1%} of graded frames".format(dead[graded].mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
