"""First end-to-end submission: our dead-ball detector + nearest player to the ball.

    python -m possession.rule_based.first_submission          # writes evaluation/results/possession_v0.csv
                                                   # and data/debug/possession_v0.explain.npz

This is a pipeline check, not a model. It answers one question: does the dead-ball detector, on its
own, move us past the published floor (baseline_smooth50)? Everything that is not the dead ball is
deliberately the baseline's own recipe, so the comparison isolates that one change.

  1. dead  = possession.rule_based.dead_ball.detect_dead(m)                      -> D
  2. live frames with a visible ball cast one vote: the team of the nearest tracked player
     (distances in metres). A frame without a ball, or without any tracked player, casts none.
  3. inside each live spell only, a centred majority vote of +/- VOTE_WINDOW frames, the window
     clipped at the spell's edges. Ties go to H. The window is the published baseline's (50
     frames), not chosen by us. Votes never cross a stoppage.
  4. a live frame with no vote in its window copies the RESULT (window majority) of the nearest
     frame in the same spell that has one; a live spell with no visible ball at all is X.
  5. frames outside the graded halves are X (never scored).

The explanation file records, per frame, exactly the quantities these steps used, so the debug
overlay shows the real reason for each label, not a reconstruction.
"""

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from .dead_ball import _runs, dead_evidence, detect_dead, features  # noqa: E402
from .explain import (SRC_DEAD, SRC_FILL, SRC_NO_BALL, SRC_UNGRADED,  # noqa: E402
                                SRC_VOTE, dead_fields, write_explain, write_submission)
from .loader import load_match  # noqa: E402

VOTE_WINDOW = 50            # frames; baseline_smooth50's window, the published floor
NOBODY = -1
NAME = "possession_v0"
OUT_PATH = os.path.join(ROOT, "evaluation", "results", NAME + ".csv")

def nearest_team(m):
    """(team, distance_m) per frame: team of the tracked player nearest the ball, NOBODY / NaN
    where the ball is not seen or nobody is tracked."""
    L, W = m.pitch
    team = np.full(m.n_frames, NOBODY, dtype=np.int8)
    dist = np.full(m.n_frames, np.nan, dtype=np.float32)
    seen = np.flatnonzero(m.ball_seen)
    for chunk in np.array_split(seen, 20):                     # keeps memory small
        p = m.players[chunk, :, :2].astype(np.float64) * (L, W)
        b = m.ball[chunk, None, :2].astype(np.float64) * (L, W)
        d = np.hypot(*(p - b).transpose(2, 0, 1))
        d = np.where(np.isnan(d), np.inf, d)
        best = d.argmin(axis=1)
        best_d = d[np.arange(len(chunk)), best]
        ok = np.isfinite(best_d)
        team[chunk[ok]] = m.team[best[ok]]
        dist[chunk[ok]] = best_d[ok]
    return team, dist


def vote_in_spell(codes, window):
    """Majority vote of +/- window inside one live spell, gaps filled from the nearest vote.

    Returns label codes plus everything the decision used, all relative to the spell:
    lo, hi (window, hi exclusive), n_home, n_away, source, fill_from (-1 if not filled).
    """
    n = len(codes)
    i = np.arange(n)
    home = np.r_[0, np.cumsum(codes == 0)]
    away = np.r_[0, np.cumsum(codes == 1)]
    lo, hi = np.maximum(0, i - window), np.minimum(n, i + window + 1)
    h, a = home[hi] - home[lo], away[hi] - away[lo]
    result = np.where(h + a == 0, NOBODY, np.where(h >= a, 0, 1)).astype(np.int8)

    source = np.full(n, SRC_VOTE, dtype=np.int8)
    fill_from = np.full(n, -1, dtype=np.int64)
    voted = np.flatnonzero(result != NOBODY)
    if len(voted) == 0:
        source[:] = SRC_NO_BALL
        return result, lo, hi, h, a, source, fill_from

    pos = np.searchsorted(voted, i)
    right = voted[np.clip(pos, 0, len(voted) - 1)]
    left = voted[np.clip(pos - 1, 0, len(voted) - 1)]
    nearest = np.where(np.abs(left - i) <= np.abs(right - i), left, right)
    nearest[voted] = voted                                     # a voted frame keeps its own result
    gap = result == NOBODY
    source[gap] = SRC_FILL
    fill_from[gap] = nearest[gap]
    return result[nearest], lo, hi, h, a, source, fill_from


def build_labels(m, dead):
    """(labels, explain). explain holds per-frame arrays on the video timeline."""
    n = m.n_frames
    labels = np.full(n, "X", dtype="<U1")
    graded = m.half > 0
    labels[graded & dead] = "D"

    vote, vote_dist = nearest_team(m)
    ex = {
        "source": np.where(graded & dead, SRC_DEAD, SRC_UNGRADED).astype(np.int8),
        "vote": np.where(graded & ~dead, vote, NOBODY).astype(np.int8),
        "vote_dist": np.where(graded & ~dead, vote_dist, np.nan).astype(np.float32),
        "win_lo": np.full(n, -1, dtype=np.int32),
        "win_hi": np.full(n, -1, dtype=np.int32),
        "spell_lo": np.full(n, -1, dtype=np.int32),
        "spell_hi": np.full(n, -1, dtype=np.int32),
        "n_home": np.zeros(n, dtype=np.int16),
        "n_away": np.zeros(n, dtype=np.int16),
        "fill_from": np.full(n, -1, dtype=np.int32),
    }
    for s, e in _runs(graded & ~dead):
        res, lo, hi, h, a, src, ff = vote_in_spell(ex["vote"][s:e], VOTE_WINDOW)
        labels[s:e] = np.where(res == 0, "H", np.where(res == 1, "A", "X"))
        ex["win_lo"][s:e], ex["win_hi"][s:e] = lo + s, hi + s
        ex["spell_lo"][s:e], ex["spell_hi"][s:e] = s, e
        ex["n_home"][s:e], ex["n_away"][s:e] = h, a
        ex["source"][s:e] = src
        ex["fill_from"][s:e] = np.where(ff >= 0, ff + s, -1)
    return labels, ex


def main():
    m = load_match(with_homography=False)
    f = features(m)
    llr = dead_evidence(m, f)
    dead, reason = detect_dead(m, llr, f, return_reason=True)
    labels, ex = build_labels(m, dead)
    graded = m.half > 0
    counts = {k: int((labels[graded] == k).sum()) for k in "HADX"}
    print("graded frames: {}".format(counts))
    print("wrote", write_submission(OUT_PATH, labels))
    ex.update(dead_fields(f, llr, reason), vote_window=VOTE_WINDOW)
    print("wrote", write_explain(NAME, "vote", labels, ex))
    return 0


if __name__ == "__main__":
    sys.exit(main())
