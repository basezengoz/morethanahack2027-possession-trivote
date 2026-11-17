"""One loader for the whole match: tracking, homography and Smart Tagging on the video timeline.

Everything returned here is indexed by **video frame** 0 .. n-1, which is also the submission row.
That is the single convention the rest of the pipeline may assume.

    python -m possession.rule_based.loader            # parse (or read the cache), run the checks, print a summary

    from possession.rule_based.loader import load_match
    m = load_match()
    m.players[f, s]     # (x, y, speed) of track slot s at video frame f; NaN if not seen
    m.ball[f]           # (x, y, s) of the ball;                          NaN if not seen
    m.team[s]           # 0 = home (H, Barcelona), 1 = away (A, Athletic)
    m.half[f]           # 0 = not graded (pre-match / halftime / post-match), 1 or 2

Alignment rules (documented in DATA.md, re-checked by check_match() below):
    ATD            video_frame = frame_id - 1
    Smart Tagging  video_frame = round(seconds * 25)
    homography     'frame' column is already the video frame

Units:
    positions are normalised 0-1 along (length, width); use to_metres() for distances.
    player speed is m/s (the XML says kph, but the values are clearly m/s -- median ~1.3, max ~11).
    the ball's third channel is rarely filled (~10% of frames) and does not behave like a speed in
    m/s; derive ball velocity from positions instead.

The parsed tracking is cached as .npz next to the raw data (data/ is gitignored, so the cache can
never reach the public repository). The cache is keyed on the raw file's size and mtime and is
rebuilt automatically if either changes.
"""

import io
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter import load_homography as _homography  # noqa: E402
from starter import load_smart_tagging as _tagging  # noqa: E402

HOME, AWAY = 0, 1
CACHE_VERSION = 1

# Smart Tagging club names -> side. Spelling as in the file ("Atletic", not "Athletic").
TAG_TEAM = {"barcelona": HOME, "atletic club": AWAY}

# Codes come in mirrored pairs on the same interval with opposite Team tags. The first of each
# pair names the team WITH the ball, the second the team defending against it.
ATTACKING_CODES = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION"}
DEFENDING_CODES = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                   "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}


@dataclass
class Match:
    game: str
    source: str
    n_frames: int
    pitch: tuple                      # (length, width) in metres
    players: np.ndarray               # (n, S, 3) float32: x, y, speed
    ball: np.ndarray                  # (n, 3) float32: x, y, s
    team: np.ndarray                  # (S,) int8: HOME / AWAY
    is_goalkeeper: np.ndarray         # (S,) bool
    team_names: tuple                 # (home, away) as written in the ATD metadata
    attack_left_to_right_1st: tuple   # (home, away) bools from the metadata
    half: np.ndarray                  # (n,) int8: 0 not graded, 1, 2
    homography: np.ndarray = None     # (n, 3, 3) float64, NaN where invalid
    reprojection_error: np.ndarray = None
    homography_valid: np.ndarray = None
    tags: list = field(default_factory=list)

    def to_metres(self, xy):
        """Normalised (..., 2) coordinates -> metres. Works on players[..., :2] and ball[..., :2]."""
        return np.asarray(xy, dtype=np.float64) * np.array(self.pitch)

    @property
    def ball_seen(self):
        return ~np.isnan(self.ball[:, 0])

    @property
    def player_seen(self):
        return ~np.isnan(self.players[:, :, 0])


# ---------------------------------------------------------------------------------------------
# tracking
# ---------------------------------------------------------------------------------------------

def read_atd_meta(meta_path):
    """Slot -> team, goalkeeper flags, team names and first-half attack directions."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    home_id = teams[0].get("id")  # FIFA/EPTS convention: first team is home (DATA.md: Barcelona)

    def param(element, name):
        for p in element.iter("ProviderParameter"):
            if p.findtext("Name") == name:
                return p.findtext("Value")
        return None

    slot_team, slot_gk = [], []
    for player in root.iter("Player"):
        slot_team.append(HOME if player.get("teamId") == home_id else AWAY)
        slot_gk.append(param(player, "position_type") == "Goalkeeper")

    return {
        "team": np.array(slot_team, dtype=np.int8),
        "is_goalkeeper": np.array(slot_gk, dtype=bool),
        "team_names": tuple(t.findtext("Name") for t in teams),
        "attack_left_to_right_1st": tuple(
            param(t, "attack_direction_first_half") == "left_to_right" for t in teams),
    }


def parse_atd_raw(raw_path, n_slots, n_frames):
    """Raw '<frame_id>:<x,y,s>;...;<x,y,s>:<x,y,s>' -> (players, ball) indexed by video frame.

    The separators are flattened to commas and the file goes through pandas' C parser in one call:
    seconds instead of minutes, and every line is checked to have exactly the expected width.
    """
    with open(raw_path, "r") as handle:
        text = handle.read()
    text = text.replace(":", ",").replace(";", ",")

    n_cols = 1 + 3 * n_slots + 3
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                        na_values=["NaN"], engine="c")
    del text
    if table.shape[1] != n_cols:
        raise ValueError("{}: {} columns per line, expected {} ({} slots)".format(
            raw_path, table.shape[1], n_cols, n_slots))

    values = table.values
    frame_id = values[:, 0].astype(np.int64)
    video_frame = frame_id - 1                                   # DATA.md: frame_id counts from 1
    if video_frame.min() < 0 or video_frame.max() >= n_frames:
        raise ValueError("frame ids {}..{} do not fit {} video frames".format(
            frame_id.min(), frame_id.max(), n_frames))

    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    players[video_frame] = values[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
    ball[video_frame] = values[:, 1 + 3 * n_slots:]
    return players, ball, frame_id


def _cache_path(raw_path, data_dir, source):
    folder = os.path.join(data_dir or DATA_DIR, "cache")
    return os.path.join(folder, "{}_atd_v{}.npz".format(source, CACHE_VERSION))


def _stamp(path):
    st = os.stat(path)
    return np.array([st.st_size, int(st.st_mtime)], dtype=np.int64)


def load_tracking(game=GAME, data_dir=None, source="tactical", use_cache=True):
    """(players, ball, frame_id, meta). Parses once, then reads the .npz cache."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    meta = read_atd_meta(meta_path)
    n_frames = source_frames(source, game)
    cache = _cache_path(raw_path, data_dir, source)

    if use_cache and os.path.isfile(cache):
        with np.load(cache) as z:
            if np.array_equal(z["stamp"], _stamp(raw_path)):
                return z["players"], z["ball"], z["frame_id"], meta

    players, ball, frame_id = parse_atd_raw(raw_path, len(meta["team"]), n_frames)
    if use_cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, players=players, ball=ball, frame_id=frame_id, stamp=_stamp(raw_path))
    return players, ball, frame_id, meta


# ---------------------------------------------------------------------------------------------
# halves, Smart Tagging
# ---------------------------------------------------------------------------------------------

def half_mask(game=GAME, n_frames=None):
    """(n,) int8: 1 or 2 inside a graded half, 0 elsewhere. From games.half_video_blocks."""
    n_frames = n_frames or GAMES[game]["n_video_frames"]
    half = np.zeros(n_frames, dtype=np.int8)
    for h, first, last in half_video_blocks(game):
        half[first:last + 1] = h
    return half


def load_tags(game=GAME, data_dir=None):
    """Smart Tagging instances on video frames, with the club mapped to a side.

    Each dict: code, start_frame, end_frame (exclusive), start_s, end_s, tags (raw), plus
      side   HOME / AWAY / None   (None for 'N/A' -- every throw-in and kickoff)
      role   'attacking' / 'defending' / 'set_piece' / 'other'
      ball_side  the side that has the ball according to the tag, or None:
                 attacking code -> its side; defending code -> the opposite side.
    """
    out = []
    for event in _tagging.load(game, data_dir):
        code = (event["code"] or "").strip().upper()
        side = TAG_TEAM.get((event["tags"].get("Team") or "").strip().lower())
        if code in ATTACKING_CODES:
            role, ball_side = "attacking", side
        elif code in DEFENDING_CODES:
            role, ball_side = "defending", (None if side is None else 1 - side)
        elif code == "SET PIECES":
            role, ball_side = "set_piece", side
        else:
            role, ball_side = "other", None
        event = dict(event, code=code, side=side, role=role, ball_side=ball_side)
        out.append(event)
    return out


# ---------------------------------------------------------------------------------------------
# everything together
# ---------------------------------------------------------------------------------------------

def load_match(game=GAME, data_dir=None, source="tactical", with_homography=True,
               with_tags=True, use_cache=True):
    """The whole match on the video timeline. Broadcast is NOT aligned to tactical -- see DATA.md."""
    players, ball, _, meta = load_tracking(game, data_dir, source, use_cache)
    n_frames = source_frames(source, game)
    m = Match(
        game=game, source=source, n_frames=n_frames, pitch=tuple(GAMES[game]["pitch"]),
        players=players, ball=ball, team=meta["team"], is_goalkeeper=meta["is_goalkeeper"],
        team_names=meta["team_names"], attack_left_to_right_1st=meta["attack_left_to_right_1st"],
        # Graded halves are defined on the tactical timeline only.
        half=half_mask(game, n_frames) if source == "tactical" else np.zeros(n_frames, np.int8),
    )
    if with_homography:
        m.homography, m.reprojection_error, m.homography_valid = _homography.load(
            game, data_dir, source)
    if with_tags and source == "tactical":
        m.tags = load_tags(game, data_dir)
    return m


def check_match(m, frame_id=None):
    """Falsifiable checks that the alignment and the slot/team mapping are what we assume.

    Returns a list of (name, ok, detail). Nothing here reads any ground truth.
    """
    checks = []
    n = m.n_frames

    def add(name, ok, detail):
        checks.append((name, bool(ok), detail))

    if frame_id is not None:
        ok = len(frame_id) == n and frame_id[0] == 1 and np.all(np.diff(frame_id) == 1)
        add("ATD frame_id is exactly 1..n", ok,
            "{} records, ids {}..{}".format(len(frame_id), frame_id[0], frame_id[-1]))

    add("players array covers every video frame", m.players.shape[0] == n, str(m.players.shape))

    if m.homography is not None:
        add("homography has one row per video frame", len(m.homography) == n,
            "{} rows".format(len(m.homography)))

    home, away = m.team_names
    add("home side is Barcelona (DATA.md)", home.lower().startswith("barcelona"),
        "H = {}, A = {}".format(home, away))
    gk = np.flatnonzero(m.is_goalkeeper)
    add("one goalkeeper slot per team", sorted(m.team[gk].tolist()) == [HOME, AWAY],
        "goalkeeper slots {}".format((gk + 1).tolist()))

    # DATA.md publishes exact counts for the ball (113,488 of all frames, 94,214 of the 158,857
    # graded ones). Matching them to the frame confirms the ATD indexing and the half mask together.
    graded = m.half > 0
    if m.source == "tactical":
        add("ball counts match DATA.md exactly",
            m.ball_seen.sum() == 113488 and graded.sum() == 158857
            and m.ball_seen[graded].sum() == 94214,
            "ball seen {} / graded frames {} / ball seen on graded {}".format(
                int(m.ball_seen.sum()), int(graded.sum()), int(m.ball_seen[graded].sum())))

    # Informational: halftime is NOT empty (warm-ups), so this is not a pass/fail on its own.
    seen = m.player_seen.sum(axis=1)
    add("~20 objects tracked during graded play", seen[graded].mean() > 15,
        "mean tracked objects: graded {:.1f}, not graded {:.1f}".format(
            seen[graded].mean(), seen[~graded].mean()))

    if m.tags:
        report = _tagging.check_time_base(m.tags)
        add("Smart Tagging kickoffs contain the real kickoffs", report["agrees"],
            "; ".join("{} kickoff {} in {}".format(r["half"], r["true_kickoff_frame"], r["tagged"])
                      for r in report["halves"]))

    spd = m.players[..., 2]
    add("player speed looks like m/s", 0.5 < np.nanmedian(spd) < 3 and np.nanmax(spd) < 15,
        "median {:.2f}, max {:.2f}".format(np.nanmedian(spd), np.nanmax(spd)))
    return checks


def main():
    import time
    t0 = time.time()
    players, ball, frame_id, _ = load_tracking()
    m = load_match()
    print("loaded {} in {:.1f} s".format(m.game, time.time() - t0))
    print("  players {}  ball {}  slots H {} / A {}".format(
        m.players.shape, m.ball.shape, int((m.team == HOME).sum()), int((m.team == AWAY).sum())))
    graded = m.half > 0
    print("  ball seen: {:.1%} of all frames, {:.1%} of graded".format(
        m.ball_seen.mean(), m.ball_seen[graded].mean()))
    print("  tags: {}  (with a ball side: {})".format(
        len(m.tags), sum(t["ball_side"] is not None for t in m.tags)))
    print("\nchecks")
    failed = 0
    for name, ok, detail in check_match(m, frame_id):
        failed += not ok
        print("  [{}] {:<50} {}".format("ok" if ok else "FAIL", name, detail))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
