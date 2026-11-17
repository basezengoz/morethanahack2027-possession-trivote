"""Tactical-video overlay as an ASS subtitle file; rules + TCN v6 is the default.

    python scripts/rule_based/overlay_subs.py  # infer rules + TCN v6 and create the subtitle
    python scripts/rule_based/overlay_subs.py --tcn-model v4
    python scripts/rule_based/overlay_subs.py --method rule_based
    python scripts/rule_based/overlay_subs.py --method old_rule_based
    python scripts/rule_based/overlay_subs.py --pred data/runs/example.csv --out data/runs/example.ass

By default the subtitle has the tactical video's stem so VLC loads it automatically. Predictions
and generated files belong in private paths. The public GT layer is an explicit opt-in only.

Layers:

  top left      Smart Tagging: code, team, set-piece type; overlapping instances stack.
                Pushed down a little so the video's burned-in match clock stays visible.
  top right     the prediction in --pred and WHY it was made. The reason is read from the
                explanation file the model wrote (data/debug/<name>.explain.npz), so it is the
                model's own bookkeeping, not a reconstruction. Without that file only the label shows.
  bottom left   optional public GT, only on an explicitly requested separate pipeline check
  bottom right  video clock, frame range and half

Timing is frame-exact: frame f starts at f / 25 s = 4f centiseconds, which ASS represents exactly.
The panel refreshes every STEP frames and on every label change; the clock every STEP frames.
VLC: E steps one frame, [ and ] change speed.

The panel's reason depends on the model that wrote the explanation file (its `kind`):

  vote      (v0) the +/- window, vote counts, a vote strip and this frame's own vote
  viterbi   (v1) the current possession, how it started (restart or change, with the decoder's
            and the snapped frame), the evidence inside it, both teams' nearest distances, this
            frame's control evidence and its factors, and an evidence strip. Only values of the
            DECODED path are shown, never forward-pass running scores, which can disagree with it.

Strip: +/- 50 frames around the frame in 5-frame cells, 10 before and 10 after the frame (the
bar). Each cell: H, A, '·' (nothing either way) or '-' (beyond the live spell).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# games.py sits at the repo root; possession is src/possession (also importable once installed).
for _p in (ROOT, os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from games import DATA_DIR, asset_paths  # noqa: E402
from possession.rule_based.dead_ball import (DEAD_ANCHORED, DEAD_BALL_STILL, DEAD_GOAL, DEAD_OUT_ALIGN,  # noqa: E402
                                  DEAD_PREKICKOFF, DEAD_UNTIL_RESTART, FLAG_TERMS,
                                  DEAD_RESTART_SPOT, LIVE_OUT_ALIGN, LIVE_SHORT_STOP, SPOT_NAMES, _runs)
from possession.rule_based.explain import (SRC_DEAD, SRC_FILL, SRC_NO_BALL, SRC_UNGRADED,  # noqa: E402
                                explain_path)
from possession.rule_based.loader import HOME, load_match  # noqa: E402

FPS = 25
STEP = 5                          # frames between refreshes of the panel and the clock
CELL = 5                          # frames per cell of the vote strip
HALF_CELLS = 10                   # cells each side of the current frame
STRIP_MIN = 0.05                  # |evidence| a strip cell needs to show a side (v1)

# Override-tag colours, &HBBGGRR&.
COLOUR = {"H": "&HFF8C1E&", "A": "&H3C3CE6&", "D": "&HB4B4B4&", "X": "&H707070&",
          "·": "&H909090&", "-": "&H505050&", "│": "&H00FFFF&", "live": "&H50DC50&",
          "white": "&HFFFFFF&"}
LABEL_TEXT = {"H": "H Barcelona", "A": "A Athletic", "D": "D ölü top", "X": "X"}

HEADER = """[Script Info]
Title: BAR-ATH debug overlay
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Tags,Consolas,28,&H00FFFFFF,&H00FFFFFF,&H60000000,&H60000000,0,0,0,0,100,100,0,0,3,6,0,7,24,24,64,1
Style: Panel,Consolas,25,&H00FFFFFF,&H00FFFFFF,&H50000000,&H50000000,0,0,0,0,100,100,0,0,3,6,0,9,24,24,20,1
Style: Clock,Consolas,28,&H0000FFFF,&H00FFFFFF,&H60000000,&H60000000,0,0,0,0,100,100,0,0,3,6,0,3,24,24,20,1
Style: Truth,Consolas,26,&H00FFFFFF,&H00FFFFFF,&H50000000,&H50000000,0,0,0,0,100,100,0,0,3,6,0,1,24,24,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ts(frame):
    """Video frame -> ASS timestamp h:mm:ss.cc (one frame = 4 cs exactly at 25 fps)."""
    cs = int(frame) * 100 // FPS
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return "{}:{:02d}:{:02d}.{:02d}".format(h, m, s, cs)


def event(start, end, style, text, layer=0):
    return "Dialogue: {},{},{},{},,0,0,0,,{}".format(layer, ts(start), ts(end), style, text)


def segments_to_events(starts, texts, n_frames, style):
    """Events for segments [starts[i], starts[i+1]); identical neighbours merged, '' skipped."""
    out, i = [], 0
    bounds = list(starts) + [n_frames]
    while i < len(texts):
        j = i
        while j + 1 < len(texts) and texts[j + 1] == texts[i]:
            j += 1
        if texts[i]:
            out.append(event(bounds[i], bounds[j + 1], style, texts[i]))
        i = j + 1
    return out


def c(colour_key):
    return "{\\c" + COLOUR[colour_key] + "}"


# ---------------------------------------------------------------------------------------------
# Smart Tagging and clock
# ---------------------------------------------------------------------------------------------

def tag_layer(m):
    lines = []
    for t in sorted(m.tags, key=lambda t: t["start_frame"]):
        team = (t["tags"].get("Team") or "").strip() or "?"
        parts = [t["code"], team]
        kind = (t["tags"].get("Type") or "").strip()
        if kind:
            parts.append(kind)
        if t["ball_side"] is not None and t["role"] != "set_piece":
            parts.append("top: " + ("H" if t["ball_side"] == HOME else "A"))
        text = " · ".join(parts)
        if t["ball_side"] is not None:
            text = c("H" if t["ball_side"] == HOME else "A") + text
        end = max(t["end_frame"], t["start_frame"] + 1)
        lines.append(event(t["start_frame"], end, "Tags", text))
    return lines


def clock_layer(m):
    starts = list(range(0, m.n_frames, STEP))
    texts = []
    for b in starts:
        sec = b // FPS
        half = {0: "değerlendirilmez", 1: "1. devre", 2: "2. devre"}[int(m.half[b])]
        texts.append("{:d}:{:02d}:{:02d}  f {}–{}  {}".format(
            sec // 3600, sec // 60 % 60, sec % 60, b, min(b + STEP, m.n_frames) - 1, half))
    return segments_to_events(starts, texts, m.n_frames, "Clock")


# ---------------------------------------------------------------------------------------------
# prediction + explanation panel
# ---------------------------------------------------------------------------------------------

class Explainer:
    """Common part: dead-ball and ungraded frames, the strip, the live-frame tail."""

    def __init__(self, ex, ball_seen, labels):
        self.ex = ex
        self.ball_seen = ball_seen
        self.labels = labels
        # For every frame, the first frame at or after it that is not dead (end of the stoppage).
        n = len(labels)
        idx = np.where(ex["source"] != SRC_DEAD, np.arange(n), n)
        self.next_live = np.minimum.accumulate(idx[::-1])[::-1]

    def strip(self, f, lo, hi, cell):
        """Coloured strip around f: 10 cells before, the bar, 10 after; cells clipped to [lo, hi)."""
        def one(a, b):
            a, b = max(a, lo), min(b, hi)
            return "-" if a >= b else cell(a, b)
        w = CELL * HALF_CELLS
        chars = ([one(f - w + CELL * j, f - w + CELL * (j + 1)) for j in range(HALF_CELLS)]
                 + ["│"]
                 + [one(f + 1 + CELL * j, f + 1 + CELL * (j + 1)) for j in range(HALF_CELLS)])
        out, prev = "", None
        for ch in chars:
            if ch != prev:
                out += c(ch)
                prev = ch
            out += ch
        return "  " + c("white") + "[" + out + c("white") + "]"

    def dead_lines(self, f):
        ex = self.ex
        active = [k for k in FLAG_TERMS if ex["flag_" + k][f]]
        def num(key):
            v = ex[key][f] if key in ex else np.nan
            return "—" if np.isnan(v) else "{:.1f}".format(float(v))
        return ["  llr {:+.1f} (+ = ölü) · tempo {} m/s".format(float(ex["llr"][f]), num("speed")),
                "    (herkes {} · topa yakın {})".format(num("speed_all"), num("speed_local")),
                "  aktif: " + (" · ".join(active) if active else "—")]

    def common(self, f):
        """Lines for ungraded and dead frames, or None for a live frame."""
        src = int(self.ex["source"][f])
        if src == SRC_UNGRADED:
            return ["değerlendirilmez (devre dışı)"]
        if src == SRC_DEAD:
            reason = int(self.ex["dead_reason"][f])
            if reason == DEAD_PREKICKOFF:
                rule = ["kural: devre başı, kickoff bekleniyor"]
            elif reason == DEAD_BALL_STILL:
                restart = int(self.next_live[f])
                rule = ["kural: top yerinde duruyor, oynanana kadar ölü",
                        "  top f {} ile oynandı ({:.1f} s sonra)".format(
                            restart, (restart - f) / FPS)]
            elif reason == DEAD_UNTIL_RESTART:
                restart = int(self.next_live[f])
                rule = ["kural: top dışarı / duran top sonrası, top yerleştirilip",
                        "  oynanana kadar ölü; oyun f {} ile başladı ({:.1f} s sonra)".format(
                            restart, (restart - f) / FPS)]
            elif reason == DEAD_OUT_ALIGN:
                rule = ["kural: top çizgiyi geçti, duraklama bu kareden başlıyor"]
            elif reason == DEAD_RESTART_SPOT:
                kind = int(self.ex["spot_kind"][f]) if "spot_kind" in self.ex else 0
                rule = ["kural: top yeniden başlama noktasında bekliyor ({})".format(
                            SPOT_NAMES.get(kind, "?")),
                        "  rakipler kuralın istediği mesafede"]
            elif reason == DEAD_GOAL:
                rule = ["kural: gol; santraya kadar ölü top"]
            elif reason == DEAD_ANCHORED:
                rule = ["kural: güçlü tetikleyici (top dışarı / duran top etiketi)",
                        "  oyun kanıtı bu duraklamayı silemez"]
            else:
                rule = ["karar: Viterbi (dead-ball kanıtları)"]
            return ["neden: ölü top modeli"] + ["  " + r for r in rule] + self.dead_lines(f)
        return None

    def live_tail(self, f):
        out = []
        if int(self.ex["dead_reason"][f]) == LIVE_SHORT_STOP:
            out.append("not: <3 s duraklama canlıya çevrildi")
        if int(self.ex["dead_reason"][f]) == LIVE_OUT_ALIGN:
            out.append("not: duraklama topun çizgiyi geçtiği ana kaydırıldı")
        out.append("dead-ball: {}CANLI{}".format(c("live"), c("white")))
        return out + self.dead_lines(f)

    def cut_keys(self):
        """Explanation arrays whose changes must start a new panel segment."""
        return ("source",)

    def lines(self, f):
        return self.common(f) or self.live_lines(f) + self.live_tail(f)


class VoteExplainer(Explainer):
    """v0: nearest-player votes, +/- window majority, fill from the nearest voted frame."""

    def __init__(self, ex, ball_seen, labels):
        super().__init__(ex, ball_seen, labels)
        self.w = int(ex["vote_window"])
        vote = ex["vote"]
        self.cum_h = np.r_[0, np.cumsum(vote == 0)]
        self.cum_a = np.r_[0, np.cumsum(vote == 1)]

    def cell(self, a, b):
        h = self.cum_h[b] - self.cum_h[a]
        x = self.cum_a[b] - self.cum_a[a]
        if h + x == 0:
            return "·"
        return "H" if h >= x else "A"

    def vote_block(self, q):
        ex = self.ex
        lo, hi = int(ex["win_lo"][q]), int(ex["win_hi"][q])
        h, a = int(ex["n_home"][q]), int(ex["n_away"][q])
        notes = []
        if lo > q - self.w:
            notes.append("aralık başı")
        if hi < q + self.w + 1:
            notes.append("aralık sonu")
        head = "oylama  f {}–{} (±{}{})".format(
            lo, hi - 1, self.w, ", " + ", ".join(notes) if notes else "")
        lines = [head,
                 "  {}H {} kare{} · {}A {} kare{} · oysuz {}".format(
                     c("H"), h, c("white"), c("A"), a, c("white"), hi - lo - h - a),
                 self.strip(q, lo, hi, self.cell)]
        if h == a and h > 0:
            lines.append("  eşitlik → H")
        vote, dist = ex["vote"][lo:hi], ex["vote_dist"][lo:hi]
        med = []
        for code, key in ((0, "H"), (1, "A")):
            d = dist[vote == code]
            med.append("{}{} {}{}".format(
                c(key), key, "—" if len(d) == 0 else "{:.1f} m".format(np.median(d)), c("white")))
        lines.append("  medyan mesafe  " + " · ".join(med))
        return lines

    def own_vote(self, f):
        v = int(self.ex["vote"][f])
        if v >= 0:
            key = "H" if v == 0 else "A"
            return "bu kare: {}{}{}, en yakın oyuncu {:.1f} m".format(
                c(key), key, c("white"), float(self.ex["vote_dist"][f]))
        if not self.ball_seen[f]:
            return "bu kare: oy yok (top görünmüyor)"
        return "bu kare: oy yok (takip edilen oyuncu yok)"

    def live_lines(self, f):
        ex = self.ex
        if int(ex["source"][f]) == SRC_NO_BALL:
            return ["neden: bu canlı aralıkta hiç oy yok",
                    "  aralık f {}–{}, top hiç görülmedi".format(
                        int(ex["spell_lo"][f]), int(ex["spell_hi"][f]) - 1)]
        out = []
        q = f
        if int(ex["source"][f]) == SRC_FILL:
            q = int(ex["fill_from"][f])
            out.append("neden: doldurma, f {}'in sonucu".format(q))
            out.append("  {} kare {} · bu karenin penceresinde oy yok".format(
                abs(q - f), "ileride" if q > f else "geride"))
        else:
            out.append("neden: oylama, en yakın oyuncunun takımı")
        return out + self.vote_block(q) + [self.own_vote(f)]


class ViterbiExplainer(Explainer):
    """v1: control evidence, restart rule, switching cost, snapped changes (possession.rule_based.model)."""

    def __init__(self, ex, ball_seen, labels):
        super().__init__(ex, ball_seen, labels)
        self.cum = np.r_[0.0, np.cumsum(ex["control"].astype(np.float64))]

    def cut_keys(self):
        return ("source", "poss_lo")

    def cell(self, a, b):
        v = self.cum[b] - self.cum[a]
        return "H" if v > STRIP_MIN else ("A" if v < -STRIP_MIN else "·")

    def start_lines(self, f, lab):
        ex = self.ex
        kind = int(ex["poss_kind"][f])
        team = c(lab) + lab + c("white")
        if kind == 4:
            line = "  başlangıç: yeniden başlatma, top f {} anında oyuna girdi, {} en yakın ({:.1f} m)".format(
                int(ex["poss_restart_frame"][f]), team, float(ex["poss_restart_dist"][f]))
        elif kind == 0:
            line = "  başlangıç: yeniden başlatma, {} en yakın ({:.1f} m)".format(
                team, float(ex["poss_restart_dist"][f]))
        elif kind == 1:
            line = "  başlangıç: yeniden başlatma, {} (Smart Tagging duran top, etiket f {})".format(
                team, int(ex["poss_restart_frame"][f]))
        elif kind == 2:
            line = "  başlangıç: yeniden başlatan belirsiz, kanıta göre {}".format(team)
        else:
            dec, lo = int(ex["poss_decoder_frame"][f]), int(ex["poss_lo"][f])
            moved = "" if dec == lo else " (Viterbi f {}, {} kare ileri)".format(dec, lo - dec)
            line = "  başlangıç: değişim, {} ilk kontrol f {}{}".format(team, lo, moved)
        out = [line]
        if kind == 3:
            out.append("  ilk 2 s kontrol kanıtı: {} +{:.1f} (bedel {:.0f})".format(
                team, float(ex["poss_start_ev"][f]), float(ex["switch_cost"])))
        return out

    def frame_line(self, f):
        ex = self.ex
        e = float(ex["control"][f])
        if "keeper" in ex and ex["keeper"][f]:
            key = "H" if e > 0 else "A"
            return "  bu kare: {}{} {:+.2f}{}  (kaleci topu tutuyor, kendi ceza sahasında)".format(
                c(key), key, e, c("white"))
        if abs(e) > 1e-6:
            key = "H" if e > 0 else "A"
            return "  bu kare: {}{} {:+.2f}{}  (yakın {:.2f} · ayrık {:.2f} · hız {:.2f})".format(
                c(key), key, e, c("white"), float(ex["near"][f]), float(ex["sep"][f]),
                float(ex["gate"][f]))
        if not ex["ball_seen_filled"][f]:
            why = "top görünmüyor"
        elif ex["gate"][f] == 0:
            why = "top havada / hızlı ({:.0f} m/s)".format(float(ex["ball_speed"][f]))
        elif ex["near"][f] == 0:
            why = "en yakın oyuncu 4 m'den uzak"
        elif ex["sep"][f] == 0:
            why = "ikili mücadele, iki takım eşit yakın"
        else:
            why = "kanıt yok"
        return "  bu kare: kanıt yok — " + why

    def live_lines(self, f):
        ex = self.ex
        lab = self.labels[f]
        lo, hi = int(ex["poss_lo"][f]), int(ex["poss_hi"][f])
        out = ["neden: Viterbi · kontrol kanıtı · değişim bedeli {:.0f}".format(
                   float(ex["switch_cost"])),
               "possession f {}–{} ({:.1f} s)".format(lo, hi - 1, (hi - lo) / FPS)]
        out += self.start_lines(f, lab)
        out.append("  içindeki kanıt  {}H +{:.1f}{} · {}A +{:.1f}{}".format(
            c("H"), float(ex["poss_ev_home"][f]), c("white"),
            c("A"), float(ex["poss_ev_away"][f]), c("white")))
        if "pass_ev" in ex:
            out.append("  içindeki paslar  {}H {}{} · {}A {}{}  (pas başı {:.1f})".format(
                c("H"), int(ex["poss_pass_home"][f]), c("white"),
                c("A"), int(ex["poss_pass_away"][f]), c("white"), float(ex["pass_evidence"])))
            recent = np.flatnonzero(ex["pass_ev"][max(0, f - FPS):f + 1] != 0)
            if len(recent):
                g = max(0, f - FPS) + int(recent[-1])
                key = "H" if ex["pass_ev"][g] > 0 else "A"
                out.append("  son pas: {}{}{} tamamladı (f {})".format(c(key), key, c("white"), g))

        def dist(v):
            return "—" if np.isnan(v) else "{:.1f} m".format(float(v))
        speed = ex["ball_speed"][f]
        out.append("kontrol  en yakın {}H{} {} · {}A{} {} · top {}".format(
            c("H"), c("white"), dist(ex["d_home"][f]), c("A"), c("white"), dist(ex["d_away"][f]),
            "—" if np.isnan(speed) else "{:.1f} m/s".format(float(speed))))
        out.append(self.strip(f, int(ex["spell_lo"][f]), int(ex["spell_hi"][f]), self.cell))
        out.append(self.frame_line(f))
        side = int(ex["tag_side"][f])
        if side != 0:
            key = "H" if side > 0 else "A"
            out.append("smart tagging: top {}{}{} ({:+.3f}/kare)".format(
                c(key), key, c("white"), float(ex["tag_ev"][f])))
        return out


def make_explainer(ex, ball_seen, labels):
    kind = str(ex["kind"]) if "kind" in ex else "vote"
    cls = {"vote": VoteExplainer, "viterbi": ViterbiExplainer}.get(kind)
    if cls is None:
        print("note: unknown explanation kind {!r}; panel shows the label only".format(kind))
        return None
    return cls(ex, ball_seen, labels)


def panel_layer(m, pred_path, title=None):
    frame = pd.read_csv(pred_path)
    if list(frame.columns) != ['frame', 'label'] or not np.array_equal(frame['frame'].values, np.arange(m.n_frames)):
        raise ValueError('prediction CSV must have ordered frame,label rows on the tactical timeline')
    labels = frame["label"].astype(str).str.strip().str.upper().values
    if not np.isin(labels, list('HADX')).all():
        raise ValueError('invalid prediction label')
    if len(labels) != m.n_frames:
        raise SystemExit("{} has {} rows, expected {}".format(pred_path, len(labels), m.n_frames))
    name = os.path.splitext(os.path.basename(pred_path))[0]

    ex_path = explain_path(name)
    explainer = None
    if os.path.isfile(ex_path):
        with np.load(ex_path) as z:
            ex = {k: z[k] for k in z.files}
        if not np.array_equal(ex["labels"], labels):
            print("warning: {} does not match {}; explanation skipped".format(ex_path, pred_path))
        else:
            explainer = make_explainer(ex, m.ball_seen, labels)
    else:
        print("note: no explanation file {}; panel shows the label only".format(ex_path))

    # Segment starts: every STEP frames, plus every label change, plus (if explained) every
    # change of the label's source, so the header and the reason are frame-exact.
    cuts = set(range(0, m.n_frames, STEP))
    cuts.update(s for lab in "HADX" for s, _ in _runs(labels == lab))
    if explainer is not None:
        for key in explainer.cut_keys():
            cuts.update(np.flatnonzero(np.diff(explainer.ex[key]) != 0) + 1)
    starts = sorted(int(s) for s in cuts)

    texts = []
    for f in starts:
        lab = labels[f]
        head = "{{\\fs20}}{}{}\\N{{\\fs44}}{}{}{{\\fs25}}{}".format(
            c("white"), title or name, c(lab), LABEL_TEXT[lab], c("white"))
        if explainer is None:
            texts.append(head)
            continue
        body = explainer.lines(f)
        texts.append(head + "\\N" + "─" * 30 + "\\N" + "\\N".join(body))
    return segments_to_events(starts, texts, m.n_frames, "Panel")


# ---------------------------------------------------------------------------------------------

def video_stem(data_dir=None):
    return os.path.splitext(str(asset_paths(data_dir=data_dir)['video']))[0]


PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def truth_layer(m, pred_labels):
    """Public ground truth, bottom left, only where it has labels; shows the prediction where it
    differs. A pipeline check, not a selection tool (RULES.md, EVALUATION.md)."""
    if not os.path.isfile(PUBLIC_GT):
        return []
    gt = pd.read_csv(PUBLIC_GT)["label"].astype(str).str.strip().str.upper().values
    if len(gt) != m.n_frames:
        return []
    same = gt == pred_labels
    cut = np.flatnonzero((gt[1:] != gt[:-1]) | (same[1:] != same[:-1])) + 1
    bounds = np.r_[0, cut, m.n_frames]
    lines = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        g = gt[a]
        if g == "X":
            continue
        text = "{\\fs20}" + c("white") + "halka açık GT\\N{\\fs36}" + c(g) + LABEL_TEXT[g]
        if not same[a]:
            p = pred_labels[a]
            text += "{\\fs24}" + c("white") + "  ≠ tahmin " + c(p) + LABEL_TEXT.get(p, p)
        lines.append(event(a, b, "Truth", text))
    # Outside the labelled window say where it is, so an empty corner is never ambiguous.
    labelled = np.flatnonzero(gt != "X")
    if len(labelled):
        lo, hi = int(labelled[0]), int(labelled[-1]) + 1
        note = "{\\fs20}" + c("X") + "halka açık GT yalnızca {} – {}".format(ts(lo)[:7], ts(hi)[:7])
        if lo > 0:
            lines.append(event(0, lo, "Truth", note))
        if hi < m.n_frames:
            lines.append(event(hi, m.n_frames, "Truth", note))
    return lines


def write_overlay(m, tags, clock, pred_path, out, include_public_gt=False, title=None):
    panel = panel_layer(m, pred_path, title)
    pred = pd.read_csv(pred_path)["label"].astype(str).str.strip().str.upper().values
    truth = truth_layer(m, pred) if include_public_gt else []
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="\n") as handle:
        handle.write(HEADER)
        for lines in (tags, panel, truth, clock):
            handle.write("\n".join(lines) + "\n")
    print("wrote {}  ({:.1f} MB, panel {} events, public GT {} events)".format(
        out, os.path.getsize(out) / 1e6, len(panel), len(truth)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pred", default=None, help="use an existing prediction CSV instead of inference")
    ap.add_argument("--out", default=None, help="output .ass (default: alongside tactical video)")
    ap.add_argument('--method', choices=('rule_based_tcn', 'rule_based', 'old_rule_based'), default='rule_based_tcn',
                    help='rule_based_tcn: kural tabanlı yöntem + TCN; rule_based: kural tabanlı yöntem; old_rule_based: previous rules')
    ap.add_argument('--tcn-model', choices=('v6', 'v5', 'v4'), default='v6')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--checkpoint-sha256', default=None)
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--title', default=None, help='display name for the prediction')
    ap.add_argument('--include-public-gt', action='store_true', help='explicit separate public pipeline check only')
    args = ap.parse_args(argv)
    if args.pred and (args.method != 'rule_based_tcn' or args.tcn_model != 'v6' or args.checkpoint or args.checkpoint_sha256):
        ap.error('--pred cannot be combined with inference method or checkpoint options')
    if not args.pred:
        import subprocess
        pred = os.path.join(ROOT, 'data', 'runs', 'overlay', args.method + '_' + args.tcn_model + '.csv')
        cmd = [sys.executable, '-m', 'possession.predict', '--method', args.method,
               '--tcn-model', args.tcn_model, '--output', pred]
        if args.data_dir is not None:
            cmd += ['--data-dir', os.path.abspath(args.data_dir)]
        if args.checkpoint:
            cmd += ['--checkpoint', os.path.abspath(args.checkpoint)]
        if args.checkpoint_sha256:
            cmd += ['--checkpoint-sha256', args.checkpoint_sha256]
        subprocess.run(cmd, cwd=ROOT, check=True)
        args.pred = pred
        if args.title is None:
            args.title = {'rule_based_tcn': 'Kural tabanlı + TCN ' + args.tcn_model,
                          'rule_based': 'Kural tabanlı', 'old_rule_based': 'Eski kural tabanlı'}[args.method]
    m = load_match(data_dir=args.data_dir, with_homography=False)
    tags, clock = tag_layer(m), clock_layer(m)
    stem = video_stem(args.data_dir)
    write_overlay(m, tags, clock, args.pred, args.out or stem + '.ass', args.include_public_gt, args.title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
