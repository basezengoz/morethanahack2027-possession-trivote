"""Model report: what changed, why, which positions changed label, and where the model is unsure.

    python scripts/rule_based/report.py        # writes data/derived/rule_based/model_report.html (usually run by
                                       # scripts/rule_based/pipeline.py after every model update)

Every model update adds one entry to data/derived/rule_based/changelog.json (what, why, results) and runs
`python scripts/rule_based/pipeline.py`. The report then shows, for that entry, every position whose label
changed against the previous snapshot, and saves the current labels as the entry's snapshot
(data/derived/rule_based/snapshots/<id>.npz). Re-running while still working on the same entry keeps
comparing against the entry before it.

The top of the report lists the moments where the current model was closest to deciding the other
way, with the reason, as a viewing list for the debug overlay.

Video reviews (data/derived/rule_based/reviews.json) are shown next to the positions they concern. They are
human observations of this match: qualitative only, never read by a model and never used to
choose a constant (RULES.md).

Everything lives under data/ (gitignored): the report names moments of the match and carries the
team's own observations, neither of which belongs in the public repository.
"""

import csv
import datetime
import html
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# games.py sits at the repo root; possession is src/possession (also importable once installed).
for _p in (ROOT, os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from games import DATA_DIR, GAME  # noqa: E402
from possession.rule_based import dead_ball as db  # noqa: E402
from possession.rule_based import model as mdl  # noqa: E402
from possession.rule_based.dead_ball import _runs  # noqa: E402
from possession.rule_based.explain import explain_path  # noqa: E402
from possession.rule_based.loader import half_mask  # noqa: E402

FPS = 25
REPORT_DIR = os.path.join(DATA_DIR, "derived", "rule_based")
SNAP_DIR = os.path.join(REPORT_DIR, "snapshots")
REVIEWS = os.path.join(REPORT_DIR, "reviews.json")
OUT = os.path.join(REPORT_DIR, "model_report.html")
SCOREBOARD = os.path.join(ROOT, "evaluation", "scoreboard.csv")
MODEL = "possession_v1"

# ---------------------------------------------------------------------------------------------
# Change log: data/derived/rule_based/changelog.json, newest last. `snapshot`: id of the labels saved after
# the change (null: none kept). `results`: measured at the time; the newest entry also gets live
# numbers. Kept out of the code because entries name moments of the match.
# ---------------------------------------------------------------------------------------------
CHANGELOG_PATH = os.path.join(REPORT_DIR, "changelog.json")


def load_changelog():
    """Entries, oldest first: id, date, snapshot (or null), title, what [], why, results []."""
    with open(CHANGELOG_PATH, encoding="utf-8") as handle:
        return json.load(handle)["entries"]


CHANGELOG = load_changelog() if os.path.isfile(CHANGELOG_PATH) else []



# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def tc(frame):
    """Video time as the team writes it: minutes:seconds from the start of the video."""
    sec = int(frame) // FPS
    return "{}:{:02d}".format(sec // 60, sec % 60)


def parse_tc(text):
    mm, ss = text.split(":")
    return (int(mm) * 60 + int(ss)) * FPS


def esc(x):
    return html.escape(str(x))


def load_reviews():
    if not os.path.isfile(REVIEWS):
        return []
    with open(REVIEWS, encoding="utf-8") as handle:
        items = json.load(handle).get("reviews", [])
    for r in items:
        r["a"], r["b"] = parse_tc(r["start"]), parse_tc(r["end"])
        r.setdefault("expected", "live" if r.get("verdict") == "correct" else "dead")
    return items


def reviews_for(reviews, a, b, snapshot=None):
    return [r for r in reviews if r["a"] < b + 2 * FPS and a < r["b"] + 2 * FPS
            and (snapshot is None or r["snapshot"] == snapshot)]


def load_snapshot(sid):
    path = os.path.join(SNAP_DIR, sid + ".npz")
    if not os.path.isfile(path):
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def save_snapshot(sid, ex):
    os.makedirs(SNAP_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(SNAP_DIR, sid + ".npz"),
                        labels=ex["labels"], dead_reason=ex["dead_reason"])


def term_values(ex):
    """Per-frame dead-ball terms rebuilt from the explanation's flags (same weights as dead_ball)."""
    speed = np.nan_to_num(ex["speed"].astype(float), nan=db.WALK_SPEED)
    t = {"tempo": np.clip(db.SPEED_WEIGHT * (db.WALK_SPEED - speed), -db.SPEED_CLIP, db.SPEED_CLIP)}
    weights = {"out": db.OUT_WEIGHT, "missing": db.MISSING_WEIGHT, "fast": db.FAST_WEIGHT,
               "setpiece": db.SETPIECE_WEIGHT, "kickoff": db.KICKOFF_WEIGHT,
               "dribble": db.DRIBBLE_WEIGHT, "chain": db.CHAIN_WEIGHT}
    for k, w in weights.items():
        if "flag_" + k in ex:
            t[k] = w * ex["flag_" + k]
    if "flag_keeper" in ex:   # cancels the dead side of tempo and missing (dead_ball docstring)
        t["keeper"] = -1.0 * ex["flag_keeper"] * (np.maximum(t["tempo"], 0) + t.get("missing", 0))
    return t


TERM_TR = {"tempo": "tempo", "out": "top dışarıda", "missing": "top görünmüyor", "fast": "hızlı top",
           "setpiece": "duran top etiketi", "kickoff": "başlama dizilişi", "dribble": "top sürme",
           "chain": "pas zinciri", "keeper": "kaleci topu tutuyor"}


def top_terms(terms, a, b, k=3):
    sums = sorted(((float(v[a:b].sum()), name) for name, v in terms.items()), key=lambda x: -abs(x[0]))
    return ", ".join("{} {:+.0f}".format(TERM_TR[n], s) for s, n in sums[:k] if abs(s) >= 1)


def play_like(ex):
    v = np.nan_to_num(ex["ball_speed"].astype(float))
    dmin = np.nan_to_num(np.fmin(ex["d_home"], ex["d_away"]).astype(float), nan=99)
    return ex["ball_seen_filled"] & ~ex["flag_out"] & (v > 2) & (v < 8) & (dmin < 1.5)


# ---------------------------------------------------------------------------------------------
# where the model is unsure
# ---------------------------------------------------------------------------------------------

def uncertain(ex, half):
    lab = ex["labels"]
    dead = lab == "D"
    llr = ex["llr"].astype(float)
    terms = term_values(ex)
    entry = 2 * db.SWITCH_COST            # entering and leaving a stoppage
    items = []

    # A. fragile stoppages: little evidence beyond what entering and leaving costs, or held by a rule
    for s, e in _runs(dead & (half > 0)):
        reason = np.bincount(ex["dead_reason"][s:e], minlength=8).argmax()
        if reason == db.DEAD_PREKICKOFF:
            continue
        total = float(llr[s:e].sum())
        if reason == db.DEAD_ANCHORED:
            items.append(dict(a=s, b=e, kind="Korunan duraklama", score=0.9,
                              text="Oyun kanıtı bu duraklamayı siliyordu; güçlü tetikleyici (top "
                                   "dışarı / duran top etiketi) korudu. Kanıtlar: " + top_terms(terms, s, e)))
        elif total - entry < 30:
            items.append(dict(a=s, b=e, kind="Kırılgan duraklama",
                              score=float(np.clip(1 - (total - entry) / 30, 0, 1)) * 0.85,
                              text="Duraklama kanıtı toplamı {:+.0f}, giriş+çıkış bedeli {:.0f}. "
                                   "Kanıtlar: {}".format(total, entry, top_terms(terms, s, e))))

    # B. almost-stoppages inside live play (best sub-run of dead evidence)
    for s, e in _runs(~dead & (half > 0)):
        x = llr[s:e]
        best, cur, cur_start, span = 0.0, 0.0, 0, (0, 0)
        for i, v in enumerate(x):
            if cur <= 0:
                cur, cur_start = v, i
            else:
                cur += v
            if cur > best:
                best, span = cur, (cur_start, i + 1)
        a, b = s + span[0], s + span[1]
        if best >= 0.6 * entry and b - a >= 2 * FPS:
            items.append(dict(a=a, b=b, kind="Neredeyse duraklama", score=min(1.0, best / entry) * 0.9,
                              text="Canlı ama {:.1f} sn boyunca ölü top kanıtı birikti ({:+.0f}, eşik "
                                   "{:.0f}). Kanıtlar: {}".format((b - a) / FPS, best, entry,
                                                                   top_terms(terms, a, b))))

    # C/D/E. possession: close changes, absorbed opponent control, unclear restarts
    lo = ex["poss_lo"]
    starts = [i for i in np.flatnonzero(np.diff(np.r_[-2, lo]) != 0) if lo[i] >= 0]
    cost = float(ex["switch_cost"])
    control = ex["control"].astype(float)
    for i in starts:
        a, b = int(lo[i]), int(ex["poss_hi"][i])
        team = lab[i]
        other = "A" if team == "H" else "H"
        kind = int(ex["poss_kind"][i])
        if kind == mdl.START_CHANGE:
            ev = float(ex["poss_start_ev"][i])
            if abs(ev - cost) < 10:
                items.append(dict(a=a, b=min(b, a + 3 * FPS), kind="Sınırda değişim",
                                  score=(1 - abs(ev - cost) / 10) * 0.8,
                                  text="Possession {} takımına geçti; ilk 2 sn kontrol kanıtı +{:.0f}, "
                                       "değişim bedeli {:.0f}.".format(team, ev, cost)))
        elif kind == mdl.START_RESTART_NONE:
            items.append(dict(a=a, b=min(b, a + 3 * FPS), kind="Belirsiz yeniden başlatma", score=0.6,
                              text="Oyunu kimin başlattığı tracking'den de etiketten de çıkmadı; "
                                   "{} kanıta göre seçildi.".format(team)))
        elif kind == mdl.START_RESTART_TRACKING and ex["poss_restart_dist"][i] > 2.0:
            items.append(dict(a=a, b=min(b, a + 3 * FPS), kind="Belirsiz yeniden başlatma", score=0.5,
                              text="Yeniden başlatan {}: en yakın oyuncu topa {:.1f} m.".format(
                                  team, float(ex["poss_restart_dist"][i]))))
        opp = float(ex["poss_ev_away"][i] if team == "H" else ex["poss_ev_home"][i])
        if opp >= 20:
            sign = -1.0 if team == "H" else 1.0
            c = np.maximum(sign * control[a:b], 0)
            w = 2 * FPS
            win = np.convolve(c, np.ones(w), mode="valid") if len(c) >= w else c
            k = int(np.argmax(win))
            items.append(dict(a=a + k, b=a + k + w, kind="Olası kaçırılmış değişim",
                              score=min(1.0, opp / 50) * 0.85,
                              text="{} possession'ı ({:.0f} sn) içinde {} takımının kontrol kanıtı +{:.0f}, "
                                   "parça parça geldiği için değişim olmadı. En yoğun yer burası.".format(
                                       team, (b - a) / FPS, other, opp)))

    items.sort(key=lambda x: -x["score"])
    picked, per_kind = [], {}
    limits = {"Korunan duraklama": 8, "Kırılgan duraklama": 6, "Neredeyse duraklama": 6,
              "Sınırda değişim": 5, "Olası kaçırılmış değişim": 6, "Belirsiz yeniden başlatma": 4}
    for it in items:
        if per_kind.get(it["kind"], 0) < limits.get(it["kind"], 5):
            per_kind[it["kind"]] = per_kind.get(it["kind"], 0) + 1
            picked.append(it)
    return picked


# ---------------------------------------------------------------------------------------------
# label diffs between two versions
# ---------------------------------------------------------------------------------------------

def diff_labels(old, new, half):
    od, nd = (old == "D") & (half > 0), (new == "D") & (half > 0)
    rows = []
    for s, e in _runs(od):
        k = np.flatnonzero(nd[s:e])
        if len(k) == 0:
            rows.append(dict(a=s, b=e, kind="Duraklama kaldırıldı",
                             detail="{}–{} ({:.1f} sn) artık canlı".format(tc(s), tc(e), (e - s) / FPS)))
            continue
        ds, de = k[0] / FPS, (e - s - 1 - k[-1]) / FPS
        if ds >= 1 or de >= 1:
            parts = []
            if ds >= 1:
                parts.append("{:.1f} sn geç başlıyor".format(ds))
            if de >= 1:
                parts.append("{:.1f} sn erken bitiyor".format(de))
            rows.append(dict(a=s, b=e, kind="Duraklama kısaldı", detail=", ".join(parts)))
    for s, e in _runs(nd):
        k = np.flatnonzero(od[s:e])
        if len(k) == 0:
            if e - s >= FPS:
                rows.append(dict(a=s, b=e, kind="Duraklama eklendi",
                                 detail="{}–{} ({:.1f} sn) artık ölü".format(tc(s), tc(e), (e - s) / FPS)))
            continue
        ds, de = k[0] / FPS, (e - s - 1 - k[-1]) / FPS
        if ds >= 1 or de >= 1:
            parts = []
            if ds >= 1:
                parts.append("{:.1f} sn erken başlıyor".format(ds))
            if de >= 1:
                parts.append("{:.1f} sn geç bitiyor".format(de))
            rows.append(dict(a=s, b=e, kind="Duraklama uzadı", detail=", ".join(parts)))
    both_live = (half > 0) & ~od & ~nd
    for s, e in _runs(both_live & (old != new)):
        if e - s >= FPS:
            rows.append(dict(a=s, b=e, kind="Possession değişti",
                             detail="{} → {} ({:.1f} sn)".format(old[s], new[s], (e - s) / FPS)))
    rows.sort(key=lambda r: r["a"])
    return rows


# ---------------------------------------------------------------------------------------------
# current state
# ---------------------------------------------------------------------------------------------

def current_metrics(ex, half):
    lab = ex["labels"]
    graded = half > 0
    dead = (lab == "D") & graded
    stops = [(s, e) for s, e in _runs(dead)]
    pl = play_like(ex)
    sus = sum(1 for s, e in _runs(dead & (ex["dead_reason"] != db.DEAD_PREKICKOFF))
              if pl[s:e].mean() >= 0.15)
    diag = mdl.diagnostics(lab, half)
    poss = sum(r["possessions"] for r in diag)
    return {
        "Duraklama": len(stops),
        "Ölü top payı": "%{:.1f}".format(100 * dead.sum() / graded.sum()),
        "Possession": poss,
        "Possession süresi (ortanca, 1./2. devre)": " / ".join("{:.1f} sn".format(r["median s"]) for r in diag),
        "2 sn'den kısa possession": " / ".join("%{:.0f}".format(100 * r["< 2 s"]) for r in diag),
        "Ev sahibi top payı": " / ".join("%{:.0f}".format(100 * r["H share"]) for r in diag),
        "Ölü karelerde 'oyun gibi' oranı": "%{:.1f}".format(100 * pl[dead].mean()),
        "Şüpheli duraklama (≥%15 oyun gibi)": sus,
        "Korunan duraklama (güçlü tetikleyici)": len(_runs((ex["dead_reason"] == db.DEAD_ANCHORED) & graded)),
    }


def scoreboard():
    if not os.path.isfile(SCOREBOARD):
        return []
    with open(SCOREBOARD, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keep = ("possession_v1", "possession_v0", "baseline_smooth50")
    return [r for r in rows if r.get("submission") in keep]


# ---------------------------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------------------------

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1f;--mute:#62646a;--line:#e3e3df;--accent:#2457c5;
--ok:#1f7a3f;--okbg:#e5f4ea;--bad:#b42318;--badbg:#fdecea;--warn:#8a5a00;--warnbg:#fff4dc;
--chip:#eef1f8}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121316;--card:#1b1d21;
--ink:#ececec;--mute:#a2a5ad;--line:#2c2f35;--accent:#7aa2ff;--ok:#7ad69a;--okbg:#14301f;
--bad:#ff8a80;--badbg:#3a1714;--warn:#f3c96b;--warnbg:#33290f;--chip:#252a35}}
:root[data-theme="dark"]{--bg:#121316;--card:#1b1d21;--ink:#ececec;--mute:#a2a5ad;--line:#2c2f35;
--accent:#7aa2ff;--ok:#7ad69a;--okbg:#14301f;--bad:#ff8a80;--badbg:#3a1714;--warn:#f3c96b;
--warnbg:#33290f;--chip:#252a35}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 10px}h3{font-size:16px;margin:0 0 6px}
.sub{color:var(--mute);margin:0 0 18px}.card{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:16px 18px;margin:12px 0}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:7px 8px;
border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--mute);font-weight:600}
.tc{font-variant-numeric:tabular-nums;white-space:nowrap;font-weight:600}
.fr{color:var(--mute);font-size:12px;white-space:nowrap}
.chip{display:inline-block;padding:1px 8px;border-radius:99px;background:var(--chip);font-size:12px;
white-space:nowrap}.ok{background:var(--okbg);color:var(--ok)}.bad{background:var(--badbg);color:var(--bad)}
.warn{background:var(--warnbg);color:var(--warn)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}
.kv{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.kv b{display:block;font-size:20px}.kv span{color:var(--mute);font-size:13px}
ul{margin:6px 0 6px 20px;padding:0}li{margin:2px 0}.note{color:var(--mute);font-size:13px}
.wrap{overflow-x:auto}code{background:var(--chip);padding:1px 5px;border-radius:5px;font-size:13px}
details summary{cursor:pointer;color:var(--accent);margin-top:8px}
"""


EXPECTED_TR = {"dead": "ölü top", "live": "canlı oyun", "H": "top Barcelona'da",
               "A": "top Athletic'te", "notA": "Athletic değil", "notH": "Barcelona değil"}


def since_review(r, labels):
    """(text, css): has the label in a reviewed span changed since the version that was reviewed?

    Deliberately neutral. The review says what the video shows; whether the model now 'agrees' is
    not scored here, because counting agreements would turn the team's own labels of this match
    into a test set (RULES.md). A change is only a prompt to watch the position again.
    """
    snap = load_snapshot(r["snapshot"]) if r.get("snapshot") else None
    if snap is None or r["b"] <= r["a"]:
        return "inceleme sürümü yok", ""
    changed = float((snap["labels"][r["a"]:r["b"]] != labels[r["a"]:r["b"]]).mean())
    if changed >= 0.2:
        return "incelemeden beri etiket değişti ({:.0%}) → tekrar izle".format(changed), "warn"
    return "incelemeden beri değişmedi", ""


def review_cell(revs, labels=None):
    if not revs:
        return '<span class="chip warn">kontrol gerekli</span>'
    out = []
    for r in revs:
        video = "video: " + EXPECTED_TR.get(r["expected"], r["expected"])
        chips = '<span class="chip">{} ({}–{})</span>'.format(esc(video), esc(r["start"]), esc(r["end"]))
        if labels is not None:
            text, cls = since_review(r, labels)
            chips += ' <span class="chip {}">{}</span>'.format(cls, esc(text))
        out.append('{} <span class="note">{}</span>'.format(chips, esc(r["note"])))
    return "<br>".join(out)


def diff_table(rows, reviews, snapshot, labels, now_labels=None):
    if not rows:
        return '<p class="note">Etiketi değişen pozisyon yok.</p>'
    order = {"Duraklama kaldırıldı": 0, "Duraklama eklendi": 1, "Duraklama kısaldı": 2,
             "Duraklama uzadı": 3, "Possession değişti": 4}
    rows = sorted(rows, key=lambda r: (order.get(r["kind"], 9), r["a"]))
    head = ("<tr><th>zaman</th><th>değişiklik</th><th>ayrıntı</th><th>durum</th>"
            + ("<th>şu an</th>" if now_labels is not None else "") + "</tr>")
    body = []
    for r in rows:
        now = ""
        if now_labels is not None:
            seg = now_labels[r["a"]:r["b"]]
            share = (seg == "D").mean() if len(seg) else 0
            now = "<td class='note'>ölü %{:.0f}</td>".format(100 * share)
        body.append("<tr><td><span class='tc'>{}</span><br><span class='fr'>f {}–{}</span></td>"
                    "<td>{}</td><td>{}</td><td>{}</td>{}</tr>".format(
                        tc(r["a"]), r["a"], r["b"], esc(r["kind"]), esc(r["detail"]),
                        review_cell(reviews_for(reviews, r["a"], r["b"], snapshot), labels), now))
    counts = {}
    for r in rows:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    summary = " · ".join("{} {}".format(k, v) for k, v in sorted(counts.items(), key=lambda x: order.get(x[0], 9)))
    big = len(rows) > 25
    table = "<div class='wrap'><table>{}{}</table></div>".format(head, "".join(body))
    if big:
        return "<p>{}</p><details><summary>Tüm {} pozisyonu göster</summary>{}</details>".format(
            esc(summary), len(rows), table)
    return "<p>{}</p>{}".format(esc(summary), table)


def build_html(ex, half, reviews, diffs):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    latest = CHANGELOG[-1]
    parts = ["<!doctype html><html lang='tr'><head><meta charset='utf-8'>"
             "<meta name='viewport' content='width=device-width,initial-scale=1'>"
             "<title>Model raporu</title><style>{}</style></head><body><main>".format(CSS)]
    parts.append("<h1>Model raporu — {}</h1><p class='sub'>{} · {} · son güncelleme: "
                 "<b>{}</b> ({})</p>".format(esc(MODEL), esc(GAME), esc(now), esc(latest["title"]),
                                             esc(latest["id"])))

    # 1. uncertain moments
    unc = uncertain(ex, half)
    parts.append("<h2>1. İncelenmesi önerilen anlar</h2><p class='note'>Modelin kararının diğer tarafa "
                 "en yakın kaldığı yerler, en kararsızdan başlayarak. Zamanlar videonun başından "
                 "dakika:saniye; overlay'deki kare numarasıyla eşleşir.</p>")
    rows = []
    for it in sorted(unc, key=lambda x: -x["score"]):
        revs = reviews_for(reviews, it["a"], it["b"])
        status = review_cell(revs, ex["labels"]) if revs else ""
        rows.append("<tr><td><span class='tc'>{}</span><br><span class='fr'>f {}–{}</span></td>"
                    "<td><span class='chip'>{}</span></td><td>{}</td><td class='note'>{:.0%}</td>"
                    "<td>{}</td></tr>".format(tc(it["a"]), it["a"], it["b"], esc(it["kind"]),
                                              esc(it["text"]), it["score"], status))
    parts.append("<div class='card wrap'><table><tr><th>zaman</th><th>tür</th><th>neden</th>"
                 "<th>kararsızlık</th><th>inceleme</th></tr>{}</table></div>".format("".join(rows)))

    # 2. reviewed positions
    if reviews:
        parts.append("<h2>2. Videoda incelenen pozisyonlar</h2><p class='note'>İncelemeden bu yana "
                     "etiketi değişen pozisyonlar 'tekrar izle' ile işaretlenir. Model ile inceleme "
                     "arasında doğru/yanlış sayılmaz: bu maç için insan etiketiyle sürüm seçmek "
                     "kurallara aykırı (RULES.md).</p>")
        rows = []
        for r in sorted(reviews, key=lambda r: r["a"]):
            text, cls = since_review(r, ex["labels"])
            rows.append("<tr><td><span class='tc'>{}–{}</span></td><td>{}</td><td><span class='chip {}'>"
                        "{}</span></td><td class='note'>{}</td><td class='note'>{}</td></tr>".format(
                            esc(r["start"]), esc(r["end"]),
                            EXPECTED_TR.get(r["expected"], r["expected"]), cls, esc(text),
                            esc(r["note"]), esc(r["snapshot"])))
        parts.append("<div class='card wrap'><table><tr><th>zaman</th><th>videoda</th><th>durum</th>"
                     "<th>not</th><th>incelendiği sürüm</th></tr>{}</table></div>".format("".join(rows)))

    # 3. current state
    parts.append("<h2>3. Şu anki durum (tam maç, ground truth kullanmadan)</h2><div class='grid'>")
    for k, v in current_metrics(ex, half).items():
        parts.append("<div class='kv'><b>{}</b><span>{}</span></div>".format(esc(v), esc(k)))
    parts.append("</div>")
    sb = scoreboard()
    if sb:
        parts.append("<div class='card'><h3>Halka açık 5 dakika — yalnızca pipeline kontrolü</h3>"
                     "<p class='note'>17 possession'lık pencere sıralama yapmaya yetmez; aralıklar "
                     "örtüştüğü sürece fark gürültüdür. Sabitler buna göre seçilmez.</p><div class='wrap'>"
                     "<table><tr><th>gönderim</th><th>spellF1</th><th>bootstrap 5–95</th><th>macroF1</th>"
                     "<th>possession (gerçek 17)</th></tr>")
        for r in sb:
            boot = "{:.2f}–{:.2f}".format(float(r["spell_bootstrap_low"]), float(r["spell_bootstrap_high"])) \
                if r.get("spell_bootstrap_low") else "—"
            parts.append("<tr><td>{}</td><td>{:.3f}</td><td>{}</td><td>{:.3f}</td><td>{}</td></tr>".format(
                esc(r["submission"]), float(r["spell_f1_mean"]), boot, float(r["macro_f1"]), esc(r["poss"])))
        parts.append("</table></div></div>")

    # 4. latest update
    parts.append("<h2>4. Bu güncellemede etiketi değişen pozisyonlar</h2>")
    parts.append("<p class='note'>{} — {}. Önceki sürüme göre. 'kontrol gerekli' olanları overlay'de "
                 "izleyip data/derived/rule_based/reviews.json'a ekleyebilirsin.</p>".format(
                     esc(latest["id"]), esc(latest["title"])))
    parts.append("<div class='card'>{}</div>".format(
        diff_table(diffs.get(latest["id"], []), reviews, None, ex["labels"])))

    # 5. history
    parts.append("<h2>5. Değişiklik geçmişi</h2>")
    for entry in reversed(CHANGELOG):
        parts.append("<div class='card'><h3>{} · {}</h3><p class='note'>{} · {}</p>".format(
            esc(entry["id"]), esc(entry["title"]), esc(entry["date"]),
            "etiket anlık görüntüsü var" if entry["snapshot"] else "anlık görüntü yok"))
        parts.append("<b>Ne yapıldı</b><ul>{}</ul>".format("".join("<li>{}</li>".format(esc(w)) for w in entry["what"])))
        parts.append("<b>Neden</b><p>{}</p>".format(esc(entry["why"])))
        res = list(entry["results"])
        if entry is latest:
            res = live_results(ex, half, diffs.get(entry["id"], []), reviews) + res
        if res:
            parts.append("<b>Sonuçlar</b><ul>{}</ul>".format("".join("<li>{}</li>".format(esc(r)) for r in res)))
        if entry["id"] in diffs and entry is not latest:
            parts.append("<details><summary>Etiketi değişen pozisyonlar ({})</summary>{}</details>".format(
                len(diffs[entry["id"]]),
                diff_table(diffs[entry["id"]], reviews, entry["snapshot"], ex["labels"], ex["labels"])))
        parts.append("</div>")

    parts.append("<h2>Notlar</h2><div class='card note'><ul>"
                 "<li>Video incelemeleri (data/derived/rule_based/reviews.json) nitel gözlemlerdir: hiçbir model "
                 "onları okumaz, sabitler onlara göre ayarlanmaz (RULES.md, bu maç için insan etiketi "
                 "yasak).</li>"
                 "<li>Her model güncellemesinde: data/derived/rule_based/changelog.json'a bir giriş ekle, sonra "
                 "<code>python scripts/rule_based/pipeline.py</code> çalıştır. Modeller, overlay, değerlendirme "
                 "ve bu rapor birlikte yenilenir.</li>"
                 "<li>Bu rapor data/ altında durur (gitignore): maçın anlarını ve takımın gözlemlerini "
                 "içerdiği için public repoya girmemeli.</li></ul></div>")
    parts.append("</main></body></html>")
    return "".join(parts)


def live_results(ex, half, rows, reviews):
    m = current_metrics(ex, half)
    out = ["Şu an: {} duraklama, ölü pay {}, {} possession; şüpheli duraklama {}; korunan duraklama {}.".format(
        m["Duraklama"], m["Ölü top payı"], m["Possession"], m["Şüpheli duraklama (≥%15 oyun gibi)"],
        m["Korunan duraklama (güçlü tetikleyici)"])]
    counts = {}
    for r in rows:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    if counts:
        out.append("Önceki sürüme göre: " + ", ".join("{} {}".format(k, v) for k, v in counts.items()) + ".")
    return out


def main():
    if not CHANGELOG:
        print("no {}: report skipped (submissions are already written)".format(CHANGELOG_PATH))
        return 0
    with np.load(explain_path(MODEL)) as z:
        ex = {k: z[k] for k in z.files}
    half = half_mask(GAME, len(ex["labels"]))
    reviews = load_reviews()

    # diffs: each snapshotted entry against the previous snapshotted one; newest against now
    diffs = {}
    snap_entries = [e for e in CHANGELOG if e["snapshot"]]
    latest = CHANGELOG[-1]
    for prev, cur in zip(snap_entries[:-1], snap_entries[1:]):
        old = load_snapshot(prev["snapshot"])
        new = {"labels": ex["labels"]} if cur is latest else load_snapshot(cur["snapshot"])
        if old is not None and new is not None:
            diffs[cur["id"]] = diff_labels(old["labels"], new["labels"], half)

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        handle.write(build_html(ex, half, reviews, diffs))
    if latest["snapshot"]:
        save_snapshot(latest["snapshot"], ex)
    print("wrote", OUT)
    print("  latest entry {}: {} changed positions".format(latest["id"], len(diffs.get(latest["id"], []))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
