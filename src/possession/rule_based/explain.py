"""Shared bookkeeping for explanation files (data/debug/<name>.explain.npz).

Every model writes, next to its submission, the per-frame quantities its decision actually used,
plus a `kind` naming the model family, so the debug overlay can show the real reason for a label.
The dead-ball part is common to all models and written here.
"""

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402
from .dead_ball import FLAG_TERMS, evidence_terms  # noqa: E402

# Where a label came from (explain["source"]). Models may use a subset.
SRC_UNGRADED, SRC_DEAD, SRC_VOTE, SRC_FILL, SRC_NO_BALL, SRC_MODEL = 0, 1, 2, 3, 4, 5


def explain_path(name):
    """Explanations live under data/ (gitignored), never next to the submission."""
    return os.path.join(DATA_DIR, "debug", name + ".explain.npz")


def dead_fields(f, llr, dead_reason):
    """The dead-ball model's per-frame state, as stored in every explanation file."""
    terms = evidence_terms(f)
    out = {"flag_" + k: terms[k] != 0 for k in FLAG_TERMS}
    out.update(llr=llr.astype(np.float32), speed=f["speed"].astype(np.float32),
               speed_all=f["speed_all"].astype(np.float32),
               speed_local=f["speed_local"].astype(np.float32),
               dead_reason=dead_reason)
    if "spot_kind" in f:
        out["spot_kind"] = f["spot_kind"]
    return out


def write_explain(name, kind, labels, fields):
    path = explain_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, kind=kind, labels=labels, **fields)
    return path


def write_submission(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path
