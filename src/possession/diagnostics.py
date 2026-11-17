"""Aggregate-only diagnostics; no per-frame or match-derived rows are emitted."""
import numpy as np


def summarize(labels, fps=25.0, quality=None):
    labels = np.asarray(labels)
    if labels.ndim != 1 or fps <= 0 or not np.isin(labels, ['H', 'A', 'D', 'X']).all():
        raise ValueError('expected one-dimensional H/A/D/X labels and positive fps')
    counts = {label: int((labels == label).sum()) for label in 'HADX'}
    transitions = int(np.count_nonzero(labels[1:] != labels[:-1]))
    runs = {label: int(np.count_nonzero((labels == label) & np.r_[True, labels[1:] != labels[:-1]]))
            for label in 'HADX'} if labels.size else {label: 0 for label in 'HADX'}
    result = {'frames': int(labels.size), 'counts': counts,
              'seconds': {k: v / fps for k, v in counts.items()},
              'runs': runs, 'transitions': transitions}
    if quality is not None:
        result['quality_counts'] = {key: int(np.count_nonzero(value))
                                    for key, value in quality.items()}
    return result
