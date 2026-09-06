import numpy as np


def An_roc(data):
    """Return ROC points, aggregating every equal-score block atomically.

    Corrected provenance: the recovered canonical implementation retained the
    *first* observation of each tied block, so its historical/raw AUC depended
    on row order.  Keeping the block's final cumulative point gives every tie
    the standard half credit and makes the result order invariant.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Incorrect input size in ROC!")

    target = data[:, 0] > 0
    scores = data[:, 1]
    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    target = target[order]

    true_positive = np.cumsum(target) / np.sum(target)
    false_positive = np.cumsum(~target) / np.sum(~target)
    block_ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    return np.r_[0, true_positive[block_ends]], np.r_[0, false_positive[block_ends]]
