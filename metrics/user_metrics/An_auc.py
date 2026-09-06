import numpy as np
from scipy.stats import norm, rankdata

from .An_roc import An_roc


def An_auc(data, alpha=0.05, flag="logit", nboot=1000, return_ci=False, rng=None):
    """Canonical ``An_auc`` API with corrected, tie-aware ROC integration.

    Values labelled ``raw`` in recovered result files came from the historical
    order-dependent tied-score implementation.  Values from this function are
    the corrected provenance and differ only when scores tie.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Incorrect input size in AUC!")

    flag = "logit" if flag is None else flag.lower()
    true_positive, false_positive = An_roc(data)
    area = np.sum(
        (false_positive[1:] - false_positive[:-1])
        * (true_positive[1:] + true_positive[:-1])
    ) / 2
    if not return_ci:
        return float(area)

    m = np.sum(data[:, 0] > 0)
    n = np.sum(data[:, 0] <= 0)
    total = m + n
    z = norm.ppf(1 - alpha / 2)
    maximum_variance = np.sqrt((area * (1 - area)) / (0.75 * total - 1))

    if flag == "hanley":
        q1 = area / (2 - area)
        q2 = (2 * area**2) / (1 + area)
        ase = np.sqrt(
            (
                area * (1 - area)
                + (m - 1) * (q1 - area**2)
                + (n - 1) * (q2 - area**2)
            )
            / (m * n)
        )
        interval = np.array([area - z * ase, area + z * ase])
    elif flag == "maxvar":
        ase = np.sqrt((area * (1 - area)) / min(m, n))
        interval = np.array([area - z * ase, area + z * ase])
    elif flag in {"mann-whitney", "logit"}:
        x = data[data[:, 0] <= 0, 1]
        y = data[data[:, 0] > 0, 1]
        m, n = len(x), len(y)
        ranks = rankdata(np.r_[np.sort(x), np.sort(y)], method="average")
        negative_ranks, positive_ranks = ranks[:m], ranks[m:]
        s102 = (
            np.sum((negative_ranks - np.arange(1, m + 1)) ** 2)
            - m * (np.mean(negative_ranks) - (m + 1) / 2) ** 2
        ) / ((m - 1) * n**2)
        s012 = (
            np.sum((positive_ranks - np.arange(1, n + 1)) ** 2)
            - n * (np.mean(positive_ranks) - (n + 1) / 2) ** 2
        ) / ((n - 1) * m**2)
        ase = np.sqrt(((m + n) * (m * s012 + n * s102) / (m + n)) / (m * n))
        if flag == "logit":
            low, high = np.log(area / (1 - area)) + np.array([-1, 1]) * z * ase / (
                area * (1 - area)
            )
            interval = np.exp([low, high]) / (1 + np.exp([low, high]))
        else:
            interval = np.array([area - z * ase, area + z * ase])
    elif flag == "wald":
        interval = np.array([area - z * maximum_variance, area + z * maximum_variance])
    elif flag == "wald-cc":
        interval = np.array(
            [
                area - (z * maximum_variance + 1 / (2 * total)),
                area + (z * maximum_variance + 1 / (2 * total)),
            ]
        )
    elif flag == "boot":
        generator = np.random.default_rng(rng)
        samples = np.array(
            [An_auc(data[generator.integers(0, total, total)]) for _ in range(nboot)]
        )
        interval = np.percentile(samples, 100 * np.array([alpha / 2, 1 - alpha / 2]))
    else:
        raise ValueError("Bad FLAG for AUC!")
    return float(area), interval
