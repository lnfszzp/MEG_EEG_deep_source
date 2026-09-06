import numpy as np


def RMSE(S, S0):
    """Canonical normalized squared Frobenius error (historically named RMSE)."""
    source = np.asarray(S, dtype=float)
    truth = np.asarray(S0, dtype=float)
    source = source / np.linalg.norm(source, "fro")
    truth = truth / np.linalg.norm(truth, "fro")
    return (np.linalg.norm(source - truth, "fro") / np.linalg.norm(truth, "fro")) ** 2
