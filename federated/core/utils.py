import numpy as np
from sklearn.utils import check_random_state


def IID_partitioner(X, n_clients: int, rng=None):
    if n_clients <= 0:
        raise ValueError("n_clients must be a positive integer.")

    X = np.asarray(X)

    n = X.shape[0]
    if n_clients > n:
        raise ValueError("n_clients cannot be greater than the number of samples in X.")

    rng = check_random_state(rng)
    rng.shuffle(X)

    return np.array_split(X, n_clients)
