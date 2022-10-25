import numpy as np

from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "random"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        "n_samples, n_features": [
            (1000, 2000),
        ],
        "ill_conditioned": [False, True],
    }

    def __init__(
        self, n_samples=10, n_features=50, ill_conditioned=False, random_state=27
    ):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.ill_conditioned = ill_conditioned
        self.random_state = random_state

    def get_data(self):

        rng = np.random.RandomState(self.random_state)
        X = rng.randn(self.n_samples, self.n_features)
        y = rng.randn(self.n_samples)

        if self.ill_conditioned:
            # TODO can save SVD by just having random orthonormal U and V
            U, s, VT = np.linalg.svd(X, full_matrices=False)
            X = np.dot(U * np.exp(-np.linspace(0, 8, len(s))), VT)
            # Hide b inside the the low eigenvectors of A. This forces gradient
            # descent to bounce around a lot before it can find b.
            y = U[:, -1] * 1e3

        return dict(X=X, y=y)
