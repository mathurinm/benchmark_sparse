from benchopt import BaseObjective, safe_import_context

# Protect import to allow manipulating objective without importing library
# Useful for autocompletion and install commands
with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    name = "Ordinary Least Squares"

    # All parameters 'p' defined here are available as 'self.p'
    parameters = {
        'reg': [0.5, 0.1, 0.01],
        # 'reg': [0.5],
    }

    def get_one_solution(self):
        # Return one solution. This should be compatible with 'self.compute'.
        return np.zeros(self.X.shape[1])

    def set_data(self, X, y):
        # The keyword arguments of this function are the keys of the `data`
        # dict in the `get_data` function of the dataset.
        # They are customizable.
        self.X, self.y = X, y
        self.lmbd = self.reg * np.linalg.norm(X.T @ y, ord=np.inf)

    def compute(self, beta):
        # The arguments of this function are the outputs of the
        # `get_result` method of the solver.
        # They are customizable.
        diff = self.y - self.X.dot(beta)
        return .5 * diff.dot(diff) + self.lmbd * np.sum(np.abs(beta) ** (2 / 3))

    def to_dict(self):
        # The output of this function are the keyword arguments
        # for the `set_objective` method of the solver.
        # They are customizable.
        return dict(X=self.X, y=self.y, lmbd=self.lmbd)
