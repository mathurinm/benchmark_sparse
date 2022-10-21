from benchopt import BaseSolver, safe_import_context


with safe_import_context() as ctx:
    import numpy as np
    from scipy.optimize import least_squares


class Solver(BaseSolver):
    """Reparametrization with scipy least squares."""
    name = 'reparam'

    def set_objective(self, X, y, lmbd):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.X, self.y = X, y
        self.lmbd = lmbd

    def run(self, n_iter):
        X, y, lmbd = self.X, self.y, self.lmbd

        # least_squares will minimize norm of func(w)
        def func(w):
            return np.hstack((
                (X.dot(w**3) - y) / np.sqrt(2),
                np.sqrt(lmbd) * w,
            ))

        def dfunc(w):
            return np.vstack((
                (3 * X * w[None, :]**2) / np.sqrt(2),
                np.sqrt(lmbd) * np.eye(len(w)),
            ))

        x0 = np.linalg.lstsq(X, y, rcond=None)[0]
        results = least_squares(
            func, np.abs(x0)**(1/3) * np.sign(x0), jac=dfunc,
            method='lm',
            max_nfev=n_iter + 1,
            verbose=1,
        )

        self.w = results.x ** 3

    def get_result(self):
        return self.w
