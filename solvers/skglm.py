from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as ctx:
    import skglm


class Solver(BaseSolver):
    name = 'skglm'

    requirements = ['pip:skglm']

    stopping_criterion = SufficientProgressCriterion(
            eps=1e-12, patience=15, strategy='iteration')

    def set_objective(self, X, y, lmbd):
        self.X, self.y = X, y
        self.lmbd = lmbd

        self.run(2)

    def run(self, n_iter):
        model = skglm.GeneralizedLinearEstimator(
            datafit=skglm.datafits.Quadratic(),
            penalty=skglm.penalties.L2_3(self.lmbd / len(self.y)),
            solver=skglm.solvers.AndersonCD(
                tol=1e-14,
                max_iter=n_iter,
                verbose=False,
                fit_intercept=False,
                ws_strategy="fixpoint"
            ),
        )
        model.fit(self.X, self.y)

        self.w = model.coef_.copy()

    def get_result(self):
        return self.w

    @staticmethod
    def get_next(n_iter):
        return n_iter + 1
