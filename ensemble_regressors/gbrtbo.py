from sklearn.ensemble import RandomForestRegressor

from bayes_opt.util import ensure_rng
from bayes_opt.event import DEFAULT_EVENTS

from bayes_opt.bayesian_optimization import BayesianOptimization, Queue, TargetSpace
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils.validation import check_array
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
import warnings

import skopt

class GBRTBO(BayesianOptimization):
    def __init__(self, f, pbounds, random_state=None, verbose=2,
                 bounds_transformer=None, use_anchor=False):
        """"""
        self._random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(f, pbounds, random_state)

        # queue
        self._queue = Queue()
        kernel_length_scale = np.array([1.0 for _ in range(len(pbounds))])

        # Internal GP regressor
        self._gp = GBRTGPR(
            kernel=Matern(nu=2.5, length_scale=kernel_length_scale),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer:
            self._bounds_transformer.initialize(self._space)
        self.result_dataframe = []

        self.error_recorder = []
        self.anchor_list = []
        self.anchor_error_recorder = []
        self.pbounds = pbounds
        self.use_anchor = use_anchor

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)


class GBRTGPR(GaussianProcessRegressor):
    def __init__(self, kernel=None, *, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None):
        super(GBRTGPR, self).__init__(kernel=kernel, alpha=alpha, optimizer=optimizer,
                                    n_restarts_optimizer=n_restarts_optimizer, normalize_y=normalize_y,
                                    copy_X_train=copy_X_train, random_state=random_state)
        self.mean_regressor = GradientBoostingRegressor(n_estimators=30)

    def fit(self, X, y):

        self.mean_regressor.fit(X, y)
        super(GBRTGPR, self).fit(X, y)

    def predict(self, X, return_std=False, return_cov=False):

        y_pred_regressor = self.mean_regressor.predict(X)

        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")

        if self.kernel is None or self.kernel.requires_vector_input:
            X = check_array(X, ensure_2d=True, dtype="numeric")
        else:
            X = check_array(X, ensure_2d=False, dtype=None)

        if not hasattr(self, "X_train_"):  # Unfitted;predict based on GP prior
            if self.kernel is None:
                kernel = (C(1.0, constant_value_bounds="fixed") *
                          RBF(1.0, length_scale_bounds="fixed"))
            else:
                kernel = self.kernel
            y_mean = np.zeros(X.shape[0])
            if return_cov:
                y_cov = kernel(X)
                return y_mean + y_pred_regressor, y_cov
                # return y_pred_regressor, y_cov
            elif return_std:
                y_var = kernel.diag(X)
                return y_mean + y_pred_regressor, np.sqrt(y_var)
                # return y_pred_regressor, np.sqrt(y_var)
            else:
                return y_mean + y_pred_regressor
                # return y_pred_regressor
        else:  # Predict based on GP posterior
            K_trans = self.kernel_(X, self.X_train_)
            y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)

            # undo normalisation
            y_mean = self._y_train_std * y_mean + self._y_train_mean

            if return_cov:
                v = cho_solve((self.L_, True), K_trans.T)  # Line 5
                y_cov = self.kernel_(X) - K_trans.dot(v)  # Line 6

                # undo normalisation
                y_cov = y_cov * self._y_train_std**2

                return y_mean + y_pred_regressor, y_cov
                # return y_pred_regressor, y_cov
            elif return_std:
                # cache result of K_inv computation
                if self._K_inv is None:
                    # compute inverse K_inv of K based on its Cholesky
                    # decomposition L and its inverse L_inv
                    L_inv = solve_triangular(self.L_.T,
                                             np.eye(self.L_.shape[0]))
                    self._K_inv = L_inv.dot(L_inv.T)

                # Compute variance of predictive distribution
                y_var = self.kernel_.diag(X)
                y_var -= np.einsum("ij,ij->i",
                                   np.dot(K_trans, self._K_inv), K_trans)

                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    warnings.warn("Predicted variances smaller than 0. "
                                  "Setting those variances to 0.")
                    y_var[y_var_negative] = 0.0

                # undo normalisation
                y_var = y_var * self._y_train_std**2

                return y_mean + y_pred_regressor, np.sqrt(y_var)
                # return y_pred_regressor, np.sqrt(y_var)
            else:
                return y_mean + y_pred_regressor
                # return y_pred_regressor
