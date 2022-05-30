from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.utils.validation import check_array
import warnings

from bayes_opt.bayesian_optimization import BayesianOptimization, Queue, TargetSpace
from bayes_opt.event import Events, DEFAULT_EVENTS
from bayes_opt.util import UtilityFunction, acq_max, ensure_rng
from sklearn.gaussian_process.kernels import Matern

# TODO: using RBF kernel for test
from sklearn.gaussian_process.kernels import RBF

import numpy as np
from scipy.linalg import cho_solve, solve_triangular, cholesky
from sklearn.base import clone
from sklearn.utils import check_random_state
from operator import itemgetter


class SurrogateKernelBayesianOptimization(BayesianOptimization):

    def __init__(self, f, pbounds, custom_list=None, random_state=None, verbose=2,
                 bounds_transformer=None, for_comparison=False):
        self._random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(f, pbounds, random_state)

        # queue
        self._queue = Queue()

        # Internal GP regressor
        '''self._gp = SurrogateKernelGPR(
            custom_list=custom_list,
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        self._original_gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )'''

        # TODO: test script

        self._gp = SurrogateKernelGPR(
            custom_list=custom_list,
            kernel=RBF(),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        self._original_gp = GaussianProcessRegressor(
            kernel=RBF(),
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
        self.for_comparison = for_comparison

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

    def suggest(self, utility_function):
        """Most promissing point to probe next"""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)
            self._original_gp.fit(self._space.params, self._space.target)
            if not self.for_comparison:
                self._gp.kernel_ = self._original_gp.kernel_
                self._gp.kernel = self._original_gp.kernel
                self._gp.L_ = self._original_gp.L_

        # Finding argmax of the acquisition function.
        suggestion = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state
        )

        # todo: debug code, delete
        mean, std = self._gp.predict(suggestion.reshape(1, -1), return_std=True)
        print('Mean: {}, Std: {}'.format(mean[0], std[0]))

        return self._space.array_to_params(suggestion)


class SurrogateKernelGPR(GaussianProcessRegressor):

    def __init__(self, custom_list=None, kernel=None, *, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None):
        super(SurrogateKernelGPR, self).__init__(kernel=kernel, alpha=alpha, optimizer=optimizer,
                                                 n_restarts_optimizer=n_restarts_optimizer, normalize_y=normalize_y,
                                                 copy_X_train=copy_X_train, random_state=random_state)
        self.linear_regressor = LinearRegression()
        self.custom_list = custom_list

    def fit(self, X, y):

        if self.custom_list is None:
            self.linear_regressor.fit(X, y)
            # todo: should delete this to get good results
            y = y - self.linear_regressor.predict(X)
            # y = y - self.linear_regressor.predict(X) * np.random.normal(np.ones(y.shape), 0.2)  # adding noise
        else:
            self.linear_regressor.fit(X[:, self.custom_list], y)
            # todo: should delete this to get good results
            y = y - self.linear_regressor.predict(X[:, self.custom_list])

        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = C(1.0, constant_value_bounds="fixed") \
                           * RBF(1.0, length_scale_bounds="fixed")
        else:
            self.kernel_ = clone(self.kernel)

        self._rng = check_random_state(self.random_state)

        if self.kernel_.requires_vector_input:
            X, y = self._validate_data(X, y, multi_output=True, y_numeric=True,
                                       ensure_2d=True, dtype="numeric")
        else:
            X, y = self._validate_data(X, y, multi_output=True, y_numeric=True,
                                       ensure_2d=False, dtype=None)

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = np.std(y, axis=0)

            # Remove mean and make unit variance
            y = (y - self._y_train_mean) / self._y_train_std

        else:
            self._y_train_mean = np.zeros(1)
            self._y_train_std = 1

        if np.iterable(self.alpha) \
                and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError("alpha must be a scalar or an array"
                                 " with same number of entries as y.(%d != %d)"
                                 % (self.alpha.shape[0], y.shape[0]))

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True, clone_kernel=False)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta,
                                                         clone_kernel=False)

            # First optimize starting from theta specified in kernel
            optima = [(self._constrained_optimization(obj_func,
                                                      self.kernel_.theta,
                                                      self.kernel_.bounds))]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = \
                        self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial,
                                                       bounds))
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.kernel_._check_bounds_params()

            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel_.theta,
                                             clone_kernel=False)

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
            # self.L_ changed, self._K_inv needs to be recomputed
            self._K_inv = None
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)  # Line 3
        return self

    def predict(self, X, return_std=False, return_cov=False):

        if self.custom_list is None:
            y_linear_predict = self.linear_regressor.predict(X)
        else:
            y_linear_predict = self.linear_regressor.predict(X[:, self.custom_list])

        # adding some manual error

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
            elif return_std:
                y_var = kernel.diag(X)
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean
        else:  # Predict based on GP posterior

            # todo: for test
            # original code:

            K_trans = self.kernel_(X, self.X_train_)
            y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)

            # undo normalisation
            y_mean = self._y_train_std * y_mean + self._y_train_mean

            y_mean += y_linear_predict

            # new code:

            # K_trans = self.kernel_(X, self.X_train_)
            # alpha = cho_solve((self.L_, True), self.y_train_ - self.linear_regressor.predict(self.X_train_))
            # y_mean = (y_linear_predict + K_trans.dot(alpha)) * 0.5 + (self._y_train_std * K_trans.dot(self.alpha_) + self._y_train_mean) * 0.5
            # y_mean = y_linear_predict + (self._y_train_std * K_trans.dot(self.alpha_) + self._y_train_mean)
            # y_mean = self._y_train_std * K_trans.dot(self.alpha_) + self._y_train_mean

            if return_cov:
                v = cho_solve((self.L_, True), K_trans.T)  # Line 5
                y_cov = self.kernel_(X) - K_trans.dot(v)  # Line 6

                # undo normalisation
                y_cov = y_cov * self._y_train_std ** 2

                return y_mean, y_cov
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
                y_var = y_var * self._y_train_std ** 2

                return y_mean, np.sqrt(y_var)
            else:
                return y_mean
