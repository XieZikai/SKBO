from bayes_opt.util import UtilityFunction, acq_max, ensure_rng
from bayes_opt.event import Events, DEFAULT_EVENTS

from bayes_opt.bayesian_optimization import BayesianOptimization, Queue, TargetSpace
from sklearn.gaussian_process.kernels import Matern, RBF, DotProduct, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils.validation import check_array
import numpy as np
from scipy.linalg import cho_solve, solve_triangular, cholesky
import warnings


class CKBO(BayesianOptimization):
    def __init__(self, f, pbounds, random_state=None, verbose=2,
                 bounds_transformer=None, use_anchor=False):
        """"""
        self._random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(f, pbounds, random_state)

        # queue
        self._queue = Queue()

        kernel = C() * Matern(nu=2.5) + C() + DotProduct()

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
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

