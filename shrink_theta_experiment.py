import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.utils.validation import check_array
import warnings

from bayes_opt.bayesian_optimization import BayesianOptimization, Queue, TargetSpace
from bayes_opt.event import Events, DEFAULT_EVENTS
from bayes_opt.util import UtilityFunction, acq_max, ensure_rng
from sklearn.gaussian_process.kernels import Matern

from sklearn.gaussian_process.kernels import RBF

from scipy.linalg import cho_solve, solve_triangular, cholesky
from sklearn.base import clone
from sklearn.utils import check_random_state
from operator import itemgetter


def random_sampling_anchor(pbounds, anchor_num=50, random_seed=10):
    anchor_list = []
    np.random.seed(random_seed)
    for _ in range(anchor_num):
        sampling = []
        for i in range(len(pbounds)):
            sampling.append(np.random.rand() * (list(pbounds.values())[i][1] - list(pbounds.values())[i][0]))
        anchor_list.append(sampling)
    return anchor_list


def shrink_theta(pbound, kernel, linear_shrink_multiplier=0.5):
    """
    Shrink the length scale of a kernel

    :param pbound:
    :param kernel:
    :param linear_shrink_multiplier:
    :return:
    """
    xs = []

    anchor_point = random_sampling_anchor(pbound, anchor_num=10)
    total_origin_kernel = 0
    for i in anchor_point:
        for j in anchor_point:
            a = np.array([i])
            b = np.array([j])
            total_origin_kernel += kernel(a, b)
            xs.append(kernel(a, b))
    total_origin_kernel *= linear_shrink_multiplier
    if isinstance(total_origin_kernel, np.ndarray):
        total_origin_kernel = total_origin_kernel.squeeze()
    xs = np.array(xs).squeeze()
    from scipy.optimize import fsolve

    def func(theta):
        return (xs ** theta).sum() - total_origin_kernel

    root = fsolve(func, np.array([0.8]))
    print('Solving result: ', root)
    return np.sqrt(1/root[0])


class ShrinkTestBO(BayesianOptimization):

    def __init__(self, f, pbounds, shrink, mean_regressor=LinearRegression, random_state=None, verbose=2,
                 use_anchor=True, bounds_transformer=None):
        self._random_state = ensure_rng(random_state)
        self.mean_regressor = mean_regressor

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(f, pbounds, random_state)

        # queue
        self._queue = Queue()

        self._gp = GaussianProcessRegressor(
            kernel=RBF(),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        self._shrinked_gp = None

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer:
            self._bounds_transformer.initialize(self._space)

        self.result_dataframe = []
        self.error_recorder = []
        self.anchor_list = []
        self.anchor_error_recorder = []
        self.pbounds = pbounds

        if use_anchor:
            self.random_sampling_anchor()

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

        self.use_anchor = use_anchor
        self.shrink = shrink

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 xi=0.0,
                 **gp_params):
        if not self.shrink:
            kappa /= 2
        super(ShrinkTestBO, self).maximize(init_points, n_iter, acq, kappa, kappa_decay, kappa_decay_delay, xi, **gp_params)

    def suggest(self, utility_function):
        """Most promissing point to probe next"""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)

        if self.shrink:
            self._gp.kernel_.theta *= self.shrink
            self._gp.kernel_._check_bounds_params()
            K = self._gp.kernel_(self._gp.X_train_)
            K[np.diag_indices_from(K)] += self._gp.alpha
            try:
                self._gp.L_ = cholesky(K, lower=True)  # Line 2
                # self.L_ changed, self._K_inv needs to be recomputed
                self._gp._K_inv = None
            except np.linalg.LinAlgError as exc:
                exc.args = ("The kernel, %s, is not returning a "
                            "positive definite matrix. Try gradually "
                            "increasing the 'alpha' parameter of your "
                            "GaussianProcessRegressor estimator."
                            % self._gp.kernel_,) + exc.args
            self._gp.alpha_ = cho_solve((self._gp.L_, True), self._gp.y_train_)

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

    @property
    def gp(self):
        return self._gp


from problems.photocatalysis_problems import *


def shrink_theta_test(random_seed=20, max_iter=50, save_result=True, acq='ucb',
                      kappa=2.576):
    np.random.seed(random_seed)

    photocatalysis_problem, input_columns = get_kuka_problem()
    bound = {}
    for i in input_columns:
        bound[i] = (0, 5)
    photocatalysis_problem.bound = bound
    photocatalysis_problem.name = 'photocatalysis_experiment'

    path = r'C:\Users\darkn\PycharmProjects\SKBO\experiment_results'
    import time

    datetime = time.localtime()
    folder_name = '{}_{}_{}_{}_{}'.format(str(datetime.tm_year), str(datetime.tm_mon), str(datetime.tm_mday),
                                          str(datetime.tm_hour), str(datetime.tm_min))
    path = os.path.join(path, folder_name)

    result_linear_custom = pd.DataFrame([])
    error_record = pd.DataFrame([])
    anchor_error_record = pd.DataFrame([])

    result_linear_custom_shrink = pd.DataFrame([])
    error_record_shrink = pd.DataFrame([])
    anchor_error_record_shrink = pd.DataFrame([])

    iter = 0
    path = path + '_' + acq + '_' + str(kappa)
    if not os.path.exists(path):
        os.makedirs(path)
    while iter < max_iter:
        optimizer = ShrinkTestBO(f=photocatalysis_problem, pbounds=photocatalysis_problem.bound, shrink=False)
            #
        try:
            optimizer.maximize(n_iter=100, acq=acq, kappa=kappa, init_points=5)
            result_linear_custom = result_linear_custom.append(pd.Series(optimizer.result_dataframe, dtype=np.float64),
                                                               ignore_index=True)
            error_record = error_record.append(pd.Series(optimizer.error_recorder, dtype=np.float64), ignore_index=True)
            anchor_error_record = anchor_error_record.append(
                pd.Series(optimizer.anchor_error_recorder, dtype=np.float64),
                ignore_index=True)
        except:
            print('Error occured in iteration {}'.format(iter))
            pass

        kernel = optimizer.gp.kernel
        shrink = shrink_theta(pbound=photocatalysis_problem.bound, kernel=kernel)

        shrink_optimizer = ShrinkTestBO(f=photocatalysis_problem, pbounds=photocatalysis_problem.bound, shrink=shrink)

        try:
            shrink_optimizer.maximize(n_iter=100, acq=acq, kappa=kappa, init_points=5)
            result_linear_custom_shrink = result_linear_custom_shrink.append(pd.Series(shrink_optimizer.result_dataframe, dtype=np.float64),
                                                               ignore_index=True)
            error_record_shrink = error_record_shrink.append(pd.Series(shrink_optimizer.error_recorder, dtype=np.float64), ignore_index=True)
            anchor_error_record_shrink = anchor_error_record_shrink.append(
                pd.Series(shrink_optimizer.anchor_error_recorder, dtype=np.float64),
                ignore_index=True)
        except:
            print('Error occured in iteration {}'.format(iter))
            pass

        iter += 1

    if save_result:
        result_linear_custom = pd.DataFrame(np.array(result_linear_custom))
        result_linear_custom.to_csv(os.path.join(path, 'vanilla_sklearn_result{}.csv'.format(photocatalysis_problem.name)))
        error_record = pd.DataFrame(np.array(error_record))
        error_record.to_csv(os.path.join(path, 'vanilla_sklearn_error_record{}.csv'.format(photocatalysis_problem.name)))
        anchor_error_record = pd.DataFrame(np.array(anchor_error_record))
        anchor_error_record.to_csv(
            os.path.join(path, 'vanilla_sklearn_anchor_error_record{}.csv'.format(photocatalysis_problem.name)))

        result_linear_custom_shrink = pd.DataFrame(np.array(result_linear_custom_shrink))
        result_linear_custom_shrink.to_csv(
            os.path.join(path, 'shrink_test_sklearn_result{}.csv'.format(photocatalysis_problem.name)))
        error_record_shrink = pd.DataFrame(np.array(error_record_shrink))
        error_record_shrink.to_csv(
            os.path.join(path, 'shrink_test_sklearn_error_record{}.csv'.format(photocatalysis_problem.name)))
        anchor_error_record_shrink = pd.DataFrame(np.array(anchor_error_record_shrink))
        anchor_error_record_shrink.to_csv(
            os.path.join(path,
                         'shrink_test_sklearn_anchor_error_record{}.csv'.format(photocatalysis_problem.name)))


if __name__ == '__main__':
    shrink_theta_test()